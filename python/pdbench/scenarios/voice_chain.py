"""음성 명령 체인: 실시간 오디오 → Silero VAD → (발화 구간) → Vosk 명령 인식 + ECAPA 화자 확인 → 판정.

스레드 구성 (실제 로봇과 같게)
  audio 스레드  : 32ms 청크마다 VAD. 발화 중인 청크를 큐로 넘김 (무거운 일은 하지 않음 → 실시간 유지)
  speech 스레드 : 큐에서 받아 Vosk 에 계속 흘려 넣음(스트리밍). VAD 'end' 가 오면
                  Vosk 최종 결과 → ECAPA 임베딩 → 등록 화자와 코사인 → 판정

지연 정의 (정답 라벨 = build_command_stream.py 의 발화 끝 샘플)
  e2e        = 판정 시각 - 정답 발화 끝이 '녹음된' 시각       ← 기준: 발화 종료 → 명령 판정 ≤ 1 s
  vad_end    = VAD 'end' 판정 시각 - 정답 발화 끝 시각        (min_silence 대기 포함)
  queue_wait = speech 스레드가 'end' 를 꺼낸 시각 - VAD 'end' 시각 (밀림)
  vosk_final / ecapa = 각 단계 처리 시간
voice="vad" 이면 VAD 만 실행 (이전 시나리오 B 와 동일).
"""
import collections
import csv
import queue
import wave
from pathlib import Path

import numpy as np

from ..common import config
from ..common.sysinfo import sha256
from ..common.timing import describe, ms, now_ns
from ..models.silero_vad import SileroVAD, VadSegmenter
from ..sources.wav_stream import RealtimeWavSource


def _read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class VoiceChain:
    def __init__(self, vad_threads: int, full: bool, wav: Path | None = None, voice_set: Path | None = None):
        self.vad_cfg = config.load("vad")
        self.full = full
        self.vad_path = config.repo_path(self.vad_cfg["model"]["path"])
        self.vad = SileroVAD(self.vad_path, self.vad_cfg, vad_threads)
        self.vad_threads = vad_threads
        self.seg = VadSegmenter(self.vad_cfg, self.vad.sr)
        self.chunk = self.vad.chunk
        self.audio_rows: list[list] = []
        self.events: list[list] = []
        self.decisions: list[dict] = []
        if not full:
            self.wav_path = Path(wav)
            self.src = RealtimeWavSource(self.wav_path, self.chunk, self.vad.sr)
            return

        vc = config.load("voice")
        self.voice_cfg = vc
        ds = config.repo_path(voice_set or vc["pipeline"]["dataset"])
        self.dataset = ds
        self.wav_path = ds / "stream.wav"
        self.src = RealtimeWavSource(self.wav_path, self.chunk, self.vad.sr)
        self.period = (len(self.src.audio) // self.chunk) * self.chunk
        with open(ds / "stream_labels.csv", encoding="utf-8") as f:
            self.labels = [dict(r, start_sample=int(r["start_sample"]), end_sample=int(r["end_sample"]))
                           for r in csv.DictReader(f)]

        from ..models.ecapa import Ecapa
        from ..models.vosk_cmd import VoskCommand, load_commands
        self.commands = load_commands(vc)
        t = now_ns()
        self.vosk = VoskCommand(config.repo_path(vc["vosk"]["model_dir"]), self.commands, vc["vosk"]["fillers"],
                                self.vad.sr)
        self.vosk_load_ms = ms(now_ns() - t)
        e = vc["ecapa"]
        self.ecapa_path = config.repo_path(e["model"])
        t = now_ns()
        self.ecapa = Ecapa(self.ecapa_path, config.repo_path(e["fbank"]), e["ort"])
        self.ecapa_load_ms = ms(now_ns() - t)
        self.spk_threshold = e["threshold"]
        self.min_seg = int(e["min_segment_ms"] * self.vad.sr / 1000)
        enroll_files = sorted((ds / "enroll").glob("*.wav"))
        if not enroll_files:
            raise FileNotFoundError(f"{ds}/enroll/*.wav 없음 — 등록 화자 음성이 필요")
        t = now_ns()
        embs = np.stack([self.ecapa.embed(_read_wav(p)) for p in enroll_files])
        m = embs.mean(axis=0)
        self.enrolled = m / np.linalg.norm(m)
        self.enroll_ms = ms(now_ns() - t)
        self.n_enroll = len(enroll_files)
        self.q: queue.Queue = queue.Queue()
        self.preroll = collections.deque(maxlen=max(1, round(vc["pipeline"]["preroll_ms"] / 1000 * self.vad.sr / self.chunk)))

    def warmup(self, n_chunks: int) -> None:
        for _ in range(n_chunks):
            self.vad(np.zeros(self.chunk, np.float32))
        self.vad.reset()
        if self.full:
            self.ecapa.embed(np.zeros(self.vad.sr, np.float32) + 1e-4)
            self.vosk.accept(np.zeros(self.vad.sr, np.float32))
            self.vosk.final()
            self.vosk.reset()

    # ---------------- 스레드 본체 ----------------
    def audio_worker(self, stop, t0: int) -> None:
        for ch in self.src:
            if stop.is_set():
                break
            t_start = now_ns()
            prob = self.vad(ch.samples)
            t_done = now_ns()
            was_triggered = self.seg.triggered
            ev = self.seg.update(prob, ch.start_sample, ch.start_sample + len(ch.samples))
            self.audio_rows.append([ch.index, (t_done - t0) / 1e9, ms(t_done - t_start),
                                    ms(t_done - ch.arrival_ns), round(prob, 4)])
            if ev:
                kind, sample = ev
                self.events.append([kind, sample, round((t_done - t0) / 1e9, 4),
                                    round(ms(t_done - self.src.sample_time_ns(sample)), 2)])
            if self.full:
                if ev and ev[0] == "start":
                    pre = list(self.preroll)
                    self.q.put(("start", pre + [ch.samples], ch.start_sample - len(pre) * self.chunk, ev[1]))
                elif was_triggered:
                    self.q.put(("audio", ch.samples))
                if ev and ev[0] == "end":
                    self.q.put(("end", ev[1], t_done))
                self.preroll.append(ch.samples)
        if self.full:
            self.q.put(None)

    def speech_worker(self, t0: int) -> None:
        buf: list[np.ndarray] = []
        buf_start = vad_start = 0
        accept_ns = 0
        while True:
            item = self.q.get()
            if item is None:
                return
            kind = item[0]
            if kind == "start":
                _, chunks, buf_start, vad_start = item
                self.vosk.reset()
                buf, accept_ns = list(chunks), 0
                t = now_ns()
                for c in chunks:
                    self.vosk.accept(c)
                accept_ns += now_ns() - t
            elif kind == "audio":
                t = now_ns()
                self.vosk.accept(item[1])
                accept_ns += now_ns() - t
                buf.append(item[1])
            elif kind == "end":
                _, end_sample, t_vad = item
                t_pick = now_ns()
                vr = self.vosk.final()
                t_v = now_ns()
                x = np.concatenate(buf)[: max(0, end_sample - buf_start)] if buf else np.zeros(0, np.float32)
                score = None
                if len(x) >= self.min_seg:
                    score = float(self.ecapa.embed(x) @ self.enrolled)
                t_e = now_ns()
                accepted, reject = False, vr.reject
                if vr.cmd != "none":
                    thr = self.commands[vr.cmd]["spk_threshold"]
                    accepted = score is not None and score >= thr
                    if not accepted:
                        reject = "short" if score is None else "speaker"
                self.decisions.append({
                    "vad_start": vad_start, "vad_end": end_sample, "seg_s": len(x) / self.vad.sr,
                    "cmd": vr.cmd, "raw_cmd": vr.raw_cmd, "reject": reject, "text": vr.text,
                    "cmd_conf": vr.conf, "spk_score": score, "accepted": accepted,
                    "t_vad_ns": t_vad, "t_decision_ns": t_e,
                    "queue_wait_ms": ms(t_pick - t_vad), "vosk_final_ms": ms(t_v - t_pick),
                    "ecapa_ms": ms(t_e - t_v), "vosk_accept_total_ms": ms(accept_ns),
                })

    # ---------------- 결과 ----------------
    def _match(self) -> None:
        """판정 ↔ 정답 라벨 매칭 (스트림 반복 재생 고려). 겹침이 가장 큰 라벨."""
        for d in self.decisions:
            p = d["vad_end"] // self.period
            s, e = d["vad_start"] - p * self.period, d["vad_end"] - p * self.period
            best, best_ov = None, 0
            for lab in self.labels:
                ov = min(e, lab["end_sample"]) - max(s, lab["start_sample"])
                if ov > best_ov:
                    best, best_ov = lab, ov
            d["label"] = best
            d["pass"] = int(p)
            if best:
                t_end = self.src.sample_time_ns(p * self.period + best["end_sample"])
                d["e2e_ms"] = ms(d["t_decision_ns"] - t_end)
                d["vad_end_ms"] = ms(d["t_vad_ns"] - t_end)

    def save(self, out_dir: Path, t0: int) -> None:
        with open(out_dir / "audio.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chunk", "t_done_s", "infer_ms", "lag_ms", "prob"])
            w.writerows([[r[0], f"{r[1]:.4f}", f"{r[2]:.3f}", f"{r[3]:.3f}", r[4]] for r in self.audio_rows])
        with open(out_dir / "events.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "sample", "t_decision_s", "decision_latency_ms"])
            w.writerows(self.events)
        if not self.full:
            return
        self._match()
        cols = ["t_decision_s", "pass", "utt", "speaker", "label_cmd", "cmd", "accepted", "raw_cmd", "reject",
                "spk_score", "cmd_conf", "text", "seg_s", "e2e_ms", "vad_end_ms", "queue_wait_ms", "vosk_final_ms", "ecapa_ms",
                "vosk_accept_total_ms"]
        with open(out_dir / "voice.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for d in self.decisions:
                lab = d.get("label") or {}
                w.writerow([round((d["t_decision_ns"] - t0) / 1e9, 3), d.get("pass"), lab.get("utt", ""),
                            lab.get("speaker", ""), lab.get("command", ""), d["cmd"], int(d["accepted"]),
                            d["raw_cmd"], d["reject"] or "",
                            "" if d["spk_score"] is None else round(d["spk_score"], 4),
                            "" if d["cmd_conf"] is None else round(d["cmd_conf"], 3), d["text"], round(d["seg_s"], 3),
                            *(("" if d.get(k) is None else round(d[k], 2)) for k in
                              ("e2e_ms", "vad_end_ms", "queue_wait_ms", "vosk_final_ms", "ecapa_ms",
                               "vosk_accept_total_ms"))])

    def summary(self) -> dict:
        chunk_ms = self.chunk / self.vad.sr * 1000
        lag = np.asarray([r[3] for r in self.audio_rows])
        ends = [e[3] for e in self.events if e[0] == "end"]
        out = {
            "mode": "full" if self.full else "vad",
            "wav": self.wav_path.name,
            "vad": {
                "model": self.vad_path.name, "model_sha256": sha256(self.vad_path), "threads": self.vad_threads,
                "chunks": len(self.audio_rows), "chunk_ms": chunk_ms,
                "infer_ms": describe([r[2] for r in self.audio_rows]), "lag_ms": describe(lag),
                "late_chunks_pct": float(100 * np.mean(lag > chunk_ms)) if lag.size else None,
                "speech_segments": len(ends), "end_decision_ms": describe(ends),
                "min_silence_ms": self.vad_cfg["segmenter"]["min_silence_ms"],
            },
        }
        if not self.full:
            return out
        ds = self.decisions
        matched = [d for d in ds if d.get("label")]
        # 실행 중 끝까지 재생된 라벨만 '놓침' 계산 대상
        fed = len(self.audio_rows) * self.chunk
        expected = [(p, lab["utt"]) for p in range(fed // self.period + 1) for lab in self.labels
                    if p * self.period + lab["end_sample"] + self.vad.sr < fed]
        hit = {(d["pass"], d["label"]["utt"]) for d in matched}
        is_cmd = lambda lab: lab["speaker"] == "target" and lab["command"] != "none"  # noqa: E731
        cmd_utts = [d for d in matched if is_cmd(d["label"])]
        neg_utts = [d for d in ds if not d.get("label") or not is_cmd(d["label"])]
        tgt = [d["spk_score"] for d in matched if d["label"]["speaker"] == "target" and d["spk_score"] is not None]
        oth = [d["spk_score"] for d in matched if d["label"]["speaker"] != "target" and d["spk_score"] is not None]
        e2e_cmd = [d["e2e_ms"] for d in cmd_utts]
        vc = self.voice_cfg
        out["voice"] = {
            "dataset": self.dataset.name, "commands": self.commands, "spk_threshold": self.spk_threshold,
            "vosk_model": Path(vc["vosk"]["model_dir"]).name, "ecapa_model": self.ecapa_path.name,
            "ecapa_sha256": sha256(self.ecapa_path), "n_enroll": self.n_enroll, "enroll_ms": self.enroll_ms,
            "vosk_load_ms": self.vosk_load_ms, "ecapa_load_ms": self.ecapa_load_ms,
            "segments": len(ds), "matched": len(matched), "spurious": len(ds) - len(matched),
            "expected_utts": len(expected), "missed_utts": len([k for k in expected if k not in hit]),
            "e2e_ms": describe([d["e2e_ms"] for d in matched]),
            "e2e_cmd_ms": describe(e2e_cmd),
            "vad_end_ms": describe([d["vad_end_ms"] for d in matched]),
            "queue_wait_ms": describe([d["queue_wait_ms"] for d in ds]),
            "vosk_final_ms": describe([d["vosk_final_ms"] for d in ds]),
            "ecapa_ms": describe([d["ecapa_ms"] for d in ds]),
            "vosk_accept_total_ms": describe([d["vosk_accept_total_ms"] for d in ds]),
            "cmd_utts": len(cmd_utts),
            "cmd_correct": sum(d["cmd"] == d["label"]["command"] for d in cmd_utts),
            "cmd_accepted_correct": sum(d["accepted"] and d["cmd"] == d["label"]["command"] for d in cmd_utts),
            "neg_utts": len(neg_utts),
            "false_accepts": sum(d["accepted"] for d in neg_utts),
            "spk_score_target": describe(tgt), "spk_score_other": describe(oth),
            "per_command": self._per_command(ds, is_cmd),
            "critical_false_accepts": [self._brief(d) for d in neg_utts
                                       if d["accepted"] and self.commands[d["cmd"]]["critical"]],
            "critical_strict_rejects": [self._brief(d) for d in ds
                                        if not d["accepted"] and d["raw_cmd"] != "none"
                                        and self.commands[d["raw_cmd"]]["critical"]],
        }
        return out

    @staticmethod
    def _brief(d: dict) -> dict:
        lab = d.get("label") or {}
        return {"utt": lab.get("utt"), "file": lab.get("file"), "speaker": lab.get("speaker"),
                "label_cmd": lab.get("command"), "cmd": d["cmd"], "raw_cmd": d["raw_cmd"], "reject": d["reject"],
                "text": d["text"], "cmd_conf": d["cmd_conf"], "spk_score": d["spk_score"]}

    def _per_command(self, ds: list[dict], is_cmd) -> dict:
        """명령별: 정답 발화 수, 정확히 수락, 놓침(거부 이유별), 다른 명령으로 오인,
        이 명령으로 잘못 수락 (다른 명령 발화에서 / 명령 아닌 말·다른 화자에서)."""
        out = {}
        for key, c in self.commands.items():
            mine = [d for d in ds if d.get("label") and is_cmd(d["label"]) and d["label"]["command"] == key]
            into = [d for d in ds if d["accepted"] and d["cmd"] == key
                    and not (d.get("label") and is_cmd(d["label"]) and d["label"]["command"] == key)]
            from_noncmd = [d for d in into if not (d.get("label") and is_cmd(d["label"]))]
            missed = [d for d in mine if not d["accepted"]]
            reasons: dict[str, int] = {}
            for d in missed:
                r = d["reject"] or ("no_match" if d["raw_cmd"] == "none" else "other")
                reasons[r] = reasons.get(r, 0) + 1
            out[key] = {
                "phrase": c["phrase"], "critical": c["critical"], "group": c["group"],
                "rules": {"min_word_conf": c["min_word_conf"], "spk_threshold": c["spk_threshold"], "exact": c["exact"]},
                "utts": len(mine),
                "correct": sum(d["accepted"] and d["cmd"] == key for d in mine),
                "confused": sum(d["accepted"] and d["cmd"] != key for d in mine),
                "missed": len(missed), "missed_reasons": reasons,
                "false_from_other_cmd": len(into) - len(from_noncmd),
                "false_from_noncmd": len(from_noncmd),
            }
        return out
