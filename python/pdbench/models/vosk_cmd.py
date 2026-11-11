"""Vosk(Kaldi) 한국어 small 모델로 제한 문법 명령 인식. ONNX 가 아니므로 ORT 설정/최적화 단계 대상 아님.

명령별 규칙 (configs/voice.toml [commands.<키>]):
  min_word_conf : 명령 단어 신뢰도 기준 (없으면 [vosk] 기본값)
  exact         : true 면 인식 결과가 그 명령 단어(반복 허용)만 있어야 수락
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VoskResult:
    cmd: str                 # 규칙을 통과한 명령 키, 없으면 "none"
    raw_cmd: str             # 규칙 적용 전 인식 결과에 들어 있던 첫 명령 키 (없으면 "none")
    text: str
    conf: float | None       # raw_cmd 단어들의 최소 신뢰도
    reject: str | None       # raw_cmd 가 있었는데 거부된 이유: "word_conf" | "not_exact"


def load_commands(voice_cfg: dict) -> dict[str, dict]:
    """[commands.<키>] 표 → {키: {phrase, min_word_conf, spk_threshold, exact, critical, group}} (기본값 채움)."""
    base_conf = voice_cfg["vosk"]["min_word_conf"]
    base_spk = voice_cfg["ecapa"]["threshold"]
    out = {}
    for key, c in voice_cfg["commands"].items():
        if isinstance(c, str):  # 이전 형식 (키 = "말") 호환
            c = {"phrase": c}
        out[key] = {"phrase": c["phrase"], "min_word_conf": c.get("min_word_conf", base_conf),
                    "spk_threshold": c.get("spk_threshold", base_spk), "exact": c.get("exact", False),
                    "critical": c.get("critical", False), "group": c.get("group")}
    return out


class VoskCommand:
    def __init__(self, model_dir: Path, commands: dict[str, dict], fillers: list[str], sample_rate: int = 16000):
        import vosk
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_dir))
        self.commands = commands
        missing = [w for c in commands.values() for w in c["phrase"].split() if self.model.vosk_model_find_word(w) < 0]
        if missing:
            raise ValueError(f"Vosk 어휘에 없는 명령 단어: {missing} (configs/voice.toml [commands] 수정)")
        fill = [w for w in fillers if self.model.vosk_model_find_word(w) >= 0]
        self.grammar = [c["phrase"] for c in commands.values()] + fill
        self.rec = vosk.KaldiRecognizer(self.model, sample_rate, json.dumps(self.grammar, ensure_ascii=False))
        self.rec.SetWords(True)

    def reset(self) -> None:
        self.rec.Reset()

    def accept(self, chunk_f32: np.ndarray) -> None:
        pcm = np.clip(np.round(chunk_f32 * 32768), -32768, 32767).astype(np.int16)
        self.rec.AcceptWaveform(pcm.tobytes())

    def final(self) -> VoskResult:
        res = json.loads(self.rec.FinalResult())
        return self.decide([w["word"] for w in res.get("result", [])], [w["conf"] for w in res.get("result", [])],
                           res.get("text", ""))

    def decide(self, toks: list[str], confs: list[float], text: str) -> VoskResult:
        first = None
        for key, c in self.commands.items():
            p = c["phrase"].split()
            for i in range(len(toks) - len(p) + 1):
                if toks[i:i + len(p)] == p:
                    conf = min(confs[i:i + len(p)])
                    if first is None:
                        first = (key, conf)
                    if conf < c["min_word_conf"]:
                        continue
                    if c["exact"] and any(t not in p for t in toks):
                        continue
                    return VoskResult(key, key, text, conf, None)
        if first is None:
            return VoskResult("none", "none", text, None, None)
        key, conf = first
        c = self.commands[key]
        reason = "word_conf" if conf < c["min_word_conf"] else "not_exact"
        return VoskResult("none", key, text, conf, reason)
