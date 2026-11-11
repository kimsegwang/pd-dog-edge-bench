"""시나리오 B/C/V: 포즈 추정 + 음성 체인을 한 프로세스에서 동시에 실행 (실제 로봇처럼).

- pose 스레드  : 녹화 프레임 처리. fps_cap=0 이면 최대 속도(= 낼 수 있는 처리 FPS),
                 fps_cap>0 이면 카메라처럼 그 간격으로만 처리 (상한을 걸었을 때 CPU·온도 확인용)
- audio 스레드 : wav 를 실시간 속도로 투입, 512 샘플(32ms)마다 VAD
                 lag = 처리 완료 시각 - 청크 도착 시각  → 32ms 를 계속 넘으면 실시간 처리 실패
- speech 스레드: voice="full" 일 때 Vosk 명령 인식 + ECAPA 화자 확인 (scenarios/voice_chain.py)
시나리오 V = 포즈 없이 음성 체인만 (res=None).
"""
import csv
import threading
import time
from pathlib import Path

from ..common.sysinfo import Monitor, peak_rss_mb, run_flags, summarize_system, wait_cooldown, write_json
from ..common.timing import now_ns
from .pose_runner import PoseRunner
from .voice_chain import VoiceChain


def run(out_dir: Path, scenario: str, res: int | None, pose_threads: int, vad_threads: int, duration_s: float,
        frames: Path, wav: Path, warmup_frames: int, warmup_chunks: int, monitor_interval: float,
        cooldown: tuple[float, float] | None, bucket_s: float | None = None, cooler: str = "unknown",
        fps_cap: float = 0.0, voice: str = "vad", voice_set: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cool = wait_cooldown(*cooldown) if cooldown else {}
    mon = Monitor(out_dir / "system.csv", monitor_interval)
    errors: list[BaseException] = []
    try:
        pose = PoseRunner(res, pose_threads, frames) if res else None
        chain = VoiceChain(vad_threads, voice == "full", wav, voice_set)
        if pose:
            for _ in range(warmup_frames):
                pose.step(record=False)
        chain.warmup(warmup_chunks)

        stop = threading.Event()

        def guarded(fn, *a):
            def body():
                try:
                    fn(*a)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    stop.set()
            return body

        t0 = now_ns()
        workers = [threading.Thread(target=guarded(chain.audio_worker, stop, t0), name="audio")]
        if chain.full:
            workers.append(threading.Thread(target=guarded(chain.speech_worker, t0), name="speech"))
        if pose:
            pose.start_clock()
            workers.append(threading.Thread(target=guarded(pose.run_loop, stop, fps_cap), name="pose"))
        for t in workers:
            t.start()
        t_end = time.monotonic() + duration_s
        next_log = time.monotonic() + 60
        while not stop.is_set() and time.monotonic() < t_end:
            time.sleep(0.5)
            if time.monotonic() >= next_log:
                next_log += 60
                el = (now_ns() - t0) / 1e9
                print(f"  [{scenario}] {el / 60:.0f}/{duration_s / 60:.0f} min"
                      + (f", pose {len(pose.rows) / el:.2f} FPS" if pose else "")
                      + f", vad chunks {len(chain.audio_rows)}, decisions {len(chain.decisions)}", flush=True)
        stop.set()
        for t in workers:
            t.join(timeout=60)
    finally:
        mon.stop()
    if errors:
        raise errors[0]

    if pose:
        pose.save(out_dir)
    chain.save(out_dir, t0)
    system = summarize_system(out_dir / "system.csv")
    summary = {
        "scenario": scenario, "lang": "python", "cooler": cooler,
        "frames_file": Path(frames).name if pose else None, "duration_s": duration_s,
        "pose": pose.summary(bucket_s) if pose else None,
        **chain.summary(),
        "peak_rss_mb": peak_rss_mb(),
        "system": system,
        **cool,
        "flags": run_flags(system, cool),
    }
    if bucket_s:
        summary["system_buckets"] = _system_buckets(out_dir / "system.csv", bucket_s)
    write_json(out_dir / "summary.json", summary)

    p, v, vo = summary["pose"], summary["vad"], summary.get("voice")
    msg = f"[{scenario}]"
    if p:
        msg += (f" res={res} pose_threads={pose_threads}{f' cap={fps_cap:g}' if fps_cap else ''}: "
                f"pose {p['fps']:.2f} FPS (1s-window p5 {p['window_fps_p5']}),")
    msg += f" VAD lag p95 {v['lag_ms'].get('p95', float('nan')):.1f} ms,"
    if vo:
        e = vo["e2e_cmd_ms"]
        msg += (f" speech end->decision p50/p95/max {e.get('p50', float('nan')):.0f}/{e.get('p95', float('nan')):.0f}/"
                f"{e.get('max', float('nan')):.0f} ms, cmd {vo['cmd_accepted_correct']}/{vo['cmd_utts']},"
                f" false accept {vo['false_accepts']}/{vo['neg_utts']},")
    else:
        msg += f" VAD end p50 {v['end_decision_ms'].get('p50', float('nan')):.0f} ms,"
    msg += f" CPU {system.get('cpu_total_mean', float('nan')):.0f}%, peak RSS {summary['peak_rss_mb']:.0f} MB, max {system.get('temp_c_max', '-')}°C"
    print(msg, flush=True)
    for fl in summary["flags"]:
        print(f"  WARN: {fl}", flush=True)
    return summary


def _system_buckets(csv_path: Path, bucket_s: float) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    acc: dict[int, dict] = {}
    for r in rows:
        b = int(float(r["t_s"]) // bucket_s)
        d = acc.setdefault(b, {"temp": [], "freq": [], "cpu": [], "rss": [], "thr": 0})
        if r["temp_c"]:
            d["temp"].append(float(r["temp_c"]))
        if r["arm_freq_mhz"]:
            d["freq"].append(float(r["arm_freq_mhz"]))
        d["cpu"].append(float(r["cpu_total_pct"]))
        d["rss"].append(float(r["rss_mb"]))
        if r["throttled_hex"] and int(r["throttled_hex"], 16) & 0xF:
            d["thr"] += 1
    return [{"t0_s": b * bucket_s,
             "temp_max_c": max(d["temp"]) if d["temp"] else None,
             "freq_min_mhz": min(d["freq"]) if d["freq"] else None,
             "cpu_total_mean": sum(d["cpu"]) / len(d["cpu"]),
             "rss_max_mb": max(d["rss"]), "throttle_samples": d["thr"]}
            for b, d in sorted(acc.items())]
