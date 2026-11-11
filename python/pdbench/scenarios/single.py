"""시나리오 A: 포즈 추정 단독 실행 (녹화 프레임을 최대 속도로 처리)."""
from pathlib import Path

from ..common.sysinfo import Monitor, peak_rss_mb, run_flags, summarize_system, wait_cooldown, write_json
from ..common.timing import now_ns
from .pose_runner import PoseRunner


def run(out_dir: Path, res: int, threads: int, frames: Path, warmup: int, measure_frames: int,
        max_measure_s: float, monitor_interval: float, cooldown: tuple[float, float] | None,
        cooler: str = "unknown") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cool = wait_cooldown(*cooldown) if cooldown else {}
    mon = Monitor(out_dir / "system.csv", monitor_interval)
    try:
        runner = PoseRunner(res, threads, frames)
        for _ in range(warmup):
            runner.step(record=False)
        runner.start_clock()
        deadline = now_ns() + int(max_measure_s * 1e9)
        # 시간 상한이 지나도 입력 영상 한 바퀴(모든 프레임)는 처리 → 해상도 간 키포인트 비교가 같은 프레임 범위가 되도록
        min_frames = max(30, runner.src.n_frames)
        while len(runner.rows) < measure_frames and (now_ns() < deadline or len(runner.rows) < min_frames):
            runner.step()
    finally:
        mon.stop()
    runner.save(out_dir)
    system = summarize_system(out_dir / "system.csv")
    summary = {"scenario": "A", "lang": "python", "cooler": cooler, "frames_file": Path(frames).name,
               "pose": runner.summary(), "peak_rss_mb": peak_rss_mb(),
               "system": system, **cool, "flags": run_flags(system, cool)}
    write_json(out_dir / "summary.json", summary)
    p = summary["pose"]
    print(f"[A] res={res} threads={threads}: {p['fps']:.2f} FPS, infer p50 {p['infer_ms']['p50']:.1f} ms, "
          f"total p95 {p['total_ms']['p95']:.1f} ms, peak RSS {summary['peak_rss_mb']:.0f} MB, "
          f"max {system.get('temp_c_max', '-')}°C", flush=True)
    for fl in summary["flags"]:
        print(f"  WARN: {fl}", flush=True)
    return summary
