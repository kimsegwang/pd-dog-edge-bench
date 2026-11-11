"""포즈 한 프레임 처리(읽기→전처리→추론→후처리)와 시간 기록. 시나리오 A/B/C 공용."""
import csv
import threading
import time
from pathlib import Path

import numpy as np

from ..common import config
from ..common.sysinfo import sha256
from ..common.timing import describe, ms, now_ns, window_fps
from ..models.yolo_pose import YoloPose
from ..sources.raw_frames import RawFrameSource

FIELDS = ["i", "frame_idx", "t_done_s", "read_ms", "pre_ms", "infer_ms", "post_ms", "total_ms", "n_det", "best_score"]


class PoseRunner:
    def __init__(self, res: int, threads: int, frames_path: Path, cfg: dict | None = None):
        self.cfg = cfg or config.load("pose")
        self.res, self.threads = res, threads
        self.model_path = config.repo_path(self.cfg["model"]["path_template"].format(res=res))
        t = now_ns()
        self.model = YoloPose(self.model_path, res, self.cfg, threads)
        self.load_ms = ms(now_ns() - t)
        self.src = RawFrameSource(frames_path)
        self.kpts = np.full((self.src.n_frames, 17, 3), np.nan, np.float32)  # 프레임별 최고점수 사람 (없으면 NaN)
        self.measured = np.zeros(self.src.n_frames, bool)  # 측정 구간에서 실제로 처리한 프레임
        self.fps_cap = 0.0
        self.rows: list[list] = []
        self.t_origin_ns = now_ns()

    def step(self, record: bool = True) -> None:
        t0 = now_ns()
        idx, rgb = self.src.read()
        t1 = now_ns()
        x, meta = self.model.preprocess(rgb)
        t2 = now_ns()
        out = self.model.infer(x)
        t3 = now_ns()
        res = self.model.postprocess(out, meta)
        t4 = now_ns()
        if not record:
            return
        self.measured[idx] = True
        best = float(res.scores.max()) if res.scores.size else 0.0
        if res.scores.size:
            self.kpts[idx] = res.kpts[int(res.scores.argmax())]
        self.rows.append([len(self.rows), idx, (t4 - self.t_origin_ns) / 1e9, ms(t1 - t0), ms(t2 - t1),
                          ms(t3 - t2), ms(t4 - t3), ms(t4 - t1), int(res.scores.size), round(best, 4)])

    def start_clock(self) -> None:
        self.t_origin_ns = now_ns()

    def run_loop(self, stop: threading.Event, fps_cap: float = 0.0) -> None:
        """stop 될 때까지 처리. fps_cap > 0 이면 카메라처럼 1/fps_cap 간격으로만 프레임을 받는다.
        처리가 간격보다 오래 걸리면 밀린 프레임을 몰아서 처리하지 않고 다음 간격부터 다시 맞춘다(= 프레임 드롭)."""
        self.fps_cap = fps_cap
        period = int(1e9 / fps_cap) if fps_cap > 0 else 0
        next_t = now_ns()
        while not stop.is_set():
            if period:
                t = now_ns()
                if t < next_t:
                    time.sleep((next_t - t) / 1e9)
                else:
                    next_t = t
                next_t += period
            self.step()

    def save(self, out_dir: Path) -> None:
        with open(out_dir / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for r in self.rows:
                w.writerow([f"{v:.3f}" if isinstance(v, float) else v for v in r])
        np.savez_compressed(out_dir / "kpts.npz", kpts=self.kpts, measured=self.measured)

    def summary(self, bucket_s: float | None = None) -> dict:
        # 열: t_done_s, read, pre, infer, post, total  (total = 전처리+추론+후처리, 파일 읽기 제외)
        a = np.asarray([r[2:8] for r in self.rows], dtype=np.float64)
        if a.size == 0:
            return {"frames": 0}
        t_done = a[:, 0]
        elapsed = float(t_done[-1])
        wfps = window_fps(t_done, 1.0)
        out = {
            "res": self.res, "threads": self.threads, "fps_cap": self.fps_cap or None, "model": self.model_path.name,
            "model_sha256": sha256(self.model_path), "load_ms": self.load_ms,
            "frames": len(self.rows), "elapsed_s": elapsed, "fps": len(self.rows) / elapsed if elapsed else None,
            "window_fps_p5": float(np.percentile(wfps, 5)) if wfps else None,
            "window_fps_p50": float(np.percentile(wfps, 50)) if wfps else None,
            "read_ms": describe(a[:, 1]), "pre_ms": describe(a[:, 2]), "infer_ms": describe(a[:, 3]),
            "post_ms": describe(a[:, 4]), "total_ms": describe(a[:, 5]),
            "detect_rate": float(np.mean([r[8] > 0 for r in self.rows])),
            "frames_covered": int(self.measured.sum()), "frames_in_source": int(self.measured.size),
        }
        if bucket_s:
            buckets = []
            for b in range(int(elapsed // bucket_s) + 1):
                m = (t_done >= b * bucket_s) & (t_done < (b + 1) * bucket_s)
                span = min(elapsed, (b + 1) * bucket_s) - b * bucket_s
                if m.sum() and span > bucket_s * 0.5:
                    buckets.append({"t0_s": b * bucket_s, "fps": float(m.sum() / span),
                                    "total_p95_ms": float(np.percentile(a[m, 5], 95))})
            out["buckets"] = buckets
        return out
