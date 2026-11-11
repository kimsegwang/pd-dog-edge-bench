"""시간 측정과 요약 통계."""
import time

import numpy as np

now_ns = time.perf_counter_ns


def ms(ns: int) -> float:
    return ns / 1e6


def pct(values, q: float) -> float | None:
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return float(np.percentile(a, q)) if a.size else None


def describe(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()), "min": float(a.min()),
    }


def window_fps(t_done_s, window_s: float = 1.0) -> list[float]:
    """처리 완료 시각(초) 목록 → 겹치지 않는 window 별 FPS (마지막 미완성 window 제외)."""
    t = np.asarray(t_done_s, dtype=np.float64)
    if t.size < 2:
        return []
    t = t - t[0]
    n_win = int(t[-1] // window_s)
    counts = np.bincount((t // window_s).astype(int), minlength=n_win + 1)[:n_win]
    return (counts / window_s).tolist()
