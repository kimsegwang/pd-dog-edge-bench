"""Silero VAD v5 (ONNX) 스트리밍 래퍼 + 발화 구간 판정기.

입력: input (1, 64+512) float32 = 직전 청크의 마지막 64 샘플(context) + 현재 512 샘플
      state (2, 1, 128) float32, sr int64 스칼라
출력: 음성 확률 (1, 1), 다음 state
공식 패키지(silero_vad.utils_vad.OnnxWrapper / VADIterator)와 같은 규칙으로 구현.
"""
from pathlib import Path

import numpy as np

from ..common.ort_session import make_session


class SileroVAD:
    def __init__(self, model_path: Path, cfg: dict, threads: int | None = None):
        m = cfg["model"]
        self.sr = m["sample_rate"]
        self.chunk, self.ctx = m["chunk_samples"], m["context_samples"]
        self.session = make_session(model_path, cfg["ort"], threads)
        names = {i.name for i in self.session.get_inputs()}
        if names != {"input", "state", "sr"}:
            raise ValueError(f"unexpected Silero VAD inputs {names} (v5 형식 필요)")
        self._sr = np.array(self.sr, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, self.ctx), np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self.context, chunk.reshape(1, -1)], axis=1)
        prob, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self._sr})
        self.context = x[:, -self.ctx:]
        return float(prob[0, 0])


class VadSegmenter:
    """확률 열 → 발화 시작/종료 이벤트 (VADIterator 와 같은 히스테리시스 + 최소 무음 규칙)."""

    def __init__(self, cfg: dict, sample_rate: int):
        s = cfg["segmenter"]
        self.th, self.neg_th = s["threshold"], s["neg_threshold"]
        self.min_silence = int(s["min_silence_ms"] * sample_rate / 1000)
        self.triggered = False
        self.temp_end = 0

    def update(self, prob: float, chunk_start: int, chunk_end: int):
        """('start', 샘플번호) / ('end', 실제 발화가 끝난 샘플번호) / None"""
        if prob >= self.th and self.temp_end:
            self.temp_end = 0
        if prob >= self.th and not self.triggered:
            self.triggered = True
            return "start", chunk_start
        if prob < self.neg_th and self.triggered:
            if not self.temp_end:
                self.temp_end = chunk_start
            if chunk_end - self.temp_end >= self.min_silence:
                end, self.temp_end, self.triggered = self.temp_end, 0, False
                return "end", end
        return None
