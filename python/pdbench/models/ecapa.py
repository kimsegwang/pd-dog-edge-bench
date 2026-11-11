"""ECAPA-TDNN 화자 임베딩 (SpeechBrain spkrec-ecapa-voxceleb → ONNX) + numpy 특징 추출.

특징 = SpeechBrain Fbank 와 동일:
  center 패딩(0, n_fft/2) → 해밍 윈도우 STFT(n_fft 400, hop 160) → 파워 스펙트럼 → 멜 행렬(80)
  → 10·log10(max(x, amin)) → 발화 전체 최대값 - top_db(80) 아래는 잘라냄 → 시간축 평균 빼기(문장 정규화)
상수(멜 행렬, 윈도우 등)는 models/ecapa_fbank.npz 에서 읽는다 (export_ecapa.py 가 SpeechBrain 에서 추출).
"""
from pathlib import Path

import numpy as np


def fbank_features(x: np.ndarray, c) -> np.ndarray:
    """float32 파형(16kHz, [-1,1]) → (frames, 80) 정규화된 log-mel."""
    n_fft, hop = int(c["n_fft"]), int(c["hop_length"])
    window, mel = c["window"], c["mel"]
    xp = np.pad(x.astype(np.float32), (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[idx] * window, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    m = power @ mel
    db = float(c["multiplier"]) * np.log10(np.maximum(m, float(c["amin"])))
    db -= float(c["multiplier"]) * float(c["db_multiplier"])
    db = np.maximum(db, db.max() - float(c["top_db"]))
    return (db - db.mean(axis=0, keepdims=True)).astype(np.float32)


class Ecapa:
    def __init__(self, model_path: Path, fbank_path: Path, ort_cfg: dict, threads: int | None = None):
        from ..common.ort_session import make_session
        self.session = make_session(model_path, ort_cfg, threads)
        self.consts = dict(np.load(fbank_path))
        self.sr = int(self.consts["sample_rate"])

    def features(self, x: np.ndarray) -> np.ndarray:
        return fbank_features(x, self.consts)

    def embed_features(self, feats: np.ndarray) -> np.ndarray:
        e = self.session.run(None, {"feats": feats[None]})[0].reshape(-1)
        return e / (np.linalg.norm(e) + 1e-12)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_features(self.features(x))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
