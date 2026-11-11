"""녹음 파일용 리샘플러 (numpy 만 사용). 짧은 발화 파일(수 초) 대상.

FFT 대역 제한 방식: 스펙트럼을 목표 샘플레이트의 나이퀴스트까지만 남기고 역변환 → 고역이 접혀 들어오지 않음(앨리어싱 없음).
양 끝 0.1초를 0 으로 덧대어 경계 왜곡을 줄인 뒤 잘라낸다.
"""
import numpy as np


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    pad = int(0.1 * sr_in)
    xp = np.pad(x, (pad, pad))
    n_out = int(round(len(xp) * sr_out / sr_in))
    spec = np.fft.rfft(xp)
    keep = min(len(spec), n_out // 2 + 1)
    y = np.fft.irfft(spec[:keep], n=n_out) * (n_out / len(xp))
    pad_out = int(round(pad * sr_out / sr_in))
    n_keep = int(round(len(x) * sr_out / sr_in))
    return y[pad_out:pad_out + n_keep].astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
