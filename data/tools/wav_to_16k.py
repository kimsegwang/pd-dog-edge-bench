"""wav 를 16kHz mono PCM16 으로 변환 (표준 라이브러리 + numpy 만 사용).

    python data/tools/wav_to_16k.py in.wav out.wav

16/24/32-bit PCM, 모노/스테레오 입력 지원. 리샘플은 data/tools/resample.py (FFT 대역 제한, 앨리어싱 없음).
휴대폰 녹음(m4a 등)은 먼저 wav 로: ffmpeg -i in.m4a -ac 1 -ar 16000 -sample_fmt s16 out.wav
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resample import resample, to_int16  # noqa: E402

TARGET_SR = 16000


def read_any_pcm(path: Path) -> tuple[np.ndarray, int]:
    """wav → (mono float32 [-1,1], sample_rate)"""
    with wave.open(str(path), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = (np.where(v & 0x800000, v - 0x1000000, v) / 8388608).astype(np.float32)
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648
    else:
        raise SystemExit(f"{path}: {sw * 8}-bit PCM 은 지원 안 함 (ffmpeg 사용)")
    return x.reshape(-1, ch).mean(axis=1), sr


def write_16k(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(to_int16(x).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    x, sr = read_any_pcm(args.src)
    y = resample(x, sr, TARGET_SR)
    write_16k(args.dst, y)
    print(f"{args.dst}: {len(y) / TARGET_SR:.1f}s 16kHz mono (from {sr}Hz)")


if __name__ == "__main__":
    main()
