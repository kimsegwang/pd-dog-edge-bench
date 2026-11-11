"""발화 파일 묶음 → 실시간 투입용 연속 스트림 wav + 정답 라벨 (발화 시작/끝 샘플).

    python data/tools/build_command_stream.py data/voice/tts_heami

입력 폴더 규칙 (make_tts_commands.ps1 / record_commands.py 결과와 동일)
  <set>/enroll/*.wav                          등록 화자 등록용 (스트림에는 안 들어감)
  <set>/utts/<speaker>__<command>__<nn>.wav    speaker: target(등록 화자) 또는 다른 이름
                                               command: configs/voice.toml [commands] 의 키 또는 none
  모든 wav 는 16kHz mono 16-bit (data/tools/wav_to_16k.py 로 변환)

출력
  <set>/stream.wav           발화 사이에 1.2~2.4초 무음(-60 dBFS 백색 잡음)을 넣어 섞은 연속 음성
  <set>/stream_labels.csv    utt, file, speaker, command, start_sample, end_sample
                             start/end = 에너지 기준 실제 음성 구간 (end = '발화 종료' 정답 시각)

정답 발화 끝 = 10ms 프레임 RMS 가 (해당 발화 최대 - 35 dB) 와 -55 dBFS 중 큰 값을 넘는 마지막 프레임의 끝.
스트림 길이는 512 샘플(VAD 청크)의 배수로 맞춘다 → 반복 재생 시 라벨 위치가 어긋나지 않음.
"""
import argparse
import csv
import wave
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
            raise SystemExit(f"{p}: 16kHz mono 16-bit 필요 (data/tools/wav_to_16k.py)")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def speech_bounds(x: np.ndarray, frame: int = 160) -> tuple[int, int] | None:
    n = len(x) // frame
    if n == 0:
        return None
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = max(db.max() - 35.0, -55.0)
    on = np.flatnonzero(db > thr)
    if on.size == 0:
        return None
    return int(on[0] * frame), int((on[-1] + 1) * frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=float, nargs=2, default=[1.2, 2.4], help="발화 사이 무음 길이 범위(초)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted((args.set_dir / "utts").glob("*.wav"))
    if not files:
        raise SystemExit(f"{args.set_dir}/utts/*.wav 없음")
    order = rng.permutation(len(files))

    def silence(sec: float) -> np.ndarray:
        return (rng.standard_normal(int(sec * SR)) * 10 ** (-60 / 20)).astype(np.float32)

    parts, labels, pos = [silence(1.0)], [], int(1.0 * SR)
    for k, i in enumerate(order):
        p = files[i]
        try:
            speaker, command, _ = p.stem.split("__")
        except ValueError:
            raise SystemExit(f"파일 이름 규칙 위반: {p.name} (<speaker>__<command>__<nn>.wav)")
        x = read_wav(p)
        b = speech_bounds(x)
        if b is None:
            print(f"skip (무음): {p.name}")
            continue
        labels.append([f"u{k:03d}", p.name, speaker, command, pos + b[0], pos + b[1]])
        parts.append(x)
        pos += len(x)
        g = silence(rng.uniform(*args.gap))
        parts.append(g)
        pos += len(g)
    y = np.concatenate(parts)
    pad = -len(y) % CHUNK
    y = np.concatenate([y, (rng.standard_normal(pad) * 10 ** (-60 / 20)).astype(np.float32)])
    assert len(y) % CHUNK == 0
    pcm = np.clip(np.round(y * 32768), -32768, 32767).astype(np.int16)
    with wave.open(str(args.set_dir / "stream.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    with open(args.set_dir / "stream_labels.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["utt", "file", "speaker", "command", "start_sample", "end_sample"])
        wr.writerows(labels)
    print(f"{args.set_dir / 'stream.wav'}: {len(y) / SR:.1f}s, {len(labels)} utterances "
          f"(enroll: {len(list((args.set_dir / 'enroll').glob('*.wav')))} files)")


if __name__ == "__main__":
    main()
