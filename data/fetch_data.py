"""오늘 측정용 공개 데이터 내려받기 + 벤치마크 입력 형식으로 변환. Pi 에서 실행.

    python data/fetch_data.py

- 영상: UR Fall Detection Dataset (Kwolek & Kepski, 2014) cam0 RGB 640x480 PNG 시퀀스
  http://fenix.ur.edu.pl/~mkepski/ds/uf.html  — 연구 목적 공개. 레포에는 커밋하지 않음.
    fall-01 : 낙상 1건 (160 프레임)
    adl-01  : 일상 동작 (비교용)
- 음성: silero-vad 저장소의 테스트 음성 (16kHz mono, 60초, MIT)
"""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DOWNLOADS = [
    ("http://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0-rgb.zip",
     "video/urfd_fall-01-cam0-rgb.zip",
     "e5ff96108d322d7fe5ba6871a4c50d9c26c59e74abea03d2b69b6be56521736f"),
    ("http://fenix.ur.edu.pl/~mkepski/ds/data/adl-01-cam0-rgb.zip",
     "video/urfd_adl-01-cam0-rgb.zip",
     "a3d51bbc71cfd06899f541f46277699d72503465f02397f91b705cf36aa2c627"),
    ("https://github.com/snakers4/silero-vad/raw/v6.0/tests/data/test.wav",
     "audio/silero_test.wav",
     "89f17d9c94c4b31eb320f424628bcbc920abaddbee6e2760fd868bfb1d9a2e47"),
]

CONVERSIONS = [
    ["data/tools/frames_to_rgb.py", "data/video/urfd_fall-01-cam0-rgb.zip", "data/video/urfd_fall-01-cam0.rgb"],
    ["data/tools/frames_to_rgb.py", "data/video/urfd_adl-01-cam0-rgb.zip", "data/video/urfd_adl-01-cam0.rgb"],
    ["data/tools/wav_to_16k.py", "data/audio/silero_test.wav", "data/audio/silero_test_16k.wav"],
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    # 변환 결과(.rgb / _16k.wav)가 이미 있는 원본은 받지 않음 (노트북에서 변환본만 옮긴 경우)
    converted = {src for _, src, dst in CONVERSIONS if (ROOT / dst).exists()}
    for url, rel, digest in DOWNLOADS:
        dst = DATA / rel
        if not dst.exists() and f"data/{rel}" in converted:
            print(f"{rel}: 변환 결과가 이미 있어 다운로드 생략")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)
        got = sha256(dst)
        if digest and got != digest:
            print(f"SHA256 MISMATCH {rel}: {got}")
            return 1
        print(f"{rel}: sha256={got[:16]}…")
    for script, src, dst in CONVERSIONS:
        if (ROOT / dst).exists():
            print(f"{dst}: exists")
            continue
        subprocess.run([sys.executable, str(ROOT / script), str(ROOT / src), str(ROOT / dst)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
