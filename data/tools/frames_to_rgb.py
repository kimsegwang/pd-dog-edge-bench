"""영상/이미지 묶음을 벤치마크 공통 입력(.rgb 무압축 RGB24 + .json 메타)으로 변환.

입력: PNG/JPG 가 든 .zip, 이미지 폴더, 또는 동영상 파일(.mp4 등)
    python data/tools/frames_to_rgb.py data/video/urfd_fall-01-cam0-rgb.zip data/video/urfd_fall-01-cam0.rgb
    python data/tools/frames_to_rgb.py clip.mp4 out.rgb --crop 320,0,320,240 --resize 640x480

왜 무압축인가: 실제 로봇은 Picamera2 가 이미 디코딩된 프레임을 주므로 코덱 디코딩은 실사용 경로가 아님.
디코더 구현 차이(Python cv2 vs Rust)를 비교에서 제거하기 위해 공정 비교 입력은 raw 로 고정한다.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def iter_frames(src: Path):
    """BGR uint8 프레임을 순서대로 yield."""
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for name in sorted(n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXT):
                yield cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
    elif src.is_dir():
        for p in sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT):
            yield cv2.imread(str(p), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(str(src))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        return ""
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, help="출력 .rgb 경로 (옆에 같은 이름 .json 생성)")
    ap.add_argument("--crop", help="x,y,w,h (리사이즈 전에 적용)")
    ap.add_argument("--resize", help="WxH")
    ap.add_argument("--fps", type=float, default=30.0, help="원본 촬영 FPS (메타 기록용)")
    args = ap.parse_args()

    crop = [int(v) for v in args.crop.split(",")] if args.crop else None
    size = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None
    n, shape = 0, None
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "wb") as out:
        for bgr in iter_frames(args.src):
            if crop:
                x, y, w, h = crop
                bgr = bgr[y:y + h, x:x + w]
            if size:
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if shape is None:
                shape = rgb.shape
            assert rgb.shape == shape, f"frame size changed: {rgb.shape} != {shape}"
            out.write(rgb.tobytes())
            n += 1
    if n == 0:
        raise SystemExit(f"no frames read from {args.src}")
    meta = {
        "format": "rgb24", "width": shape[1], "height": shape[0], "frames": n, "fps": args.fps,
        "source": args.src.name, "source_sha256": file_sha256(args.src),
        "crop": crop, "resize": list(size) if size else None,
    }
    args.dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{args.dst}: {n} frames {shape[1]}x{shape[0]} ({args.dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
