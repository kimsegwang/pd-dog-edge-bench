"""[노트북 전용] YOLO11n-pose 를 해상도별 고정 shape FP32 ONNX 로 export.

    python models/convert/export_yolo_pose.py            # 640/416/320
    python models/convert/export_yolo_pose.py --res 320

결과: models/yolo11n-pose_{res}_fp32.onnx + models/yolo11n-pose_export.json (버전/sha256/출력 검증)
"""
import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
WORK = Path(__file__).resolve().parent / "_work"
OPSET = 17  # Pi 의 onnxruntime 이 확실히 지원하는 opset 으로 고정
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-url", default=WEIGHTS_URL)
    ap.add_argument("--res", type=int, nargs="+", default=[640, 416, 320])
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO

    WORK.mkdir(exist_ok=True)
    weights = WORK / Path(args.weights_url).name
    if not weights.exists():
        import urllib.request
        print(f"downloading {args.weights_url}")
        urllib.request.urlretrieve(args.weights_url, weights)
    report = {
        "weights_url": args.weights_url,
        "weights_sha256": sha256(weights),
        "opset": OPSET,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "host": platform.platform(),
        "models": {},
    }
    for res in args.res:
        model = YOLO(str(weights))
        out = Path(model.export(format="onnx", imgsz=res, opset=OPSET, simplify=True,
                                dynamic=False, half=False, batch=1, device="cpu"))
        dst = MODELS / f"yolo11n-pose_{res}_fp32.onnx"
        shutil.move(out, dst)

        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        y = sess.run(None, {i.name: np.zeros(i.shape, np.float32)})[0]
        expected_anchors = sum((res // s) ** 2 for s in (8, 16, 32))
        assert y.shape == (1, 56, expected_anchors), y.shape
        report["models"][str(res)] = {
            "file": dst.name, "sha256": sha256(dst), "bytes": dst.stat().st_size,
            "input": [i.name, i.shape], "output": [o.name, list(y.shape)],
        }
        print(f"[{res}] {dst.name}  in={i.shape} out={y.shape}")
    (MODELS / "yolo11n-pose_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote models/yolo11n-pose_export.json")


if __name__ == "__main__":
    main()
