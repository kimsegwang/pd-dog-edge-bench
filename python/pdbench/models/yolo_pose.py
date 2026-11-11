"""YOLO11n-pose (ONNX, 고정 입력 해상도) 전처리 / 추론 / 후처리.

출력 텐서: (1, 56, N)  — 56 = box(cx,cy,w,h) 4 + person conf 1 + 키포인트 17×(x,y,vis) 51
Ultralytics export 는 conf 와 키포인트 vis 에 이미 sigmoid 를 적용한 상태.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..common.ort_session import make_session


@dataclass
class PoseResult:
    boxes: np.ndarray    # (K, 4) xyxy, 원본 이미지 좌표
    scores: np.ndarray   # (K,)
    kpts: np.ndarray     # (K, 17, 3) x, y, vis


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


class YoloPose:
    def __init__(self, model_path: Path, imgsz: int, cfg: dict, threads: int | None = None):
        m = cfg["model"]
        self.imgsz = imgsz
        self.conf_thr, self.iou_thr, self.max_det = m["conf_threshold"], m["iou_threshold"], m["max_det"]
        self.pad_value = m["pad_value"]
        self.session = make_session(model_path, cfg["ort"], threads)
        inp = self.session.get_inputs()[0]
        if list(inp.shape) != [1, 3, imgsz, imgsz]:
            raise ValueError(f"{model_path.name}: input shape {inp.shape} != [1,3,{imgsz},{imgsz}]")
        self.input_name = inp.name
        self._canvas = np.full((imgsz, imgsz, 3), self.pad_value, np.uint8)

    def preprocess(self, rgb: np.ndarray):
        """letterbox(비율 유지 + 가운데 패딩) → NCHW float32 [0,1]."""
        h, w = rgb.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nw, nh = round(w * r), round(h * r)
        left, top = (s - nw) // 2, (s - nh) // 2
        canvas = self._canvas
        canvas[:] = self.pad_value
        resized = rgb if (nw, nh) == (w, h) else cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[top:top + nh, left:left + nw] = resized
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        x *= 1.0 / 255.0
        return x, (r, left, top, w, h)

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def postprocess(self, out: np.ndarray, meta) -> PoseResult:
        r, left, top, w, h = meta
        p = out[0].T                      # (N, 56)
        p = p[p[:, 4] > self.conf_thr]
        if p.shape[0] == 0:
            return PoseResult(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32))
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        keep = nms(boxes, p[:, 4], self.iou_thr, self.max_det)
        boxes, scores, kpts = boxes[keep], p[keep, 4], p[keep, 5:].reshape(-1, 17, 3).copy()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clip(0, w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clip(0, h)
        kpts[..., 0] = ((kpts[..., 0] - left) / r).clip(0, w)
        kpts[..., 1] = ((kpts[..., 1] - top) / r).clip(0, h)
        return PoseResult(boxes, scores, kpts)
