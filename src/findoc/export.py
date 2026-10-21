"""Export a fine-tuned checkpoint to ONNX with a dynamic batch axis.

A static-batch ONNX graph can still be called with a list of images, because
Ultralytics feeds it one chunk at a time, but then the "batched" call is not
batched at the runtime level. A dynamic batch axis lets onnxruntime run the
whole Arrow batch as one tensor. YOLO26's end-to-end head is kept (nms=False).

    python -m findoc.export work/runs/synth/weights/best.pt
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def export_onnx(weights: str, imgsz: int = 640, dynamic: bool = True) -> str:
    return str(YOLO(weights).export(format="onnx", imgsz=imgsz, nms=False, simplify=True, dynamic=dynamic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--static", action="store_true", help="export a fixed batch of 1 instead")
    a = ap.parse_args()
    print(export_onnx(a.weights, a.imgsz, dynamic=not a.static))
