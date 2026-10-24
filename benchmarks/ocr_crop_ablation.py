"""Does crop handling explain the OCR errors? Re-reads the same detections with different crop preparation.

Runs on the VALIDATION split only, so nothing is tuned on test. The paper's headline configuration
(no padding, no scaling, no thresholding) is not changed by this script.

    python benchmarks/ocr_crop_ablation.py work/data work/runs/synth/weights/best.onnx results/ocr_crop_ablation.json --conf 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc.evaluate import score
from findoc.pipeline import run

CONFIGS = [
    ("as reported (pad 0)", dict(pad=0, scale=1.0, binarize=False)),
    ("pad 4", dict(pad=4, scale=1.0, binarize=False)),
    ("pad 8", dict(pad=8, scale=1.0, binarize=False)),
    ("pad 4, upscale 2x", dict(pad=4, scale=2.0, binarize=False)),
    ("pad 4, upscale 2x, Otsu", dict(pad=4, scale=2.0, binarize=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    a = ap.parse_args()
    data = Path(a.data)
    rows = []
    for name, kw in CONFIGS:
        out = f"work/out_ablation_{len(rows)}"
        run(str(data / "images" / "val"), a.model, out, partitions=2, conf=a.conf, nms_iou=a.nms_iou, **kw)
        ev = score(out, str(data / "val_truth.json"), str(data / "images" / "val"), baseline=False)
        rows.append({"config": name, **kw, "end_to_end": ev["pipeline_end_to_end_field_accuracy"],
                     "ocr_errors": len(ev["ocr_errors"]),
                     "per_class_ocr": {c: v["ocr_exact"] for c, v in ev["per_class"].items()}})
        print(rows[-1], flush=True)
    Path(a.out).write_text(json.dumps({"split": "val", "conf": a.conf, "nms_iou": a.nms_iou, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
