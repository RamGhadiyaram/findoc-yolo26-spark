"""Fine-tune YOLO26 on a financial-document field dataset and export to ONNX.

This is Listing 1 of the paper as a runnable CLI. Every hyperparameter is an
argument, and the run writes a small JSON summary (metrics, wall time, ONNX
path) next to the weights so results in the paper trace back to a file.

Example (CPU smoke run on the synthetic dataset):
    python -m findoc.train data/financial_docs.yaml --model yolo26n.pt \
        --epochs 20 --batch 8 --device cpu --project runs --name synth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from findoc.export import export_onnx


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_yaml")
    ap.add_argument("--model", default="yolo26x.pt", help="Pretrained checkpoint (yolo26n/s/m/l/x.pt).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr0", type=float, default=0.001, help="0.001 is the AdamW value Ultralytics recommends; at 0.01 val mAP50 was unstable in epochs 2-3 (see results/early_epochs_lr0.01.csv)")
    ap.add_argument("--optimizer", default="AdamW", help="YOLO26 defaults to MuSGD; AdamW is explicit here.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="finetune")
    a = ap.parse_args(argv)

    model = YOLO(a.model)
    t0 = time.perf_counter()
    model.train(
        data=a.data_yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, lr0=a.lr0,
        optimizer=a.optimizer, device=a.device, workers=a.workers, seed=a.seed,
        # absolute project path: Ultralytics nests relative paths under runs/detect/
        project=str(Path(a.project).resolve()), name=a.name, exist_ok=True, plots=False, verbose=False,
    )
    train_seconds = time.perf_counter() - t0

    best = Path(model.trainer.best)
    tuned = YOLO(str(best))
    onnx_path = export_onnx(str(best), a.imgsz, dynamic=True)  # dynamic batch axis, end-to-end head kept

    val = YOLO(str(best)).val(data=a.data_yaml, split="val", device=a.device, imgsz=a.imgsz, plots=False, verbose=False)
    summary = {
        "weights": str(best),
        "onnx": str(onnx_path),
        "train_seconds": round(train_seconds, 1),
        "epochs": a.epochs,
        "val_map50": round(float(val.box.map50), 4),
        "val_map50_95": round(float(val.box.map), 4),
        "per_class_map50": {n: round(float(v), 4) for n, v in zip(val.names.values(), val.box.ap50)},
    }
    (best.parent.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
