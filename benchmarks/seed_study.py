"""Training-seed variance: retrain with several seeds on the SAME dataset and rerun the evaluation protocol.

The data and test split are fixed, so the spread shows training variance only. Intended for a GPU
(about 10 minutes per seed on a T4). Results go to results_seeds/seed<N>/.

    python benchmarks/seed_study.py work/data --seeds 1 2 3 4 --device 0 --out results_gpu_seeds
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from findoc import protocol, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="results_gpu_seeds")
    a = ap.parse_args()
    data = Path(a.data)
    for seed in a.seeds:
        res = Path(a.out) / f"seed{seed}"
        if (res / "eval_report_class_nms.json").exists():
            print("skip seed", seed, "(already done)")
            continue
        summary = train.main([str(data / "financial_docs.yaml"), "--model", a.model, "--epochs", str(a.epochs),
                              "--batch", "8", "--device", a.device, "--project", "work/runs_seeds",
                              "--name", f"seed{seed}", "--seed", str(seed)])
        onnx = Path(summary["onnx"])
        protocol.main(str(data), str(onnx), str(res))
        res.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx.parent.parent / "summary.json", res / "train_summary.json")
        print("done seed", seed, flush=True)


if __name__ == "__main__":
    main()
