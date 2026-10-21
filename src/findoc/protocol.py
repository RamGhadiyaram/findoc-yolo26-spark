"""The evaluation protocol, end to end.

For each detector variant (NMS-free as exported, and with class-wise NMS before OCR):
  1. run the pipeline on the VALIDATION split at a low confidence floor,
  2. choose the confidence threshold that maximizes detection F1 on validation,
  3. run the pipeline ONCE on the TEST split at that threshold and score it.
The test split never influences a choice. The untuned default (conf 0.25, NMS-free)
is also scored on test and reported, so the effect of tuning is visible.

    python -m findoc.protocol work/data work/runs/synth/weights/best.onnx results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc.evaluate import score, sweep
from findoc.pipeline import run

GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
VARIANTS = {"nms_free": 0.0, "class_nms": 0.5}


def main(data: str, model: str, results: str, partitions: int = 2) -> dict:
    data_p, res_p = Path(data), Path(results)
    res_p.mkdir(parents=True, exist_ok=True)
    work = Path("work")
    val_imgs, test_imgs = str(data_p / "images" / "val"), str(data_p / "images" / "test")
    val_truth, test_truth = str(data_p / "val_truth.json"), str(data_p / "test_truth.json")
    summary: dict = {}
    baseline: dict = {}

    for name, nms in VARIANTS.items():
        out_val = str(work / f"out_val_{name}")
        run(val_imgs, model, out_val, partitions=partitions, conf=GRID[0], nms_iou=nms)
        rows = sweep(out_val, val_truth, GRID)
        best = max(rows, key=lambda r: (r["f1"], r["conf"]))
        (res_p / f"val_sweep_{name}.json").write_text(json.dumps(rows, indent=2))

        out_test = str(work / f"out_test_{name}")
        rep = run(test_imgs, model, out_test, partitions=partitions, conf=best["conf"], nms_iou=nms)
        ev = score(out_test, test_truth, test_imgs, baseline=not baseline)
        if not baseline:
            baseline = {k: ev[k] for k in ev if k.startswith("fullpage")}
        ev.update(baseline)
        ev["chosen_conf"], ev["val_f1_at_chosen"] = best["conf"], best["f1"]
        (res_p / f"run_report_{name}.json").write_text(json.dumps(rep, indent=2))
        (res_p / f"eval_report_{name}.json").write_text(json.dumps(ev, indent=2))
        summary[name] = {"chosen_conf": best["conf"], "val_f1": best["f1"], "precision": ev["overall_precision"],
                         "recall": ev["overall_recall"], "end_to_end": ev["pipeline_end_to_end_field_accuracy"]}

    # untuned default for transparency
    out_def = str(work / "out_test_default")
    rep = run(test_imgs, model, out_def, partitions=partitions, conf=0.25, nms_iou=0.0)
    ev = score(out_def, test_truth, test_imgs, baseline=False)
    ev.update(baseline)
    (res_p / "run_report_default.json").write_text(json.dumps(rep, indent=2))
    (res_p / "eval_report_default.json").write_text(json.dumps(ev, indent=2))
    summary["default_conf0.25_nms_free"] = {"precision": ev["overall_precision"], "recall": ev["overall_recall"],
                                           "end_to_end": ev["pipeline_end_to_end_field_accuracy"]}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("model"); ap.add_argument("results")
    ap.add_argument("--partitions", type=int, default=2)
    a = ap.parse_args()
    main(a.data, a.model, a.results, a.partitions)
