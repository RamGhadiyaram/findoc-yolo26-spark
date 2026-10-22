"""Score pipeline output against ground truth, with a full-page Tesseract baseline.

Metrics (all on the held-out synthetic test split):
  * detection precision / recall at IoU >= 0.5, per class and overall
  * field-level OCR exact match (whitespace-insensitive) on correctly localized fields
  * baseline: full-page Tesseract, counting a field as captured if its exact
    string appears anywhere in the page text. This baseline cannot say WHICH
    field a string belongs to, so it is a generous upper bound on what
    page-level OCR gives you, not a like-for-like extractor.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import pandas as pd
import pytesseract
from PIL import Image


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def norm(s: str) -> str:
    return "".join(str(s).split())


def score(parquet_dir: str, truth_json: str, images_dir: str | None = None, baseline: bool = True,
          min_conf: float = 0.0) -> dict:
    truth = json.load(open(truth_json))
    det = pd.read_parquet(parquet_dir)
    det = det[det["confidence"] >= min_conf].reset_index(drop=True)
    det["name"] = det["path"].map(os.path.basename)

    tp = defaultdict(int); fn = defaultdict(int); fp = defaultdict(int)
    ocr_ok = defaultdict(int); ocr_n = defaultdict(int)
    dup_conf: list[float] = []; top_conf: list[float] = []; ocr_errors: list[dict] = []
    for name, fields in truth.items():
        d = det[det["name"] == name].sort_values("confidence", ascending=False)
        used = set()
        for f in fields:
            best, best_iou = None, 0.0
            for i, r in d.iterrows():
                if i in used or r["field_class"] != f["cls"]:
                    continue
                v = iou(f["box"], (r["x1"], r["y1"], r["x2"], r["y2"]))
                if v > best_iou:
                    best, best_iou = i, v
            if best is not None and best_iou >= 0.5:
                used.add(best); tp[f["cls"]] += 1
                if f["text"]:
                    ocr_n[f["cls"]] += 1
                    good = norm(d.loc[best, "text"]) == norm(f["text"])
                    ocr_ok[f["cls"]] += int(good)
                    if not good:
                        ocr_errors.append({"cls": f["cls"], "truth": f["text"], "read": d.loc[best, "text"]})
                top_conf.append(float(d.loc[best, "confidence"]))
            else:
                fn[f["cls"]] += 1
        for i, r in d.iterrows():
            if i not in used:
                fp[r["field_class"]] += 1
                # a duplicate: unmatched box that still overlaps a same-class ground-truth field at IoU >= 0.5
                if any(f["cls"] == r["field_class"] and iou(f["box"], (r["x1"], r["y1"], r["x2"], r["y2"])) >= 0.5 for f in fields):
                    dup_conf.append(float(r["confidence"]))

    classes = sorted(set(tp) | set(fn) | set(fp))
    per = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per[c] = {"precision": round(p, 4), "recall": round(r, 4),
                  "ocr_exact": round(ocr_ok[c] / ocr_n[c], 4) if ocr_n[c] else None, "n": tp[c] + fn[c]}
    T, F_, P = sum(tp.values()), sum(fn.values()), sum(fp.values())
    text_classes = [c for c in classes if ocr_n[c]]
    n_text_truth = sum(1 for fs in truth.values() for f in fs if f["text"])
    out = {
        "overall_precision": round(T / (T + P), 4) if T + P else 0.0,
        "overall_recall": round(T / (T + F_), 4) if T + F_ else 0.0,
        "pipeline_end_to_end_field_accuracy": round(sum(ocr_ok.values()) / n_text_truth, 4),
        "per_class": per,
        "duplicate_boxes": len(dup_conf),
        "duplicate_mean_conf": round(sum(dup_conf) / len(dup_conf), 3) if dup_conf else None,
        "top_box_mean_conf": round(sum(top_conf) / len(top_conf), 3) if top_conf else None,
        "missed_fields": int(sum(fn.values())),
        "ocr_errors": ocr_errors,
    }
    if baseline and images_dir:
        t0 = time.perf_counter(); hit = 0
        for name, fields in truth.items():
            page = norm(pytesseract.image_to_string(Image.open(os.path.join(images_dir, name))))
            hit += sum(norm(f["text"]) in page for f in fields if f["text"])
        out["fullpage_tesseract_field_capture"] = round(hit / n_text_truth, 4)
        out["fullpage_tesseract_seconds_per_doc"] = round((time.perf_counter() - t0) / len(truth), 3)
    return out


def sweep(parquet_dir: str, truth_json: str, thresholds: list[float]) -> list[dict]:
    """Detection precision, recall and F1 at each confidence threshold (no baseline, no OCR timing)."""
    rows = []
    for t in thresholds:
        r = score(parquet_dir, truth_json, baseline=False, min_conf=t)
        p, rc = r["overall_precision"], r["overall_recall"]
        rows.append({"conf": t, "precision": p, "recall": rc,
                     "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
                     "end_to_end": r["pipeline_end_to_end_field_accuracy"]})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("truth_json")
    ap.add_argument("--images"); ap.add_argument("--report")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true", help="print P/R/F1 over a confidence grid and exit")
    a = ap.parse_args()
    if a.sweep:
        grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
        res = sweep(a.parquet, a.truth_json, grid)
        for r in res:
            print(r)
        if a.report:
            json.dump(res, open(a.report, "w"), indent=2)
        raise SystemExit
    res = score(a.parquet, a.truth_json, a.images, min_conf=a.min_conf)
    print(json.dumps(res, indent=2))
    if a.report:
        json.dump(res, open(a.report, "w"), indent=2)
