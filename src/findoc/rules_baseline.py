"""Rule-based baseline: full-page Tesseract followed by one regular expression per field class.

This asks how much the detector actually buys on this synthetic data. The patterns are the formats the
generator in synth.py produces (10 digits, mm/dd/yyyy, $x,xxx.xx, two capitalized words), so the baseline is
generous: real forms would not be this regular. Each class takes the FIRST match on the page. A field counts
as correct when that string equals the ground truth after whitespace removal. No threshold is tuned.

    python -m findoc.rules_baseline work/data test results/rules_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pytesseract
from PIL import Image

from findoc.evaluate import norm

PATTERNS = {
    "account_number": re.compile(r"(?<!\d)\d{10}(?!\d)"),
    "date": re.compile(r"(?<!\d)\d{2}/\d{2}/\d{4}(?!\d)"),
    "amount": re.compile(r"\$\d{1,3}(?:,\d{3})*\.\d{2}"),
    "name": re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),
}


def run(data: str, split: str) -> dict:
    truth = json.load(open(Path(data) / f"{split}_truth.json"))
    ok = {c: 0 for c in PATTERNS}
    n = {c: 0 for c in PATTERNS}
    any_ok = {c: 0 for c in PATTERNS}
    t0 = time.perf_counter()
    for name, fields in truth.items():
        page = pytesseract.image_to_string(Image.open(Path(data) / "images" / split / name))
        for f in fields:
            c = f["cls"]
            if c not in PATTERNS or not f["text"]:
                continue
            matches = PATTERNS[c].findall(page)
            n[c] += 1
            ok[c] += int(bool(matches) and norm(matches[0]) == norm(f["text"]))
            any_ok[c] += int(any(norm(m) == norm(f["text"]) for m in matches))
    tot, tot_any = sum(n.values()), sum(any_ok.values())
    return {"split": split, "fields": tot, "first_match_accuracy": round(sum(ok.values()) / tot, 4),
            "any_match_accuracy": round(tot_any / tot, 4),
            "per_class_first_match": {c: round(ok[c] / n[c], 4) for c in n},
            "seconds_per_doc": round((time.perf_counter() - t0) / len(truth), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("split"); ap.add_argument("out")
    a = ap.parse_args()
    res = run(a.data, a.split)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
