"""Synthetic financial-form generator with exact ground truth.

Why synthetic: there is no public, field-level labeled KYC/financial
document dataset (FUNSD and SROIE use different label schemas, see
docs/DATASETS.md). To validate the *pipeline* end to end, a deterministic
generator gives exact boxes and exact field text, so detection and OCR can
both be scored without any annotation noise.

What this does and does not show: results on this data demonstrate that the
pipeline is correct and measurable. They are NOT evidence about accuracy on
real scanned documents, which is harder.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLASSES = ["account_number", "date", "amount", "name", "signature"]
IMG_W, IMG_H = 960, 640  # deliberately non-square to exercise letterboxing

_FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
]
_FONT_FILES = ["DejaVuSans.ttf", "DejaVuSansMono.ttf", "DejaVuSerif.ttf"]


@dataclass
class Field:
    cls: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    text: str


def _font(size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    for d in _FONT_DIRS:
        for name in rng.sample(_FONT_FILES, len(_FONT_FILES)):
            p = Path(d) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def fonts_available() -> list[str]:
    """Font files the generator can actually find here. Empty means PIL's default font is used,
    which changes every image. Recorded with each run so results from different machines can be compared."""
    return sorted(str(Path(d) / n) for d in _FONT_DIRS for n in _FONT_FILES if (Path(d) / n).exists())


def _value_for(cls: str, rng: random.Random) -> str:
    if cls == "account_number":
        return "".join(rng.choice("0123456789") for _ in range(10))
    if cls == "date":
        return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(2019, 2026)}"
    if cls == "amount":
        return f"${rng.randint(10, 99999):,}.{rng.randint(0, 99):02d}"
    if cls == "name":
        first = rng.choice(["Maria", "James", "Aisha", "Chen", "Olga", "Samir", "Lucia", "Tomas"])
        last = rng.choice(["Hughes", "Patel", "Okafor", "Nguyen", "Rossi", "Kim", "Silva", "Brandt"])
        return f"{first} {last}"
    return ""


def render_form(seed: int) -> tuple[Image.Image, list[Field]]:
    rng = random.Random(seed)
    bg = rng.randint(225, 250)
    img = Image.new("RGB", (IMG_W, IMG_H), (bg, bg, bg - rng.randint(0, 8)))
    d = ImageDraw.Draw(img)

    # page chrome so detection has to ignore non-field content
    d.rectangle([20, 20, IMG_W - 20, IMG_H - 20], outline=(120, 120, 120), width=2)
    d.text((40, 30), "FIRST NATIONAL TRUST  -  ACCOUNT FORM", fill=(60, 60, 60), font=_font(22, rng))

    rows = [(c, 90 + i * 100 + rng.randint(-8, 8)) for i, c in enumerate(CLASSES)]
    rng.shuffle(rows)
    fields: list[Field] = []
    for cls, y in rows:
        label_font = _font(16, rng)
        d.text((40, y + 6), cls.replace("_", " ").upper() + ":", fill=(90, 90, 90), font=label_font)
        x = rng.randint(300, 420)
        if cls == "signature":
            w, h = rng.randint(220, 300), rng.randint(46, 64)
            pts = [(x + i * (w // 12), y + h // 2 + int(np.sin(i * 1.3 + rng.random()) * h * 0.35)) for i in range(13)]
            d.line(pts, fill=(20, 20, 90), width=3)
            box = (x - 4, y, x + w + 4, y + h)
            fields.append(Field(cls, box, ""))
            continue
        text = _value_for(cls, rng)
        font = _font(rng.randint(24, 32), rng)
        l, t, r, b = d.textbbox((x, y), text, font=font)
        d.text((x, y), text, fill=(15, 15, 15), font=font)
        pad = 4
        fields.append(Field(cls, (l - pad, t - pad, r + pad, b + pad), text))

    arr = np.asarray(img).astype(np.float32)
    arr += np.random.default_rng(seed).normal(0, rng.uniform(1.0, 5.0), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    return img, fields


def _yolo_line(f: Field) -> str:
    x1, y1, x2, y2 = f.box
    cx, cy = (x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H
    w, h = (x2 - x1) / IMG_W, (y2 - y1) / IMG_H
    return f"{CLASSES.index(f.cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_dataset(out_dir: str | Path, n_train: int = 240, n_val: int = 60, n_test: int = 60, seed0: int = 1000) -> Path:
    """Write a YOLO-format dataset plus a ground-truth JSON for the test split."""
    out = Path(out_dir).resolve()
    splits = {"train": n_train, "val": n_val, "test": n_test}
    seed = seed0
    truth: dict[str, dict[str, list[dict]]] = {}
    for split, n in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, fields = render_form(seed)
            name = f"{split}_{i:04d}"
            img.save(out / "images" / split / f"{name}.jpg", quality=92)
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(_yolo_line(f) for f in fields) + "\n")
            if split in ("val", "test"):
                truth.setdefault(split, {})[f"{name}.jpg"] = [{"cls": f.cls, "box": f.box, "text": f.text} for f in fields]
            seed += 1
    for split in ("val", "test"):  # val truth is used only to choose the confidence threshold
        (out / f"{split}_truth.json").write_text(json.dumps(truth[split]))
    yaml = (
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (out / "financial_docs.yaml").write_text(yaml)
    return out / "financial_docs.yaml"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the synthetic financial-form dataset.")
    ap.add_argument("out_dir")
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    a = ap.parse_args()
    print(build_dataset(a.out_dir, a.train, a.val, a.test))
