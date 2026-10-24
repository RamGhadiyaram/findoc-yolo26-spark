import json
from pathlib import Path

import numpy as np

from findoc import synth
from findoc.evaluate import iou, norm


def test_render_is_deterministic():
    a_img, a_f = synth.render_form(7)
    b_img, b_f = synth.render_form(7)
    assert a_f == b_f
    assert np.array_equal(np.asarray(a_img), np.asarray(b_img))  # every pixel, not a prefix


def test_form_has_all_classes_and_valid_boxes():
    img, fields = synth.render_form(11)
    assert {f.cls for f in fields} == set(synth.CLASSES)
    for f in fields:
        x1, y1, x2, y2 = f.box
        assert 0 <= x1 < x2 <= img.width and 0 <= y1 < y2 <= img.height


def test_dataset_layout(tmp_path: Path):
    yaml = synth.build_dataset(tmp_path, n_train=3, n_val=2, n_test=2)
    assert yaml.exists()
    assert len(list((tmp_path / "images" / "train").glob("*.jpg"))) == 3
    lab = (tmp_path / "labels" / "train" / "train_0000.txt").read_text().split("\n")[0].split()
    assert len(lab) == 5 and 0 <= int(lab[0]) < len(synth.CLASSES)
    assert len(json.loads((tmp_path / "test_truth.json").read_text())) == 2


def test_iou_and_norm():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert norm(" 12 34\n") == "1234"


def test_class_nms_suppresses_same_class_duplicates_only():
    from findoc.pipeline import class_nms
    cands = [("a", .9, 0, 0, 10, 10), ("a", .5, 1, 1, 10, 10), ("b", .6, 0, 0, 10, 10), ("a", .4, 50, 50, 60, 60)]
    kept = class_nms(cands, 0.5)
    assert [(c[0], c[1]) for c in kept] == [("a", .9), ("b", .6), ("a", .4)]


def test_missing_fonts_fail_loudly(monkeypatch):
    import random

    import pytest

    monkeypatch.setattr(synth, "_FONT_DIRS", ["/nonexistent"])
    monkeypatch.delenv("FINDOC_ALLOW_DEFAULT_FONT", raising=False)
    with pytest.raises(RuntimeError, match="fonts-dejavu-core"):
        synth._font(20, random.Random(0))
    monkeypatch.setenv("FINDOC_ALLOW_DEFAULT_FONT", "1")
    assert synth._font(20, random.Random(0)) is not None


def test_prepare_crop_defaults_are_the_plain_crop():
    import cv2

    from findoc.pipeline import prepare_crop

    im = (np.arange(60 * 100 * 3) % 251).astype(np.uint8).reshape(60, 100, 3)
    plain = cv2.cvtColor(im[10:30, 20:60], cv2.COLOR_BGR2GRAY)
    assert np.array_equal(prepare_crop(im, (20.4, 10.9, 60.2, 30.7)), plain)  # int() truncation, no padding
    assert prepare_crop(im, (20, 10, 60, 30), pad=4).shape == (28, 48)
    assert prepare_crop(im, (20, 10, 60, 30), scale=2.0).shape == (40, 80)
    assert set(np.unique(prepare_crop(im, (20, 10, 60, 30), binarize=True))) <= {0, 255}
    assert prepare_crop(im, (500, 500, 600, 600)) is None


def test_rule_patterns_match_generator_formats():
    from findoc.rules_baseline import PATTERNS

    page = "ACCOUNT FORM  Maria Rossi 5055730749 on 10/14/2020 paid $29,421.91"
    assert PATTERNS["account_number"].findall(page) == ["5055730749"]
    assert PATTERNS["date"].findall(page) == ["10/14/2020"]
    assert PATTERNS["amount"].findall(page) == ["$29,421.91"]
    assert PATTERNS["name"].findall(page) == ["Maria Rossi"]
