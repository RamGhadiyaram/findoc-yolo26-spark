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
