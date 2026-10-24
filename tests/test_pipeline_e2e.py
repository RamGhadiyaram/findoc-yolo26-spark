"""End-to-end smoke test: real Spark, real YOLO26 weights, real Tesseract.

Uses the pretrained COCO yolo26n.pt (auto-downloaded), so it checks that the
plumbing is correct (decode -> batched detect -> crop -> OCR -> parquet ->
aggregate), not detection quality. Quality is measured by `make protocol`.
"""
import pandas as pd
import pytest

from findoc import synth
from findoc.pipeline import FIELD_SCHEMA, run

pytestmark = pytest.mark.e2e


def test_pipeline_runs_and_writes_expected_schema(tmp_path, monkeypatch):
    from ultralytics import YOLO

    monkeypatch.chdir(tmp_path)  # Ultralytics downloads weights into the cwd
    YOLO("yolo26n.pt")           # the pipeline ships a real file to workers, so it must exist first
    synth.build_dataset(tmp_path / "d", n_train=1, n_val=1, n_test=4)
    res = run(str(tmp_path / "d" / "images" / "test"), str(tmp_path / "yolo26n.pt"), str(tmp_path / "out"),
              master="local[2]", partitions=2)
    df = pd.read_parquet(tmp_path / "out")
    assert list(df.columns) == [f.name for f in FIELD_SCHEMA.fields]
    assert res["docs"] <= 4 and res["seconds"] > 0
