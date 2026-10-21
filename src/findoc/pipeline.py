"""Partition-level Spark pipeline: YOLO26 field detection + Tesseract OCR.

Corrected version of the paper's Listing 2. Design points, each validated by
tests/ and by scripts/run_e2e.sh:

* Images are read with Spark's ``binaryFile`` source, which returns the
  *encoded* file bytes (JPEG/PNG) and works for local paths, HDFS and s3://.
  (Spark's ``image`` source returns decoded raw BGR pixels instead, which is
  not decodable with an image library, so it is not used here.)
* The model is loaded once per partition inside ``mapInPandas``, shipped to
  workers with ``SparkContext.addFile``.
* Inference is one batched call per Arrow batch. No manual resize: Ultralytics
  letterboxes internally and returns boxes in original-image pixels.
* Each detected field is cropped from the original image and read by
  Tesseract (single-line mode). Signatures are not OCR'd.
* Output is long format, one row per detected field.
* Aggregation uses cuDF when RAPIDS is installed and pandas otherwise: each
  partition emits a partial (sum, count) per class and Spark does the final
  reduce, the standard two-phase aggregation pattern.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pyspark import SparkFiles, TaskContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType,
)

try:  # optional GPU DataFrame library
    import cudf  # type: ignore

    HAVE_CUDF = True
except ImportError:
    cudf = None
    HAVE_CUDF = False

FIELD_SCHEMA = StructType([
    StructField("path", StringType()),
    StructField("field_class", StringType()),
    StructField("confidence", DoubleType()),
    StructField("x1", DoubleType()), StructField("y1", DoubleType()),
    StructField("x2", DoubleType()), StructField("y2", DoubleType()),
    StructField("text", StringType()),
])
PARTIAL_SCHEMA = StructType([
    StructField("field_class", StringType()),
    StructField("conf_sum", DoubleType()),
    StructField("n", LongType()),
])
NO_OCR_CLASSES = {"signature"}


def build_spark(master: str = "local[*]", gpu: bool = False, arrow_batch: int = 16) -> SparkSession:
    b = (
        SparkSession.builder.appName("findoc-yolo26")
        .master(master)
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_batch))
        .config("spark.ui.showConsoleProgress", "false")
    )
    if gpu:
        # One GPU per executor and one per task; Spark hands each task its device index.
        b = (
            b.config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.python.gpu.enabled", "true")
            .config("spark.rapids.sql.concurrentGpuTasks", "1")
            .config("spark.executor.resource.gpu.amount", "1")
            .config("spark.task.resource.gpu.amount", "1")
            .config("spark.executor.resource.gpu.discoveryScript", os.environ.get(
                "GPU_DISCOVERY_SCRIPT", "/opt/sparkRapidsPlugin/getGpusResources.sh"))
        )
    return b.getOrCreate()


def _task_device() -> str | int:
    """GPU index Spark assigned to this task, else CPU."""
    ctx = TaskContext.get()
    if ctx is not None:
        gpus = ctx.resources().get("gpu")
        if gpus is not None and gpus.addresses:
            return int(gpus.addresses[0])
    return "cpu"


def class_nms(cands: list[tuple], iou_thr: float) -> list[tuple]:
    """Greedy per-class suppression. cands items: (cls, conf, x1, y1, x2, y2). Highest confidence wins."""
    keep: list[tuple] = []
    for c in sorted(cands, key=lambda t: -t[1]):
        ok = True
        for k in keep:
            if k[0] != c[0]:
                continue
            ix1, iy1, ix2, iy2 = max(c[2], k[2]), max(c[3], k[3]), min(c[4], k[4]), min(c[5], k[5])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (c[4] - c[2]) * (c[5] - c[3]) + (k[4] - k[2]) * (k[5] - k[3]) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def make_detect_fn(model_basename: str, ocr_lang: str = "eng", conf: float = 0.25, nms_iou: float = 0.0):
    """Return the mapInPandas function. Closure holds only small picklable values."""

    def detect_partition(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        from ultralytics import YOLO  # imported on the worker

        device = _task_device()
        model = YOLO(SparkFiles.get(model_basename), task="detect")  # once per partition
        for batch in batches:
            imgs, paths = [], []
            for p, raw in zip(batch["path"], batch["content"]):
                im = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
                if im is not None:  # skip undecodable files instead of crashing the job
                    imgs.append(im)
                    paths.append(p)
            rows = []
            if imgs:
                results = model(imgs, device=device, conf=conf, verbose=False)  # one batched call
                for path, im, res in zip(paths, imgs, results):
                    h, w = im.shape[:2]
                    cands = []
                    for b in res.boxes:
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                        cands.append((res.names[int(b.cls.item())], float(b.conf.item()), x1, y1, x2, y2))
                    if nms_iou > 0:  # optional post-hoc suppression, applied before the costly OCR step
                        cands = class_nms(cands, nms_iou)
                    for cls, cf, x1, y1, x2, y2 in cands:
                        text = ""
                        if cls not in NO_OCR_CLASSES:
                            crop = im[max(int(y1), 0):min(int(y2), h), max(int(x1), 0):min(int(x2), w)]
                            if crop.size:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                text = pytesseract.image_to_string(
                                    gray, lang=ocr_lang, config="--psm 7").strip()
                        rows.append((path, cls, cf, x1, y1, x2, y2, text))
            yield pd.DataFrame(rows, columns=[f.name for f in FIELD_SCHEMA.fields])

    return detect_partition


def partial_aggregate(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Per-partition (sum, count) of confidence by class; cuDF if available."""
    for pdf in batches:
        if pdf.empty:
            yield pd.DataFrame(columns=[f.name for f in PARTIAL_SCHEMA.fields])
            continue
        if HAVE_CUDF:
            g = cudf.from_pandas(pdf[["field_class", "confidence"]])
            agg = g.groupby("field_class")["confidence"].agg(["sum", "count"]).reset_index().to_pandas()
        else:
            agg = pdf.groupby("field_class")["confidence"].agg(["sum", "count"]).reset_index()
        agg.columns = ["field_class", "conf_sum", "n"]
        yield agg


def run(images: str, model: str, out: str, master: str = "local[*]", gpu: bool = False,
        partitions: int | None = None, glob: str = "*.jpg", conf: float = 0.25,
        nms_iou: float = 0.0) -> dict:
    spark = build_spark(master, gpu)
    spark.sparkContext.addFile(model)
    t0 = time.perf_counter()

    df = spark.read.format("binaryFile").option("pathGlobFilter", glob).option("recursiveFileLookup", "true").load(images)
    df = df.select("path", "content")
    if partitions:
        df = df.repartition(partitions)

    fields = df.mapInPandas(make_detect_fn(os.path.basename(model), conf=conf, nms_iou=nms_iou), schema=FIELD_SCHEMA)
    fields.write.mode("overwrite").parquet(out)

    written = spark.read.parquet(out)  # re-read so the timing covers the full job exactly once
    n_fields, n_docs = written.count(), written.select("path").distinct().count()
    elapsed = time.perf_counter() - t0

    summary = (
        written.mapInPandas(partial_aggregate, schema=PARTIAL_SCHEMA)
        .groupBy("field_class").agg(F.sum("conf_sum").alias("s"), F.sum("n").alias("n"))
        .withColumn("mean_confidence", F.col("s") / F.col("n")).drop("s")
        .orderBy("field_class").toPandas()
    )
    spark.stop()
    return {"docs": n_docs, "fields": n_fields, "seconds": round(elapsed, 2),
            "docs_per_second": round(n_docs / elapsed, 3), "cudf": HAVE_CUDF, "conf": conf, "nms_iou": nms_iou,
            "per_class": summary.to_dict("records")}


def main(argv: list[str] | None = None) -> dict:
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="Directory or URI (file://, s3://) of images")
    ap.add_argument("--model", required=True, help="Path to .onnx or .pt checkpoint")
    ap.add_argument("--out", required=True, help="Output parquet directory")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--gpu", action="store_true", help="Enable RAPIDS GPU scheduling configs")
    ap.add_argument("--partitions", type=int)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--nms-iou", type=float, default=0.0, help="if > 0, apply class-wise NMS at this IoU before OCR (default: off, NMS-free)")
    ap.add_argument("--report", help="Write run summary JSON here")
    a = ap.parse_args(argv)
    res = run(a.images, a.model, a.out, a.master, a.gpu, a.partitions, a.glob, a.conf, a.nms_iou)
    print(json.dumps(res, indent=2))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()
