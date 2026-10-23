"""Measured comparison of three ways to run the real YOLO26 model inside Spark.

  row_udf          plain Python UDF; loads the model inside the function, so
                   once per row. This is the anti-pattern.
  partition_b1     mapInPandas; model loaded once per partition; Arrow batch
                   size 1, i.e. no batching of images into one model call.
  partition_b16    mapInPandas; model loaded once per partition; Arrow batch
                   size 16, i.e. one batched model call per 16 images.

Detection only (no OCR) so the comparison isolates the inference pattern.
Same images, same model, same Spark session settings. Each configuration is
run `--repeats` times in a fresh JVM-warm session; the first run of each is a
warm-up and is discarded.

    python benchmarks/inference_pattern_benchmark.py IMAGES_DIR MODEL.onnx \
        --docs 48 --partitions 2 --repeats 3 --out benchmarks/results/pattern.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import cv2
import numpy as np
import pandas as pd
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType, StringType, StructField, StructType

DEVICE = int(os.environ["FINDOC_DEVICE"]) if os.environ.get("FINDOC_DEVICE", "cpu").isdigit() else os.environ.get("FINDOC_DEVICE", "cpu")
OUT_SCHEMA = StructType([StructField("path", StringType()), StructField("n_boxes", LongType())])


def _decode(raw) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)


def make_row_udf(model_basename: str):
    @udf(returnType=LongType())
    def row_wise(raw):
        from ultralytics import YOLO  # model constructed on EVERY call

        model = YOLO(SparkFiles.get(model_basename), task="detect")
        im = _decode(raw)
        return len(model(im, device=DEVICE, verbose=False)[0].boxes) if im is not None else 0

    return row_wise


def make_partition_fn(model_basename: str):
    def detect(batches):
        from ultralytics import YOLO

        model = YOLO(SparkFiles.get(model_basename), task="detect")  # once per partition
        for b in batches:
            imgs, paths = [], []
            for p, raw in zip(b["path"], b["content"]):
                im = _decode(raw)
                if im is not None:
                    imgs.append(im)
                    paths.append(p)
            counts = [len(r.boxes) for r in model(imgs, device=DEVICE, verbose=False)] if imgs else []
            yield pd.DataFrame({"path": paths, "n_boxes": counts})

    return detect


def session(batch: int) -> SparkSession:
    return (
        SparkSession.builder.master("local[2]").appName("pattern-bench")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(batch))
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )


def run_once(mode: str, images: str, model: str, docs: int, partitions: int) -> tuple[float, int]:
    batch = 16 if mode == "partition_b16" else 1
    spark = session(batch)
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.addFile(model)
    base = os.path.basename(model)
    df = (
        spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").load(images)
        .select("path", "content").limit(docs).repartition(partitions).cache()
    )
    assert df.count() == docs
    t0 = time.perf_counter()
    if mode == "row_udf":
        total = df.select(make_row_udf(base)("content").alias("n")).agg({"n": "sum"}).collect()[0][0]
    else:
        total = df.mapInPandas(make_partition_fn(base), schema=OUT_SCHEMA).agg({"n_boxes": "sum"}).collect()[0][0]
    dt = time.perf_counter() - t0
    spark.stop()
    return dt, int(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images"); ap.add_argument("model")
    ap.add_argument("--docs", type=int, default=48)
    ap.add_argument("--partitions", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/results/pattern.json")
    a = ap.parse_args()

    result = {"docs": a.docs, "partitions": a.partitions, "repeats": a.repeats, "local_cores": 2,
              "device": str(DEVICE), "scope": f"device={DEVICE}, local[2], real fine-tuned YOLO26n ONNX, detection only", "modes": {}}
    for mode in ["row_udf", "partition_b1", "partition_b16"]:
        run_once(mode, a.images, a.model, a.docs, a.partitions)  # warm-up, discarded
        times, boxes = [], set()
        for _ in range(a.repeats):
            dt, n = run_once(mode, a.images, a.model, a.docs, a.partitions)
            times.append(dt); boxes.add(n)
        result["modes"][mode] = {
            "seconds_mean": round(statistics.mean(times), 2),
            "seconds_sd": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
            "docs_per_second": round(a.docs / statistics.mean(times), 3),
            "all_seconds": [round(t, 2) for t in times],
            "total_boxes": sorted(boxes),
        }
        print(mode, result["modes"][mode], flush=True)
    base = result["modes"]["row_udf"]["seconds_mean"]
    for m in result["modes"].values():
        m["speedup_vs_row_udf"] = round(base / m["seconds_mean"], 2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
