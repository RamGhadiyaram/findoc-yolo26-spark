# findoc: reading financial form fields with YOLO26, Tesseract and Apache Spark

This repository is the code and data behind the paper in [`paper/`](paper/). It builds a pipeline that finds the important fields on a financial form (account number, date, amount, name, signature), reads the text in each field, and runs the whole thing at scale as an Apache Spark job.

## What each tool does

Two tools do two different jobs. They do not compete.

| Tool | Job | Output |
|---|---|---|
| **YOLO26** (object detector) | Looks at the page image and finds *where* each field is | A box and a label per field, for example "amount, at these pixels" |
| **Tesseract** (OCR engine) | Reads the characters inside one box | The text, for example `$1,510.35` |
| **Apache Spark** | Runs the two steps above over many images in parallel | A table with one row per field: file, field label, text, confidence |

YOLO26 cannot read text, and Tesseract on a whole page cannot say which string is the account number. Together they give labeled text: every extracted string already knows which field it belongs to.

```
image  ->  YOLO26 finds field boxes  ->  crop each box  ->  Tesseract reads the crop  ->  row: (file, field, text, confidence)
                 \_______________ one Spark task per group of images (partition) _______________/
```

## What the paper says, in plain words

1. **Putting a model inside Spark is easy to get wrong.** The first version of this pipeline had several defects that only showed up when the code was run, for example Spark's `image` source returning raw pixels instead of file bytes. The paper lists each one and the fix (Table I).
2. **Load the model once per group of images, not once per image.** Doing it per partition with `mapInPandas` was 1.7 times faster than per row on two CPU cores, and 2.1 times faster on a GPU.
3. **The new end-to-end YOLO head still needs cleanup.** At the default threshold it produced 90 duplicate boxes, and precision fell to 75.9%. Removing duplicates of the same class before reading brought precision to 98.7%.
4. **The accuracy numbers come with limits.** The data is synthetic and easy, so absolute accuracy says little about real scans. A plain full-page Tesseract run also finds almost every string, but it cannot say which field a string belongs to, so the paper does not claim an accuracy win.
5. **A GPU did not help this small model.** Training was much faster on a T4 GPU, inference was not, and detection and reading results stayed the same.

## Results

All numbers are produced by the code here from the files in `results/` (CPU) and `results_gpu/` (one Colab T4 run). The paper's numbers are generated from those files, not typed by hand. Full tables are in [`docs/RESULTS.md`](docs/RESULTS.md).

| Measure (60 test forms, 240 text fields) | CPU, 2 cores | One T4 GPU |
|---|---|---|
| Detection precision / recall (class-wise suppression) | 98.7% / 99.3% | 99.7% / 100.0% |
| Text fields read correctly end to end | 97.1% (95% interval 94.1 to 98.6) | 98.3% (95% interval 95.8 to 99.4) |
| Row-wise vs per-partition inference speedup | 1.7x | 2.1x |
| Training time, 15 epochs | 32 min | 117 s |

Each column is a single training run. Small differences between columns are not significant.

Not measured: real scanned documents, multi-node or multi-GPU clusters, Spark GPU scheduling with RAPIDS, AWS EMR, and Amazon Textract. The AWS notes in `docs/AWS_DEPLOYMENT.md` are a design, not a result.

## Requirements

| | Needed | Tested with |
|---|---|---|
| OS | Linux or macOS | Ubuntu 24.04 |
| Python | 3.10 to 3.13 | 3.13 |
| Java | 17 or 21 | OpenJDK 21 |
| Tesseract | 5.x (system package) | 5.3.4 |
| Fonts | `fonts-dejavu-core` (the data generator stops with an error without it) | DejaVu |
| PySpark | **4.0.x** (3.5 fails on JDK 21, see CHANGELOG) | 4.0.1 |
| Hardware | 2 cores and 7 GB RAM are enough. No GPU needed. | 2 vCPU, 7 GB |

Details: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md).

## Quick start

```bash
sudo apt-get install -y openjdk-21-jre-headless tesseract-ocr fonts-dejavu-core   # macOS: brew install openjdk@21 tesseract
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # optional, avoids a CUDA download
make install
make test            # unit tests plus a real Spark + YOLO26 + Tesseract smoke test (about 1 min)
```

## Reproduce the CPU results

```bash
make e2e             # data -> train -> protocol -> pattern -> paper artifacts (about 50 min on 2 cores)
make paper           # builds paper/paper.pdf
```

| Step | Command | What it does | Time (2 cores) |
|---|---|---|---|
| data | `make data` | 360 synthetic forms with exact boxes and text | 20 s |
| train | `make train` | fine-tune YOLO26n for 15 epochs, export ONNX with dynamic batch | 32 min |
| protocol | `make protocol` | choose the confidence threshold on validation, score test once, plus a full-page Tesseract baseline | 6 min |
| pattern | `make pattern` | row-wise UDF vs per-partition inference with the real model | 5 min |

Training is not bit-for-bit repeatable across machines, so a rerun gives numbers close to, not identical to, the published ones. The test images themselves are identical when the same fonts are installed (SHA-256 recorded in `results_gpu/environment.json`).

## Run it on your own images

```bash
python -m findoc.pipeline "path/to/images" --model work/runs/synth/weights/best.onnx --out out_parquet --conf 0.15 --nms-iou 0.5
```

The output is a Parquet table with one row per detected field and the columns `path, field_class, confidence, x1, y1, x2, y2, text`. `--nms-iou 0.5` removes duplicate boxes of the same class before reading. Leave it off to keep the raw end-to-end head output. The model here is trained on the synthetic forms, so for real documents you would fine-tune on your own labeled boxes first (`python -m findoc.train`).

## Reproduce the GPU run (free Colab)

1. Open `notebooks/colab_gpu_run.ipynb` in Google Colab.
2. Runtime, Change runtime type, T4 GPU.
3. Runtime, Run all (30 to 60 minutes).
4. Download `results_gpu.zip` from the last cell.

Details and caveats: [`docs/COLAB.md`](docs/COLAB.md). This runs Spark in local mode on one GPU. It does not test Spark's multi-GPU scheduling.

## Layout

```
src/findoc/   synth.py  train.py  export.py  pipeline.py  evaluate.py  protocol.py
benchmarks/   inference_pattern_benchmark.py
notebooks/    colab_gpu_run.ipynb
tests/        unit tests and an end-to-end Spark test
results/      JSON from the CPU runs (the paper reads these)
results_gpu/  JSON from the Colab T4 run
paper/        LaTeX source, figures, results_macros.tex (generated)
docs/         REQUIREMENTS, RESULTS, DATASETS, COLAB, AWS_DEPLOYMENT
```

## Scope and honesty

Synthetic data only, generated by `src/findoc/synth.py`. See `docs/DATASETS.md` for why public datasets were not used, and `CHANGELOG.md` for the defects found and the mistakes made, including a GPU run that gave misleading accuracy because the fonts were missing.

## License

MIT.
