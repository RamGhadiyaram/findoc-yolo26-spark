# findoc: YOLO26 + Tesseract field extraction on Apache Spark

Detects fields on financial forms (account number, date, amount, name, signature) with YOLO26, reads each crop with Tesseract, and runs the whole thing as a partition-level Spark job (`mapInPandas`, model loaded once per partition). Companion code for the paper in `paper/`.

Measured results are in [`docs/RESULTS.md`](docs/RESULTS.md). The headline, honestly: on clean synthetic forms the pipeline reaches 97.1% end-to-end field accuracy, full-page Tesseract reads the same strings at least as well (without saying which field is which), and loading the model once per partition is 1.7x faster than per row on two CPU cores. No GPU, EMR, or Textract numbers exist yet.

## Requirements

| | Needed | Tested with |
|---|---|---|
| OS | Linux or macOS | Ubuntu 24.04 |
| Python | 3.10 to 3.13 | 3.13 |
| Java | 17 or 21 | OpenJDK 21 |
| Tesseract | 5.x (system package) | 5.3.4 |
| PySpark | **4.0.x** (3.5 fails on JDK 21, see CHANGELOG) | 4.0.1 |
| Hardware | 2 cores, 7 GB RAM is enough; no GPU needed | 2 vCPU, 7 GB |

Details and GPU notes: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md).

## Quick start

```bash
sudo apt-get install -y openjdk-21-jre-headless tesseract-ocr fonts-dejavu-core   # macOS: brew install openjdk@21 tesseract
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # optional, avoids a CUDA download
make install
make test            # unit tests plus a real Spark + YOLO26 + Tesseract smoke test (about 1 min)
```

## Reproduce every number in the paper

```bash
make e2e             # data -> train -> protocol -> pattern -> paper artifacts (about 50 min on 2 cores)
make paper           # builds paper/paper.pdf
```

| Step | Command | What it does | Time (2 cores) |
|---|---|---|---|
| data | `make data` | 360 synthetic forms with exact boxes and text (byte-identical on regeneration) | 20 s |
| train | `make train` | fine-tune YOLO26n 15 epochs, export ONNX with dynamic batch | 32 min |
| protocol | `make protocol` | choose confidence on val, score test once, per variant, plus full-page Tesseract baseline | 6 min |
| pattern | `make pattern` | row-wise UDF vs per-partition inference with the real model | 5 min |

## Layout

```
src/findoc/   synth.py  train.py  export.py  pipeline.py  evaluate.py  protocol.py
benchmarks/   inference_pattern_benchmark.py
tests/        unit tests and an end-to-end Spark test
results/      JSON produced by the runs above (the paper reads these)
paper/        LaTeX source, Graphviz figures, results_macros.tex (generated)
docs/         REQUIREMENTS, RESULTS, DATASETS, AWS_DEPLOYMENT
```

## Scope

Synthetic data only, CPU only. The GPU configuration (`--gpu`) and the AWS deployment notes are untested. See `docs/DATASETS.md` for why public datasets were not used and `CHANGELOG.md` for the defects found and the mistakes made along the way.

## License

MIT.

## GPU run

A ready-made Colab notebook lives in `notebooks/colab_gpu_run.ipynb` (see `docs/COLAB.md`). It writes to `results_gpu/`.
