# Requirements

## Verified environment

Everything in `docs/RESULTS.md` was produced on this configuration:

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04, Linux 6.x | any recent Linux or macOS should work |
| CPU | 2 vCPU Intel Xeon, 7 GB RAM | no GPU needed for the CPU results; the GPU replication used a Colab T4 (see COLAB.md) |
| Python | 3.13 | 3.10 to 3.13 supported |
| Java | OpenJDK 21 | 17 or 21; Spark 4.0 needs 17+ |
| Apache Spark (PySpark) | 4.0.1 | see "Why Spark 4" below |
| Ultralytics | 8.4.x | provides YOLO26 |
| PyTorch | 2.14 | pulled in by Ultralytics |
| ONNX Runtime | 1.29 | CPU inference of the exported model |
| Tesseract | 5.3.4 | system package, not pip |

## Install

System packages:

```bash
# Debian/Ubuntu
sudo apt-get install -y openjdk-21-jre-headless tesseract-ocr fonts-dejavu-core
# macOS
brew install openjdk@21 tesseract
```

Python:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

On a CPU-only machine, install the CPU build of PyTorch first to avoid a multi-GB CUDA download:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Why Spark 4

`mapInPandas` moves data through Apache Arrow. PySpark 3.5.x bundles an Arrow
Java build that fails on JDK 21 with `sun.misc.Unsafe or
java.nio.DirectByteBuffer.<init>(long, int) not available`, even with
`--add-opens` flags. PySpark 4.0 works on JDK 21. If you must stay on Spark 3.5,
use JDK 11 or 17.

## GPU deployment (optional). The Spark GPU-scheduling path below was not exercised; the reported GPU run used one GPU in local mode via `FINDOC_DEVICE=0`

- NVIDIA GPU with CUDA 12 driver, one GPU per Spark task slot
- RAPIDS Accelerator jar matching your Spark version, plus `requirements-gpu.txt` (cuDF)
- Spark resource scheduling: `spark.executor.resource.gpu.amount`,
  `spark.task.resource.gpu.amount`, and a GPU discovery script
  (`--gpu` on `findoc.pipeline` sets these)
- Install `onnxruntime-gpu` instead of `onnxruntime`, or use the `.pt` checkpoint with a CUDA build of PyTorch

See `docs/AWS_DEPLOYMENT.md` for EMR.

## Fine-tuning the full-size model

The reported run fine-tunes `yolo26n` on CPU. Training `yolo26x` as in the paper
needs a GPU with roughly 16 GB or more of memory at `--batch 32 --imgsz 640`.

## Fonts (matters for reproducing the synthetic data)

The data generator draws text with the DejaVu fonts (`fonts-dejavu-core`). Without them the generator now stops with an error instead of silently using a different font, which an earlier Colab run showed changes every image and cuts end-to-end accuracy (see CHANGELOG).
