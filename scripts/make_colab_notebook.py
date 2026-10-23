"""Generate notebooks/colab_gpu_run.ipynb. Kept as code so the notebook can be rebuilt and diffed.

    python scripts/make_colab_notebook.py
"""
from pathlib import Path

import nbformat as nbf

M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    M("""# findoc-yolo26-spark: GPU run on Colab

Reruns the whole study on a GPU and writes everything to `results_gpu/`, so the CPU results used in the paper stay untouched.

**Before you start:** Runtime > Change runtime type > **T4 GPU** (the free tier is enough). Then Runtime > Run all. Expect roughly 30 to 60 minutes.

Steps: check the GPU, install system packages, clone the repo, install Python deps, generate data, train, run the evaluation protocol, run the Spark inference-pattern benchmark, bundle the results."""),
    C("""# 1. GPU visible?
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then rerun."
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))"""),
    C("""# 2. System packages: Java for Spark, Tesseract for OCR, DejaVu fonts (the data generator needs them)
!apt-get -qq update && apt-get -qq install -y openjdk-17-jdk-headless tesseract-ocr fonts-dejavu-core > /dev/null
!java -version 2>&1 | head -1
!tesseract --version 2>&1 | head -1"""),
    C("""# 3. Get the code. Change REPO if you use a fork or an anonymized mirror.
REPO = "ANONYMOUS-REPO-URL"
import os
if not os.path.isdir("/content/findoc"):
    !git clone --depth 1 {REPO} /content/findoc
%cd /content/findoc
!git log --oneline | head -1"""),
    C("""# 4. Python deps. Swap onnxruntime for the GPU build so ONNX inference can use CUDA.
!pip -q install -r requirements-dev.txt
!pip -q install -e . --no-deps
!pip -q uninstall -y onnxruntime
!pip -q install onnxruntime-gpu
import onnxruntime as ort
print("ONNX Runtime providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "No CUDA provider. Runtime > Restart session, then rerun from cell 1."
"""),
    C("""# 5. Settings. FINDOC_DEVICE=0 sends inference to GPU 0. Results go to results_gpu/.
import os
os.environ["FINDOC_DEVICE"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
MAKE = "make DEVICE=0 RESULTS=results_gpu"
!{MAKE} data"""),
    C("""# 6. Fine-tune YOLO26n for 15 epochs (same settings as the CPU run) and export ONNX
!{MAKE} train"""),
    C("""# 6b. Record the environment (versions, fonts, GPU, ONNX provider actually used, hash of the test images)
!python scripts/record_environment.py work/data work/runs/synth/weights/best.onnx results_gpu/environment.json"""),
    C("""# 7. Evaluation protocol: confidence chosen on validation, test scored once
!{MAKE} protocol"""),
    C("""# 8. Spark inference-pattern benchmark (row-wise UDF vs per-partition), now on GPU
!{MAKE} pattern"""),
    C("""# 9. Summary
import json, pathlib
r = pathlib.Path("results_gpu")
train = json.loads((r / "train_summary.json").read_text())
print("train seconds:", train["train_seconds"], "| val mAP50:", train["val_map50"])
for v in ["default", "nms_free", "class_nms"]:
    e = json.loads((r / f"eval_report_{v}.json").read_text())
    print(f"{v:10s} P={e['overall_precision']:.3f} R={e['overall_recall']:.3f} end-to-end={e['pipeline_end_to_end_field_accuracy']:.3f}")
p = json.loads((r / "pattern.json").read_text())
print("pattern device:", p.get("device"))
for k, m in p["modes"].items():
    print(f"{k:14s} {m['seconds_mean']:6.1f}s +/- {m['seconds_sd']:.1f}  speedup vs row UDF: {m['speedup_vs_row_udf']}x")"""),
    C("""# 10. Bundle results and download. Send results_gpu.zip back so the paper tables can be regenerated.
!zip -qr results_gpu.zip results_gpu
from google.colab import files
files.download("results_gpu.zip")"""),
    M("""Optional: save a copy to Google Drive instead of downloading.

```python
from google.colab import drive
drive.mount('/content/drive')
!cp results_gpu.zip /content/drive/MyDrive/
```"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
}
nbf.validate(nb)
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_gpu_run.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
