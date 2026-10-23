# Changelog

## Unreleased

- Added a Colab GPU notebook (`notebooks/colab_gpu_run.ipynb`, `docs/COLAB.md`). It writes to `results_gpu/` and does not touch the CPU results used in the paper.
- Added `FINDOC_DEVICE` (default `cpu`) to choose the inference device, and a `RESULTS` variable in the Makefile. CPU behavior is unchanged.
- First Colab GPU run: detection replicated (class-NMS precision and recall 1.00), but end-to-end text accuracy was 0.717 against 0.971 on CPU. The recorded environment showed the cause: Colab had no DejaVu fonts, so the generator silently fell back to PIL's default font and produced different test images (hash 840f3d... against 225dd7... on the CPU machine). Tesseract was identical (5.3.4) and the GPU was really used (CUDA execution provider confirmed). The generator now raises an error when the fonts are missing, and the notebook installs `fonts-dejavu-core`. The paper reports that run only as a labeled note.
- Third Colab run (T4, fonts installed): test images byte-identical to the CPU machine (same SHA-256), ONNX Runtime used the CUDA provider. Class-NMS end-to-end 98.3% against 97.1% on CPU, same ordering of variants, same per-partition speedup (2.1x), no inference speedup from the GPU for the nano model. Added as a replication subsection with numbers generated from `results_gpu/`.

## 0.1.0

First runnable release. Everything below was found by running the code, not by reading it.

### Defects fixed relative to the pipeline sketched in the original paper draft

| # | Defect | What actually happened | Fix |
|---|---|---|---|
| 1 | Spark `image` data source used as input to an image decoder | `image.data` holds decoded raw pixels (1,843,200 bytes for a 640x960x3 image vs a 132,098 byte JPEG). `cv2.imdecode` on it returns `None`. | Read with `binaryFile`, whose `content` is byte-identical to the file. |
| 2 | `cv2.imread(path)` on an `s3://` URI | Returns `None`. | Decode the bytes Spark already read. Never re-open the path. |
| 3 | `mapInPandas` on PySpark 3.5.3 with JDK 21 | Fails in the JVM: `sun.misc.Unsafe or java.nio.DirectByteBuffer.<init>(long, int) not available`. Persists with `--add-opens` and `-Dio.netty.tryReflectionSetAccessible=true`. | Use PySpark 4.0.1 on JDK 21. (PySpark 3.5 on JDK 17 is the documented alternative; not tested here.) |
| 4 | Model built inside a row-wise UDF | Loads once per row. Cost is measured in `results/pattern.json`. | Load once per partition in `mapInPandas`. |
| 5 | One image per model call | No batching. | One call per Arrow batch; ONNX exported with a dynamic batch axis (`findoc/export.py`). |
| 6 | Manual `cv2.resize` to 640x640 | Distorts non-square pages and needs a box remap. | Pass the original image; Ultralytics letterboxes and returns boxes in original pixels. |
| 7 | `device=0` hard-coded, `concurrentGpuTasks=2` | Can place two model copies on one GPU. **Untested: no GPU available.** | Device read from `TaskContext` resources; one GPU per task. |
| 8 | cuDF used to count a substring in a JSON string | Too small to benefit; searched for a key not present in the output schema. | Per-partition partial (sum, count) by class, reduced by Spark. |
| 9 | `--project work/runs` passed to Ultralytics | Relative project paths are nested under `runs/detect/`, so `best.onnx` was not where the Makefile looked. | `findoc.train` resolves the project path to an absolute path. |

### Mistakes made while building this, kept here on purpose

- An earlier revision "fixed" defect 1 by decoding `image.data` with PIL. That is wrong for the reason in row 1 and would have failed on the first image. It was replaced, not patched.
- An earlier revision of the benchmark used `time.sleep(0.15)` as a stand-in for model loading. It was removed. `benchmarks/inference_pattern_benchmark.py` uses the real fine-tuned model.
- Training with AdamW at `lr0=0.01` (the original draft's value) showed unstable validation mAP50 in the first epochs (0.425, 0.051, 0.000, 0.808; see `results/early_epochs_lr0.01.csv`). That run was stopped after four epochs, so nothing is claimed about where it would have ended. The default is now `lr0=0.001`.
