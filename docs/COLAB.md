# Running on a GPU with Google Colab

The paper's numbers come from a 2-core CPU machine. This notebook repeats the full study on a GPU and writes to `results_gpu/`, so nothing from the CPU run is overwritten.

1. Open `notebooks/colab_gpu_run.ipynb` in Colab (File > Open notebook > GitHub, or upload it).
2. Runtime > Change runtime type > T4 GPU.
3. Runtime > Run all. About 30 to 60 minutes.
4. Download `results_gpu.zip` from the last cell.

## What it measures

Same dataset, same training settings, same protocol, same Spark benchmark as the CPU run. Only the device changes (`FINDOC_DEVICE=0`).

Spark still runs in `local[2]` mode with inference on the single Colab GPU. This is **not** the multi-GPU, RAPIDS-scheduled cluster described in `docs/AWS_DEPLOYMENT.md`, which remains design only.

## Knobs

| Setting | How |
| --- | --- |
| Different repo or mirror | edit `REPO` in cell 3 |
| Device | `FINDOC_DEVICE` (`0` for GPU 0, default `cpu`) |
| Results folder | `make RESULTS=...` |

## Reporting GPU results

Report them as a separate row or table, labeled with the GPU model printed in cell 1 and the number of cores Colab gave you. Do not mix them with the CPU table.
