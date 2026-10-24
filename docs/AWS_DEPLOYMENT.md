# AWS EMR deployment notes

**Status: design only. Nothing in this file was run.** The measured results in the paper come from a two-core CPU machine.

## Intended shape

1. Upload to S3 raises an event.
2. Step Functions starts or resizes an EMR cluster.
3. `spark-submit` runs `findoc.pipeline` with `--gpu` on GPU nodes (for example g4dn) with the RAPIDS Accelerator.
4. Parquet output lands in S3; Glue loads it into Redshift.

## What to check first on a real cluster

- The RAPIDS Accelerator jar matches the Spark version, and `requirements-gpu.txt` (cuDF) is installed on workers.
- A GPU discovery script exists at the path in `GPU_DISCOVERY_SCRIPT`.
- `spark.task.resource.gpu.amount` and `spark.executor.resource.gpu.amount` give one GPU per task slot, so two tasks never share a device.
- Use `onnxruntime-gpu` instead of `onnxruntime`, or the `.pt` checkpoint with a CUDA build of PyTorch.
- Cold cluster start-up is a fixed cost not captured by any throughput figure here. For interactive latency use a warm cluster or EMR Serverless instead.

## Not measured

GPU throughput, multi-node scaling, cluster cold start, cost per thousand documents, and Amazon Textract comparison.


## Architecture sketch

The diagram source is `paper/figures/architecture.dot` (rendered in `architecture.png`). Solid boxes are the Spark job that is measured in the paper. Dashed boxes are this design, which has not been run.
