# Provenance

Hashes of the result files that `scripts/make_paper_artifacts.py` turns into the paper's macros.
Written at code commit `67e1a42` (the commit that contains this file is its child).
CPU results were produced before the final code changes and were then re-checked: rerunning the protocol with the
released code on the stored CPU model reproduced all eight evaluation files exactly (timings excluded).

## results/

| file | sha256 (first 16) |
|---|---|
| environment.json | `7551d5bf7b449786` |
| eval_report_class_nms.json | `1735de82f22060c3` |
| eval_report_default.json | `bfe29b006780ae9b` |
| eval_report_nms_free.json | `e63c35506919b9c9` |
| ocr_crop_ablation.json | `725f6d11a45a4a27` |
| pattern.json | `86eebb7ac2a332dc` |
| rules_baseline.json | `f1a749573dd6aaa7` |
| run_report_class_nms.json | `4cb98c28956fdb0e` |
| run_report_default.json | `644693a84cbc5ffb` |
| run_report_nms_free.json | `84498a73003c6b1e` |
| train_summary.json | `5651c757ac42e62e` |
| val_sweep_class_nms.json | `0e10a25e21a1da4b` |
| val_sweep_nms_free.json | `56b23bdc765ec3e0` |

## results_gpu/

| file | sha256 (first 16) |
|---|---|
| environment.json | `8244df8873a7e7ce` |
| eval_report_class_nms.json | `e9f9ce61ae166a6d` |
| eval_report_default.json | `2b328626ccc69cd6` |
| eval_report_nms_free.json | `7ef9e7f03b333c57` |
| pattern.json | `bc03539cea3c1068` |
| run_report_class_nms.json | `618a85de3786a3c5` |
| run_report_default.json | `13169047c39cac59` |
| run_report_nms_free.json | `0d94c91939f67974` |
| train_summary.json | `60717f2e0f6d2155` |
| val_sweep_class_nms.json | `6f60d175e3aaa18d` |
| val_sweep_nms_free.json | `4c5790ee8fd70bd2` |

## results_gpu_font_fallback/

| file | sha256 (first 16) |
|---|---|
| environment.json | `9449dff6d022ca9b` |
| eval_report_class_nms.json | `521c3e82e6ea7a16` |
