# Datasets

## Why the evaluation uses synthetic data

The target classes are field regions on financial forms: account number, date, amount, name, signature.

| Dataset | Annotation | Why it does not fit |
|---|---|---|
| FUNSD (199 scanned forms) | Entities labelled question / answer / header / other | No field-class labels such as account number or signature |
| SROIE 2019 (1,000 receipts) | Text-line boxes and key-information strings | No detection-style field-region boxes; receipts, not forms |

`findoc.synth` generates account forms at 960x640 with exact boxes and exact field strings, so detection and OCR can both be scored with no annotation noise. It varies layout, field order, font, noise and blur.

## What this does and does not show

Results on this data show that the pipeline is correct and measurable end to end. They are **not** evidence about accuracy on real scanned documents, which are harder: skew, stamps, handwriting, compression artifacts, many templates.

Replacing the generator with real labelled documents needs only a YOLO-format dataset YAML and a ground-truth JSON in the format written by `findoc.synth.build_dataset`.
