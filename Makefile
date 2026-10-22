.PHONY: help install data train protocol pattern artifacts paper e2e test clean

PY ?= python
DATA ?= work/data
RUNS ?= work/runs
EPOCHS ?= 15
MODEL ?= yolo26n.pt
DEVICE ?= cpu
RESULTS ?= results

help:            ## list targets
	@grep -E '^[a-z_]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

install:         ## Python deps (system deps: Java 17/21 and Tesseract, see docs/REQUIREMENTS.md)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

data:            ## synthetic financial-form dataset with exact ground truth
	$(PY) -m findoc.synth $(DATA)

train:           ## fine-tune YOLO26 and export ONNX (CPU: about 35 min for yolo26n x 15 epochs on 2 cores)
	$(PY) -m findoc.train $(DATA)/financial_docs.yaml --model $(MODEL) --epochs $(EPOCHS) --batch 8 --device $(DEVICE) --project $(RUNS) --name synth
	mkdir -p $(RESULTS) && cp $(RUNS)/synth/summary.json $(RESULTS)/train_summary.json

protocol:        ## per variant: pick confidence on val, score on test once, plus full-page Tesseract baseline
	$(PY) -m findoc.protocol $(DATA) $(RUNS)/synth/weights/best.onnx $(RESULTS)

pattern:         ## row-wise UDF vs per-partition inference, real model (about 10 min on 2 cores)
	$(PY) benchmarks/inference_pattern_benchmark.py $(DATA)/images/test $(RUNS)/synth/weights/best.onnx --docs 48 --partitions 2 --repeats 3 --out $(RESULTS)/pattern.json

artifacts:       ## regenerate paper macros and charts from results/*.json
	$(PY) scripts/make_paper_artifacts.py

paper:           ## build paper/paper.pdf
	cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex

e2e: data train protocol pattern artifacts   ## everything, in order

test:
	$(PY) -m pytest -q

clean:
	rm -rf work build *.egg-info .pytest_cache
