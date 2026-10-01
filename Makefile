SHELL   := /bin/bash
PYTHON  ?= $(shell command -v python3.12 || command -v python3)
VENV    := .venv
BIN     := $(VENV)/bin
PY      := $(BIN)/python
CLI     := $(PY) -m kyrgyz_whisper_small

FILE    ?= samples/long.wav
N       ?= 20
SPLIT   ?= dev
DEVICE  ?= auto
DTYPE   ?= auto
LANG_ID ?= kk
BEAMS   ?= 1
SECONDS ?= 5
SOURCE  ?= hub
REPO    ?= ramput-labs/kyrgyz-whisper-small
ARGS    ?=
FLEURS  := data/fleurs_ky

COMMON  = --device $(DEVICE) --dtype $(DTYPE)
DECODE  = -l $(LANG_ID) -b $(BEAMS) $(COMMON)
TRANSCRIBE = $(CLI) transcribe "$(FILE)" $(DECODE)

.DEFAULT_GOAL := help
.PHONY: help setup install download samples long-sample quickstart info transcribe srt json \
        detect-lang mic eval bench demo upload test lint clean clean-all

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<target>\033[0m [VAR=value]\n\n"} \
	     /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo -e "\nVars: FILE=$(FILE) DEVICE=$(DEVICE) DTYPE=$(DTYPE) LANG_ID=$(LANG_ID) BEAMS=$(BEAMS) N=$(N) SPLIT=$(SPLIT) SOURCE=$(SOURCE)\n"

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

setup: $(PY) ## Create venv and install requirements.txt
	$(BIN)/pip install -q -r requirements.txt
	@echo "✓ ready — try: make quickstart"

install: setup ## Alias for setup

download: ## Download model weights (~1 GB) into ./models (SOURCE=hub|drive)
	$(CLI) download --source $(SOURCE)

samples: ## Fetch N Kyrgyz clips from Google FLEURS (streams, only a few MB)
	$(PY) -m scripts.fetch_fleurs --split $(SPLIT) -n $(N) --out $(FLEURS)

long-sample: ## Glue the first 10 FLEURS clips into a ~2 min file (samples/long.wav)
	$(PY) -m scripts.concat_audio $$(ls $(FLEURS)/wavs/*.wav | head -10) -o samples/long.wav

quickstart: setup download samples long-sample transcribe ## Everything: setup → model → data → first transcript

info: ## Model architecture / config summary
	$(CLI) info

transcribe: ## Transcribe FILE (with timestamps)
	$(TRANSCRIBE) -t

srt: ## Write subtitles for FILE into out/
	$(TRANSCRIBE) -f srt -o out

json: ## Write JSON (text, segments, timing) for FILE into out/
	$(TRANSCRIBE) -f json -o out

detect-lang: ## Which Whisper language token does the model "hear" in FILE?
	$(CLI) detect-lang "$(FILE)" $(COMMON)

mic: ## Record SECONDS from the microphone and transcribe (loops until Ctrl+C)
	$(CLI) mic -s $(SECONDS) --loop $(DECODE)

eval: ## WER/CER on the FLEURS manifest (make samples N=100 first for a real number)
	$(CLI) evaluate $(FLEURS)/manifest.tsv $(DECODE) --report out/eval.jsonl

bench: ## Speed comparison cpu/mps × fp32/fp16 on FILE
	$(CLI) bench "$(FILE)"

demo: ## Gradio web UI, opened in the browser (upload or record; ARGS=--share for a public link)
	@$(PY) -c "import gradio" 2>/dev/null || $(BIN)/pip install -q gradio
	$(PY) -m scripts.gradio_demo --open --device $(DEVICE) $(ARGS)

upload: ## Upload ./models/kyrgyz-whisper-small + model card to REPO (needs `hf auth login`)
	$(PY) -m scripts.upload_model --repo $(REPO)

test: ## Run unit tests (fast; model tests auto-skip if weights missing)
	$(BIN)/pytest -q

clean: ## Remove outputs and caches (keeps model + data)
	rm -rf out .pytest_cache
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -exec rm -rf {} +

clean-all: clean ## Also remove venv, model weights and downloaded data
	rm -rf $(VENV) models data samples
