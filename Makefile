SHELL   := /bin/bash
PYTHON  ?= $(shell command -v python3.12 || command -v python3)
VENV    := .venv
BIN     := $(VENV)/bin
CLI     := $(BIN)/python -m whisper_small

FILE    ?= samples/long.wav
N       ?= 20
SPLIT   ?= dev
DEVICE  ?= auto
DTYPE   ?= auto
LANG_ID ?= kk
BEAMS   ?= 1
SECONDS ?= 5
FLEURS  := data/fleurs_ky

COMMON  = --device $(DEVICE) --dtype $(DTYPE)

.DEFAULT_GOAL := help
.PHONY: help setup install download samples long-sample quickstart info transcribe srt json \
        detect-lang mic eval bench demo test lint clean clean-all

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<target>\033[0m [VAR=value]\n\n"} \
	     /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo -e "\nVars: FILE=$(FILE) DEVICE=$(DEVICE) DTYPE=$(DTYPE) LANG_ID=$(LANG_ID) BEAMS=$(BEAMS) N=$(N) SPLIT=$(SPLIT)\n"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

setup: $(BIN)/python ## Create venv and install requirements.txt
	$(BIN)/pip install -q -r requirements.txt
	@echo "✓ ready — try: make quickstart"

install: setup ## Alias for setup

download: ## Download model weights (~1 GB) into ./models
	$(CLI) download

samples: ## Fetch N Kyrgyz clips from Google FLEURS (streams, only a few MB)
	$(BIN)/python -m scripts.fetch_fleurs --split $(SPLIT) -n $(N) --out $(FLEURS)

long-sample: ## Glue the first 10 FLEURS clips into a ~2 min file (samples/long.wav)
	$(BIN)/python -m scripts.concat_audio $$(ls $(FLEURS)/wavs/*.wav | head -10) -o samples/long.wav

quickstart: setup download samples long-sample ## Everything: setup → model → data → first transcript
	$(CLI) transcribe samples/long.wav -t $(COMMON)

info: ## Model architecture / config summary
	$(CLI) info

transcribe: ## Transcribe FILE (with timestamps)
	$(CLI) transcribe "$(FILE)" -t -l $(LANG_ID) -b $(BEAMS) $(COMMON)

srt: ## Write subtitles for FILE into out/
	$(CLI) transcribe "$(FILE)" -f srt -o out -l $(LANG_ID) -b $(BEAMS) $(COMMON)

json: ## Write JSON (text, segments, timing) for FILE into out/
	$(CLI) transcribe "$(FILE)" -f json -o out -l $(LANG_ID) -b $(BEAMS) $(COMMON)

detect-lang: ## Which Whisper language token does the model "hear" in FILE?
	$(CLI) detect-lang "$(FILE)" $(COMMON)

mic: ## Record SECONDS from the microphone and transcribe (loops until Ctrl+C)
	$(CLI) mic -s $(SECONDS) --loop -l $(LANG_ID) $(COMMON)

eval: ## WER/CER on the FLEURS manifest (make samples N=100 first for a real number)
	$(CLI) evaluate $(FLEURS)/manifest.tsv -l $(LANG_ID) -b $(BEAMS) $(COMMON) --report out/eval.jsonl

bench: ## Speed comparison cpu/mps × fp32/fp16 on FILE
	$(CLI) bench "$(FILE)"

demo: ## Gradio web UI (upload or record in the browser)
	$(BIN)/pip install -q gradio
	$(BIN)/python -m scripts.gradio_demo

test: ## Run unit tests (fast; model tests auto-skip if weights missing)
	$(BIN)/pytest -q

clean: ## Remove outputs and caches (keeps model + data)
	rm -rf out .pytest_cache **/__pycache__

clean-all: clean ## Also remove venv, model weights and downloaded data
	rm -rf $(VENV) models data samples
