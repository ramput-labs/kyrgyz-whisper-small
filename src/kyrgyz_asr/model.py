"""Model download, device selection and loading."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

from .config import DEFAULT_MODEL_DIR, MODEL_ID


def download_model(local_dir: Path = DEFAULT_MODEL_DIR, repo_id: str = MODEL_ID) -> Path:
    """Download the model snapshot into ``local_dir`` (idempotent, resumable)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def resolve_model_path(model: str | Path | None) -> str:
    """Prefer the local snapshot; fall back to the Hub id (downloads to HF cache)."""
    if model is None:
        return str(DEFAULT_MODEL_DIR) if (DEFAULT_MODEL_DIR / "config.json").exists() else MODEL_ID
    return str(model)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        # fp16 is a clear win on CUDA/MPS; CPU kernels are fastest in fp32.
        return torch.float16 if device in ("cuda", "mps") else torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


def load_model(
    model: str | Path | None = None, device: str = "auto", dtype: str = "auto"
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, str, torch.dtype]:
    # Keep the console readable: transformers v5 is chatty about Whisper generation kwargs.
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    path = resolve_model_path(model)
    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, dev)
    processor = WhisperProcessor.from_pretrained(path)
    net = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch_dtype)
    net.to(dev).eval()
    # The checkpoint ships use_cache=False (training leftover); caching K/V makes decoding much faster.
    net.config.use_cache = True
    net.generation_config.use_cache = True
    net.generation_config.max_length = None  # we always pass max_new_tokens
    return net, processor, dev, torch_dtype
