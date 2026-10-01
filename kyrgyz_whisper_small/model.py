from __future__ import annotations

import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import torch
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

from .config import DEFAULT_MODEL_DIR, DRIVE_FILE_ID, MODEL_ID


def _fetch_resumable(
    url: str, dest: Path, on_progress: Callable[[int, int], None] | None = None, retries: int = 20
) -> None:
    # Google Drive drops long downloads; resume from the partial file with Range requests.
    for attempt in range(retries):
        done = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.headers.get_content_type() == "text/html":
                    raise RuntimeError("Google Drive returned a web page, not the file. Is the link public?")
                if done and resp.status != 206:
                    done = 0
                total = done + int(resp.headers.get("Content-Length", 0))
                with dest.open("ab" if done else "wb") as f:
                    while chunk := resp.read1(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if on_progress:
                            on_progress(done, total)
            if done >= total:
                return
        except urllib.error.HTTPError as e:
            if e.code == 416:
                return
            if attempt == retries - 1:
                raise
            time.sleep(min(2**attempt, 30))
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Download incomplete after {retries} attempts; run it again to resume.")


def download_from_drive(
    local_dir: Path = DEFAULT_MODEL_DIR,
    file_id: str = DRIVE_FILE_ID,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = local_dir.with_suffix(".zip.part")
    _fetch_resumable(url, archive, on_progress)
    with tempfile.TemporaryDirectory(dir=local_dir.parent) as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        src = next(Path(tmp).rglob("config.json")).parent
        shutil.rmtree(local_dir, ignore_errors=True)
        shutil.move(str(src), local_dir)
    archive.unlink()
    return local_dir


def download_from_hub(local_dir: Path = DEFAULT_MODEL_DIR, repo_id: str = MODEL_ID) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def resolve_model_path(model: str | Path | None) -> str:
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
        return torch.float16 if device in ("cuda", "mps") else torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


def load_model(
    model: str | Path | None = None, device: str = "auto", dtype: str = "auto"
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, str, torch.dtype]:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    path = resolve_model_path(model)
    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, dev)
    processor = WhisperProcessor.from_pretrained(path)
    net = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch_dtype)
    net.to(dev).eval()
    # The checkpoint ships use_cache=False; K/V caching makes decoding much faster.
    net.config.use_cache = True
    net.generation_config.use_cache = True
    net.generation_config.max_length = None
    return net, processor, dev, torch_dtype
