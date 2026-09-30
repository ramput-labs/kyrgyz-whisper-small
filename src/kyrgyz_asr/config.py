"""Project-wide constants and paths."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "the-cramer-project/AkylAI-STT-small"  # Hub repo the weights are downloaded from (Apache-2.0)
SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono
CHUNK_SECONDS = 30  # Whisper's native receptive window
# Whisper has no <|ky|> token. The fine-tune was trained under the Kazakh token: auto-detect
# picks "kk" and forcing it measured best on FLEURS ky (vs. auto / "ru"). "auto" = let the model pick.
DEFAULT_LANGUAGE = "kk"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("KYASR_MODELS_DIR", PROJECT_ROOT / "models"))
DEFAULT_MODEL_DIR = MODELS_DIR / "whisper-small-ky"
DATA_DIR = PROJECT_ROOT / "data"
