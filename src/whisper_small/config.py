from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "the-cramer-project/AkylAI-STT-small"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
# Whisper has no <|ky|> token; the model is trained under the Kazakh one.
DEFAULT_LANGUAGE = "kk"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("WHISPER_SMALL_MODELS_DIR", PROJECT_ROOT / "models"))
DEFAULT_MODEL_DIR = MODELS_DIR / "whisper-small-ky"
DATA_DIR = PROJECT_ROOT / "data"
