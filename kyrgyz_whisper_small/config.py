from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "ramput-labs/kyrgyz-whisper-small"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
# Whisper has no <|ky|> token; forcing Kazakh ("kk") gives the best Kyrgyz accuracy.
DEFAULT_LANGUAGE = "kk"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("KYRGYZ_WHISPER_SMALL_MODELS_DIR", PROJECT_ROOT / "models"))
DEFAULT_MODEL_DIR = MODELS_DIR / "kyrgyz-whisper-small"
