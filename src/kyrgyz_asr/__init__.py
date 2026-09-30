"""kyrgyz-asr: Kyrgyz speech recognition."""

from .config import MODEL_ID
from .transcriber import Result, Segment, Transcriber

__all__ = ["MODEL_ID", "Result", "Segment", "Transcriber"]
__version__ = "0.1.0"
