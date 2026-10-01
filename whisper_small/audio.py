from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .config import SAMPLE_RATE


def _load_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Cannot decode {path.suffix} with libsndfile and ffmpeg is not installed. "
            "Install it with `brew install ffmpeg` or convert the file to wav/flac/mp3/ogg."
        )
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return audio.astype(np.float32, copy=False)


def load_audio(path: str | Path) -> np.ndarray:
    """Load an audio file as 16 kHz mono float32 in [-1, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        return _load_with_ffmpeg(path, SAMPLE_RATE)
    return to_mono_16k(audio, sr)


def record(seconds: float, device: int | str | None = None) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Microphone support needs `pip install sounddevice`") from e
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0]


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), audio, sr)


def duration(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return len(audio) / sr


def _frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def split_on_silence(
    audio: np.ndarray,
    max_seconds: float = 28.0,
    min_seconds: float = 10.0,
    frame_ms: int = 20,
    smooth_ms: int = 300,
    sr: int = SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """Cut audio into <= max_seconds pieces at the quietest point of each window.

    Cutting at pauses avoids split words and stitching overlapping chunks.
    """
    total = len(audio)
    max_len, min_len = int(max_seconds * sr), int(min_seconds * sr)
    if total <= max_len:
        return [(0, total)]

    frame = sr * frame_ms // 1000
    db = _frame_db(audio, frame)
    k = max(1, smooth_ms // frame_ms)
    smooth = np.convolve(db, np.ones(k) / k, mode="same")

    spans, start = [], 0
    while total - start > max_len:
        lo, hi = (start + min_len) // frame, (start + max_len) // frame
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * frame
        spans.append((start, cut))
        start = cut
    spans.append((start, total))
    return spans


def is_silent(audio: np.ndarray, peak_db: float = -60.0) -> bool:
    # Not a loudness gate: real speech can peak around -40 dBFS.
    if len(audio) == 0:
        return True
    return 20 * np.log10(float(np.abs(audio).max()) + 1e-12) < peak_db
