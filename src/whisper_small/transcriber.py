from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .audio import duration, is_silent, load_audio, split_on_silence
from .config import CHUNK_SECONDS, DEFAULT_LANGUAGE, SAMPLE_RATE
from .model import load_model


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    text: str
    audio_seconds: float
    elapsed_seconds: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.elapsed_seconds / max(self.audio_seconds, 1e-9)


class Transcriber:
    def __init__(
        self,
        model: str | Path | None = None,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 8,
    ):
        self.model, self.processor, self.device, self.dtype = load_model(model, device, dtype)
        self.batch_size = batch_size

    @torch.inference_mode()
    def _generate(self, clips: list[np.ndarray], gen_kwargs: dict) -> list[str]:
        texts: list[str] = []
        for i in range(0, len(clips), self.batch_size):
            batch = clips[i : i + self.batch_size]
            feats = self.processor.feature_extractor(
                batch, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(self.device, self.dtype)
            ids = self.model.generate(feats, **gen_kwargs)
            texts += self.processor.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        *,
        language: str | None = DEFAULT_LANGUAGE,
        num_beams: int = 1,
        max_new_tokens: int = 440,
    ) -> Result:
        """Transcribe a file path or 16 kHz mono float32 array of any length."""
        if not isinstance(audio, np.ndarray):
            audio = load_audio(audio)
        gen_kwargs: dict = {"task": "transcribe", "num_beams": num_beams, "max_new_tokens": max_new_tokens}
        if language and language != "auto":
            gen_kwargs["language"] = language

        t0 = time.perf_counter()
        spans = [(s, e) for s, e in split_on_silence(audio, max_seconds=CHUNK_SECONDS - 2) if not is_silent(audio[s:e])]
        texts = self._generate([audio[s:e] for s, e in spans], gen_kwargs) if spans else []
        elapsed = time.perf_counter() - t0

        segments = [
            Segment(start=round(s / SAMPLE_RATE, 2), end=round(e / SAMPLE_RATE, 2), text=t)
            for (s, e), t in zip(spans, texts)
            if t
        ]
        return Result(
            text=" ".join(seg.text for seg in segments),
            audio_seconds=duration(audio),
            elapsed_seconds=elapsed,
            segments=segments,
        )

    @torch.inference_mode()
    def detect_language(self, audio: str | Path | np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if not isinstance(audio, np.ndarray):
            audio = load_audio(audio)
        feats = self.processor(
            audio[: CHUNK_SECONDS * SAMPLE_RATE], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device, self.dtype)
        sot = self.model.generation_config.decoder_start_token_id
        decoder_ids = torch.tensor([[sot]], device=self.device)
        logits = self.model(input_features=feats, decoder_input_ids=decoder_ids).logits[0, -1].float()

        lang_to_id: dict[str, int] = self.model.generation_config.lang_to_id
        codes = [tok.strip("<|>") for tok in lang_to_id]
        ids = torch.tensor(list(lang_to_id.values()), device=logits.device)
        probs = logits[ids].softmax(-1)
        top = probs.topk(min(top_k, len(codes)))
        return [(codes[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
