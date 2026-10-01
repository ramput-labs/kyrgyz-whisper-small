#!/usr/bin/env python
"""Concatenate audio files with a short pause between them into one 16 kHz wav."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kyrgyz_whisper_small.audio import load_audio, save_wav
from kyrgyz_whisper_small.config import SAMPLE_RATE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--gap", type=float, default=0.7, help="Silence between clips, seconds.")
    args = ap.parse_args()

    gap = np.zeros(int(args.gap * SAMPLE_RATE), dtype=np.float32)
    parts = []
    for p in sorted(args.inputs):
        parts += [load_audio(p), gap]
    audio = np.concatenate(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_wav(args.out, audio)
    print(f"{len(args.inputs)} clips -> {args.out} ({len(audio) / SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
