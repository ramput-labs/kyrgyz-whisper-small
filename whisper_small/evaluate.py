from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import jiwer

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


@dataclass
class Utterance:
    path: Path
    reference: str
    hypothesis: str = ""


def read_manifest(path: str | Path) -> list[Utterance]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if rows and rows[0][:2] == ["path", "text"]:
        rows = rows[1:]
    return [Utterance(path=(path.parent / r[0]).resolve(), reference=r[1]) for r in rows if r]


def has_digits(text: str) -> bool:
    # The model spells numbers out while FLEURS uses digits, which inflates WER.
    return any(ch.isdigit() for ch in text)


def score(refs: list[str], hyps: list[str]) -> dict[str, float]:
    refs_n = [normalize(r) for r in refs]
    hyps_n = [normalize(h) for h in hyps]
    return {"wer": jiwer.wer(refs_n, hyps_n), "cer": jiwer.cer(refs_n, hyps_n)}
