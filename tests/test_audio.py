import numpy as np
import pytest

from whisper_small.audio import is_silent, load_audio, save_wav, split_on_silence, to_mono_16k
from whisper_small.config import SAMPLE_RATE
from whisper_small.evaluate import normalize, score

SR = SAMPLE_RATE


def tone(seconds: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_resample_and_downmix():
    t = np.arange(48_000) / 48_000
    mono = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = to_mono_16k(np.stack([mono, mono], axis=1), 48_000)
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) - SR) <= 1


def test_load_roundtrip(tmp_path):
    p = tmp_path / "x.wav"
    save_wav(p, tone(0.5))
    assert len(load_audio(p)) == SR // 2


def test_short_audio_is_one_span():
    assert split_on_silence(tone(10)) == [(0, 10 * SR)]


def test_long_audio_is_cut_at_the_pause():
    gap = np.zeros(SR, dtype=np.float32)
    audio = np.concatenate([tone(20), gap, tone(20)])
    spans = split_on_silence(audio, max_seconds=28)
    assert len(spans) == 2
    cut = spans[0][1] / SR
    assert 20.0 <= cut <= 21.0
    assert spans[-1][1] == len(audio)
    assert all((e - s) / SR <= 28 for s, e in spans)


def test_silence_detection_keeps_quiet_speech():
    assert is_silent(np.zeros(SR, dtype=np.float32))
    assert not is_silent(tone(1.0, amp=0.008))


def test_normalize_kyrgyz():
    assert normalize("Салам, Дүйнө!  Ёлка") == "салам дүйнө елка"


def test_score():
    m = score(["бир эки үч"], ["бир эки төрт"])
    assert m["wer"] == pytest.approx(1 / 3)
