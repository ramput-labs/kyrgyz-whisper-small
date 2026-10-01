from pathlib import Path

import pytest

from whisper_small.config import DEFAULT_MODEL_DIR, PROJECT_ROOT
from whisper_small.evaluate import read_manifest, score

MANIFEST = PROJECT_ROOT / "data/fleurs_ky/manifest.tsv"

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.safetensors").exists() or not MANIFEST.exists(),
    reason="model or FLEURS samples not downloaded",
)


@pytest.fixture(scope="module")
def asr():
    from whisper_small import Transcriber

    return Transcriber()


def test_transcribes_kyrgyz(asr):
    utts = read_manifest(MANIFEST)[:3]
    hyps = [asr.transcribe(u.path).text for u in utts]
    assert score([u.reference for u in utts], hyps)["cer"] < 0.15


def test_language_token_is_kazakh_slot(asr):
    top_lang, _ = asr.detect_language(Path(read_manifest(MANIFEST)[0].path))[0]
    assert top_lang == "kk"
