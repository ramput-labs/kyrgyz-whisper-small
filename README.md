# kyrgyz-whisper-small

Kyrgyz speech recognition (кыргызча кепти текстке айландыруу) with Whisper-small:
CLI, Python API, microphone input, web demo and WER/CER evaluation.

Model: [ramput-labs/kyrgyz-whisper-small](https://huggingface.co/ramput-labs/kyrgyz-whisper-small) on Hugging Face.

## Quickstart

Requires Python 3.10+ and ~1 GB disk. `ffmpeg` is optional (for m4a/mp4/webm).

```bash
git clone https://github.com/ramput-labs/kyrgyz-whisper-small.git
cd kyrgyz-whisper-small
make quickstart   # venv → install → download model → sample audio → first transcript
make help         # all commands
```

Without Make:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m kyrgyz_whisper_small download
python -m kyrgyz_whisper_small transcribe audio.wav -t
```

## Usage

| Task | Command |
|---|---|
| Transcribe | `make transcribe FILE=audio.wav` |
| Subtitles / JSON | `make srt FILE=audio.wav` · `make json FILE=audio.wav` |
| Microphone | `make mic` |
| Web demo | `make demo` |
| WER/CER on FLEURS | `make samples N=100 && make eval` |
| Speed benchmark | `make bench FILE=audio.wav` |

Options: `DEVICE=auto|cpu|mps|cuda`, `DTYPE=auto|fp32|fp16`, `BEAMS=1`.

```python
from kyrgyz_whisper_small import Transcriber

asr = Transcriber()                # cuda / mps / cpu, picked automatically
res = asr.transcribe("audio.wav")  # any length
print(res.text)
```

## Results

FLEURS Kyrgyz dev, 100 clips without digits: **WER 16.3% / CER 4.5%**.

Whisper has no Kyrgyz language token, so the model runs with `kk` (the default). Numbers come out
as words ("жети миң"), which counts as an error against references that use digits.

## Credits

- Model weights: [the-cramer-project/AkylAI-STT-small](https://huggingface.co/the-cramer-project/AkylAI-STT-small)
  by The Cramer Project, a Kyrgyz fine-tune of [openai/whisper-small](https://huggingface.co/openai/whisper-small)
  by OpenAI. We re-host the weights without retraining them.
- Evaluation audio: [google/fleurs](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).

## License

The code is [MIT](LICENSE). The model weights are Apache-2.0, inherited from AkylAI-STT-small and Whisper.
