# whisper-small

Kyrgyz speech recognition (кыргызча кепти текстке айландыруу) on Whisper-small.
Comes with a CLI, Python API, microphone input, web demo and WER/CER evaluation.

**[English](#english) · [Кыргызча](#кыргызча)**

---

## English

### Requirements

- Python 3.10+
- ~1 GB disk for model weights
- Optional: `ffmpeg` for m4a/mp4/webm (`brew install ffmpeg`)

### Setup with Make

```bash
make quickstart     # venv → install → model → sample audio → first transcript
make help           # list all commands
```

### Setup without Make

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m whisper_small download                          # model weights (~1 GB)
python -m scripts.fetch_fleurs -n 20 --out data/fleurs_ky # sample Kyrgyz clips
```

Run all commands from the project root.

### Examples

| Task | Make | CLI |
|---|---|---|
| Transcribe a file | `make transcribe FILE=audio.wav` | `python -m whisper_small transcribe audio.wav -t` |
| Subtitles (.srt) | `make srt FILE=audio.wav` | `python -m whisper_small transcribe audio.wav -f srt -o out` |
| JSON output | `make json FILE=audio.wav` | `python -m whisper_small transcribe audio.wav -f json -o out` |
| Microphone | `make mic` | `python -m whisper_small mic --loop` |
| Web demo | `make demo` | `pip install gradio && python -m scripts.gradio_demo` |
| Accuracy (WER/CER) | `make eval` | `python -m whisper_small evaluate data/fleurs_ky/manifest.tsv` |
| Speed benchmark | `make bench FILE=audio.wav` | `python -m whisper_small bench audio.wav` |
| Tests | `make test` | `pytest -q` |

Make variables: `FILE`, `DEVICE` (auto/cpu/mps/cuda), `DTYPE` (auto/fp32/fp16), `BEAMS`, `LANG_ID`, `N`, `SECONDS`.

```bash
make transcribe FILE=my.wav DEVICE=cpu BEAMS=5
```

### Python

```python
from whisper_small import Transcriber

asr = Transcriber()                  # picks cuda/mps/cpu automatically
res = asr.transcribe("audio.wav")    # any length
print(res.text)
for s in res.segments:
    print(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
```

### Results

FLEURS Kyrgyz, 100 dev clips: **WER 16.8% / CER 4.8%** (clips without digits).
Apple M5 Pro, fp16: ~140× faster than real time.

> Whisper has no Kyrgyz token, so the model runs under `kk` (default). Numbers are written
> as words ("жети миң"), which counts as an error against digit references.

---

## Кыргызча

### Талаптар

- Python 3.10+
- Модель үчүн ~1 GB орун
- Кошумча: m4a/mp4/webm үчүн `ffmpeg` (`brew install ffmpeg`)

### Make менен орнотуу

```bash
make quickstart     # venv → орнотуу → модель → үлгү аудио → биринчи транскрипция
make help           # бардык буйруктар
```

### Make'сиз орнотуу

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m whisper_small download                          # модель (~1 GB)
python -m scripts.fetch_fleurs -n 20 --out data/fleurs_ky # кыргызча үлгү аудиолор
```

Бардык буйруктарды долбоордун негизги папкасынан иштетиңиз.

### Мисалдар

| Эмне кылат | Make | CLI |
|---|---|---|
| Файлды текстке айлантуу | `make transcribe FILE=audio.wav` | `python -m whisper_small transcribe audio.wav -t` |
| Субтитр (.srt) | `make srt FILE=audio.wav` | `python -m whisper_small transcribe audio.wav -f srt -o out` |
| JSON | `make json FILE=audio.wav` | `python -m whisper_small transcribe audio.wav -f json -o out` |
| Микрофон | `make mic` | `python -m whisper_small mic --loop` |
| Веб-демо | `make demo` | `pip install gradio && python -m scripts.gradio_demo` |
| Тактык (WER/CER) | `make eval` | `python -m whisper_small evaluate data/fleurs_ky/manifest.tsv` |
| Ылдамдык | `make bench FILE=audio.wav` | `python -m whisper_small bench audio.wav` |
| Тесттер | `make test` | `pytest -q` |

Make өзгөрмөлөрү: `FILE`, `DEVICE` (auto/cpu/mps/cuda), `DTYPE` (auto/fp32/fp16), `BEAMS`, `LANG_ID`, `N`, `SECONDS`.

```bash
make transcribe FILE=my.wav DEVICE=cpu BEAMS=5
```

### Python

```python
from whisper_small import Transcriber

asr = Transcriber()                  # cuda/mps/cpu өзү тандайт
res = asr.transcribe("audio.wav")    # каалаган узундукта
print(res.text)
```

### Натыйжалар

FLEURS кыргызча, 100 клип: **WER 16.8% / CER 4.8%** (сандары жок клиптер).
Apple M5 Pro, fp16: реалдуу убакыттан ~140 эсе ылдам.

> Whisper'де кыргыз тилинин токени жок, ошондуктан модель `kk` токени менен иштейт (демейки).
> Сандар сөз менен жазылат ("жети миң").

---

## License

Apache-2.0. Base weights: [AkylAI-STT-small](https://huggingface.co/the-cramer-project/AkylAI-STT-small) (Apache-2.0).
Evaluation audio: [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
