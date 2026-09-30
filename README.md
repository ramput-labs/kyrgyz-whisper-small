# kyrgyz-asr

Speech recognition for **Kyrgyz (кыргыз тили)** built on a Whisper-small Kyrgyz model. It includes
a CLI (`kyasr`), a Python API, evaluation against Google FLEURS, a speed benchmark, microphone
input and a Gradio web demo.

```bash
make quickstart          # venv → model (~1 GB) → 20 FLEURS clips → first transcript
make help                # every target
```

## What's inside

```
kyrgyz-asr/
├── Makefile                 # all entry points (make help)
├── pyproject.toml           # package + `kyasr` console script; extras: mic, demo, dev
├── src/kyrgyz_asr/
│   ├── config.py            # model id, paths, DEFAULT_LANGUAGE="kk"
│   ├── model.py             # download, device/dtype selection, loading
│   ├── audio.py             # load any file → 16 kHz mono, mic recording, pause-based splitter
│   ├── transcriber.py       # Transcriber: batched long-form transcription + language detection
│   ├── evaluate.py          # text normalisation, WER/CER, manifest reader
│   └── cli.py               # `kyasr` command (typer + rich)
├── scripts/
│   ├── fetch_fleurs.py      # stream N Kyrgyz clips + references from FLEURS (no 700 MB download)
│   ├── concat_audio.py      # build long test files
│   └── gradio_demo.py       # web UI (upload / record in the browser)
├── tests/                   # unit tests + end-to-end model tests (auto-skip without weights)
├── models/                  # downloaded weights (git-ignored)
└── data/, samples/, out/    # downloaded audio, generated files (git-ignored)
```

## CLI

```bash
kyasr info                                   # architecture/config summary
kyasr transcribe speech.wav -t               # text + per-segment timestamps
kyasr transcribe *.mp3 -f srt -o out/        # subtitles (also: -f json)
kyasr transcribe speech.wav -b 5 -d cpu      # beam search, force device
kyasr detect-lang speech.wav                 # which Whisper language token it "hears"
kyasr mic -s 5 --loop                        # talk into the microphone
kyasr evaluate data/fleurs_ky/manifest.tsv   # WER / CER
kyasr bench samples/long.wav                 # cpu/mps × fp32/fp16 speed table
```

Each of these has a Makefile shortcut, and every knob can be overridden:
`make transcribe FILE=my.wav DEVICE=cpu LANG_ID=auto BEAMS=5`.

Audio is loaded with libsndfile (wav/flac/ogg/mp3). m4a/mp4/webm need `brew install ffmpeg`.

## Python API

```python
from kyrgyz_asr import Transcriber

asr = Transcriber()                       # picks mps/cuda/cpu and fp16/fp32 automatically
res = asr.transcribe("speech.wav")        # any length
print(res.text, f"RTF={res.rtf:.3f}")
for s in res.segments:
    print(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
```

## Model notes

**Architecture.** `WhisperForConditionalGeneration`, 12 encoder + 12 decoder layers, d_model 768,
80 mel bins, fp32 safetensors (~967 MB). All numbers below were measured with this project.

**No Kyrgyz token, so the model uses the Kazakh one.** Whisper's vocabulary has no `<|ky|>`. The
generation config leaves the language slot open. `kyasr detect-lang` shows the model puts about
80% probability on `kk` (Kazakh, the closest Turkic language Whisper knows), so the fine-tune
evidently trained under that token. On FLEURS ky dev (first 10 clips), with MPS fp16 and greedy
decoding:

| language token | WER | CER |
|---|---|---|
| `kk` (forced, **default**) | **12.4%** | **3.0%** |
| auto (model picks) | 13.6% | 3.4% |
| `ru` (forced) | 43.8% | 18.0% |
| `kk` + beam 5 | 14.2% | 3.5% |

Beam search didn't help here and is about 5× slower, so greedy decoding is the default.

**Bigger sample: 100 FLEURS dev clips** (`make samples N=100 && make eval`):

| subset | utts | WER | CER |
|---|---|---|---|
| all | 100 | 21.0% | 7.8% |
| references without digits | 81 | 16.8% | 4.8% |

Most of the gap comes from formatting, not recognition. The model **writes numbers as words**
("жети миң" for 7000, "бир миң жети жүз алтымыш жетинчи жылдан" for "1767-ж."), while FLEURS
references use digits. The evaluator therefore reports the digit-free subset separately. Other
errors are mostly foreign proper nouns (Хонсю → "ханчу") and single-letter suffix differences.
The median per-utterance CER is about 4.6%.

**Long audio.** The stock HF pipeline (`chunk_length_s=30`) misbehaves with this checkpoint. With
timestamps on, it produced repetition loops ("жаш эле жаш эле …"). With timestamps off, it
silently dropped sentences where chunks were stitched together. The fine-tune doesn't seem to have
kept Whisper's timestamp ability. Instead, `Transcriber` cuts audio at the quietest point in each
10–28 s window, decodes the pieces as one batch, and uses the cut points as segment timestamps.
That gives no stitching, no split words, and complete output. Only near-digital silence is
skipped: some FLEURS speech peaks at about -42 dBFS, and a loudness gate threw it away.

**Speed** (Apple M5 Pro, 121 s file, `make bench`):

| device | dtype | time | RTF |
|---|---|---|---|
| cpu | fp32 | 6.6 s | 0.054 |
| mps | fp32 | 2.1 s | 0.017 |
| mps | fp16 | **0.86 s** | **0.007** (~140× real-time) |

**Other checkpoint quirks handled in code:** `use_cache=False` was left over from training and is
turned back on for faster decoding. The duplicate `max_length`/`max_new_tokens` settings and the
noisy transformers-v5 warnings are silenced.

## Ideas to play with next

- Run `make samples N=500 SPLIT=test` for a proper test-set number. Add a Kyrgyz number verbaliser
  so digits don't count as errors.
- Compare against `openai/whisper-small` with `--language kk` (`kyasr transcribe -m openai/whisper-small`)
  to see what the fine-tune bought.
- Speed up further with `torch.compile`, a CTranslate2/faster-whisper conversion, or
  whisper.cpp / MLX export for on-device use.
- Stream from the microphone: a VAD loop on top of `audio.split_on_silence`.
- Fine-tune further on Common Voice `ky` with `Seq2SeqTrainer`, reusing `evaluate.normalize` for metrics.

## Credits

Model weights: [`the-cramer-project/AkylAI-STT-small`](https://huggingface.co/the-cramer-project/AkylAI-STT-small)
(Apache-2.0), downloaded by `make download`. Evaluation audio: [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
