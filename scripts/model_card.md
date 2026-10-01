---
language:
  - ky
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: the-cramer-project/AkylAI-STT-small
tags:
  - whisper
  - kyrgyz
  - speech-recognition
datasets:
  - google/fleurs
metrics:
  - wer
  - cer
model-index:
  - name: kyrgyz-whisper-small
    results:
      - task:
          type: automatic-speech-recognition
        dataset:
          name: FLEURS (ky_kg, dev, 100 clips without digits)
          type: google/fleurs
          config: ky_kg
          split: validation
        metrics:
          - type: wer
            value: 16.3
          - type: cer
            value: 4.5
---

# kyrgyz-whisper-small

Whisper-small for Kyrgyz speech recognition (кыргызча кепти текстке айландыруу).

The weights are [the-cramer-project/AkylAI-STT-small](https://huggingface.co/the-cramer-project/AkylAI-STT-small)
by The Cramer Project, itself a Kyrgyz fine-tune of [openai/whisper-small](https://huggingface.co/openai/whisper-small)
by OpenAI. We did not retrain them; we re-saved them and added a generation setup for Kyrgyz
(`language="kk"`) and evaluation results.

## Usage

```python
import soundfile as sf
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="ramput-labs/kyrgyz-whisper-small")
audio, sr = sf.read("audio.wav", dtype="float32")
print(asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "kk", "task": "transcribe"})["text"])
```

Whisper has no Kyrgyz token, so pass `language="kk"`. It gives the best accuracy.
For audio longer than 30 s, CLI, microphone and web demo, use the
[kyrgyz-whisper-small](https://github.com/ramput-labs/kyrgyz-whisper-small) toolkit:

```bash
make quickstart
```

## Results

FLEURS Kyrgyz dev, 100 clips without digits: **WER 16.3% / CER 4.5%**.
Numbers are written as words ("жети миң"), so references with digits score lower.

## License

Apache-2.0 (see `LICENSE`), as inherited from AkylAI-STT-small and Whisper.
The toolkit code on GitHub is MIT.
