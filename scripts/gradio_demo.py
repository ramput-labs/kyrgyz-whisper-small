"""Web UI: upload or record audio in the browser and get Kyrgyz text."""

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np

from kyrgyz_whisper_small import Transcriber
from kyrgyz_whisper_small.audio import to_mono_16k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    ap.add_argument("--open", action="store_true", help="Open the UI in the default browser.")
    args = ap.parse_args()

    asr = Transcriber(device=args.device)

    def run(audio, language, beams):
        if audio is None:
            return "", []
        sr, data = audio
        audio_f = data.astype(np.float32)
        if data.dtype.kind == "i":
            audio_f /= np.iinfo(data.dtype).max + 1
        res = asr.transcribe(to_mono_16k(audio_f, sr), language=language, num_beams=int(beams))
        rows = [[f"{s.start:.2f}", f"{s.end:.2f}", s.text] for s in res.segments]
        return f"{res.text}\n\n({res.audio_seconds:.1f}s audio in {res.elapsed_seconds:.2f}s on {asr.device})", rows

    with gr.Blocks(title="kyrgyz-whisper-small") as ui:
        gr.Markdown("## kyrgyz-whisper-small · Кыргызча кепти текстке айландыруу")
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio")
                language = gr.Dropdown(["kk", "auto", "ru"], value="kk", label="Whisper language token")
                beams = gr.Slider(1, 5, value=1, step=1, label="Beam size")
                btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                text = gr.Textbox(label="Transcript", lines=8)
                segs = gr.Dataframe(headers=["start", "end", "text"], label="Segments", wrap=True)
        btn.click(run, [audio, language, beams], [text, segs])

    ui.launch(share=args.share, inbrowser=args.open)


if __name__ == "__main__":
    main()
