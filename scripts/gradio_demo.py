#!/usr/bin/env python
"""Tiny web UI: upload a file or record in the browser, get Kyrgyz text + segments.

    make demo            # or: python scripts/gradio_demo.py --share
"""

from __future__ import annotations

import argparse

import gradio as gr

from kyrgyz_asr import Transcriber
from kyrgyz_asr.audio import to_mono_16k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    args = ap.parse_args()

    asr = Transcriber(device=args.device)

    def run(audio, language, beams):
        if audio is None:
            return "", []
        sr, data = audio
        audio_f = data.astype("float32")
        if data.dtype.kind == "i":  # gradio hands back int16 PCM
            audio_f /= 32768.0
        res = asr.transcribe(to_mono_16k(audio_f, sr), language=language, num_beams=int(beams))
        rows = [[f"{s.start:.2f}", f"{s.end:.2f}", s.text] for s in res.segments]
        return f"{res.text}\n\n({res.audio_seconds:.1f}s audio in {res.elapsed_seconds:.2f}s on {asr.device})", rows

    with gr.Blocks(title="kyrgyz-asr") as ui:
        gr.Markdown("## kyrgyz-asr · Кыргызча кепти текстке айландыруу")
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

    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
