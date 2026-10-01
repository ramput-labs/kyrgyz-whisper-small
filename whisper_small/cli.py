from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .config import DEFAULT_LANGUAGE, DEFAULT_MODEL_DIR

app = typer.Typer(help="whisper-small: Kyrgyz speech recognition (Whisper-small).", no_args_is_help=True)
console = Console()


class Fmt(str, Enum):
    text = "text"
    json = "json"
    srt = "srt"


ModelOpt = typer.Option(None, "--model", "-m", help="Local dir or Hub id (default: ./models snapshot, else Hub).")
DeviceOpt = typer.Option("auto", "--device", "-d", help="auto | mps | cuda | cpu")
DtypeOpt = typer.Option("auto", "--dtype", help="auto | fp32 | fp16 | bf16")
BeamsOpt = typer.Option(1, "--beams", "-b", help="Beam size (1 = greedy, fastest).")
LangOpt = typer.Option(DEFAULT_LANGUAGE, "--language", "-l", help="Whisper language token. 'kk' (default) works best; 'auto' lets the model pick.")


def _load(model, device, dtype):
    from .transcriber import Transcriber

    with console.status(f"Loading model on [bold]{device}[/] ..."):
        t0 = time.perf_counter()
        asr = Transcriber(model=model, device=device, dtype=dtype)
    console.print(f"[dim]model ready on {asr.device} ({str(asr.dtype).removeprefix('torch.')}) in {time.perf_counter() - t0:.1f}s[/]")
    return asr


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _to_srt(result) -> str:
    lines = []
    for i, seg in enumerate(result.segments, 1):
        lines += [str(i), f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}", seg.text, ""]
    return "\n".join(lines)


@app.command()
def download(local_dir: Path = typer.Option(DEFAULT_MODEL_DIR, help="Where to store the snapshot.")):
    """Download the model weights from the Hugging Face Hub."""
    from .model import download_model

    with console.status("Downloading model weights (~1 GB) ..."):
        path = download_model(local_dir)
    console.print(f"[green]✓[/] weights -> {path}")


@app.command()
def info(model: Optional[str] = ModelOpt):
    """Show model architecture, size and generation defaults."""
    from transformers import AutoConfig, GenerationConfig

    from .model import resolve_device, resolve_model_path

    path = resolve_model_path(model)
    cfg = AutoConfig.from_pretrained(path)
    gen = GenerationConfig.from_pretrained(path)
    t = Table(title="whisper-small", show_header=False)
    rows = {
        "source": path,
        "architecture": cfg.architectures[0],
        "encoder / decoder layers": f"{cfg.encoder_layers} / {cfg.decoder_layers}",
        "d_model / heads": f"{cfg.d_model} / {cfg.encoder_attention_heads}",
        "mel bins": cfg.num_mel_bins,
        "vocab size": cfg.vocab_size,
        "max target tokens": cfg.max_target_positions,
        "task token": "transcribe (forced)",
        "language token": f"{DEFAULT_LANGUAGE} (no <|ky|> in Whisper; model uses the Kazakh slot)",
        "has 'ky' token": "<|ky|>" in gen.lang_to_id,
        "best device here": resolve_device(),
    }
    for k, v in rows.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def transcribe(
    files: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="Audio files (wav/flac/mp3/ogg, others need ffmpeg)."),
    model: Optional[str] = ModelOpt,
    device: str = DeviceOpt,
    dtype: str = DtypeOpt,
    beams: int = BeamsOpt,
    language: str = LangOpt,
    timestamps: bool = typer.Option(False, "--timestamps", "-t", help="Print per-segment timestamps (segments are cut at pauses)."),
    fmt: Fmt = typer.Option(Fmt.text, "--format", "-f", help="text | json | srt"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Write <name>.<fmt> files here instead of stdout."),
):
    """Transcribe one or more audio files."""
    asr = _load(model, device, dtype)
    for f in files:
        res = asr.transcribe(f, language=language, num_beams=beams)
        if fmt is Fmt.json:
            payload = json.dumps(
                {
                    "file": str(f),
                    "text": res.text,
                    "audio_seconds": round(res.audio_seconds, 2),
                    "elapsed_seconds": round(res.elapsed_seconds, 3),
                    "rtf": round(res.rtf, 4),
                    "segments": [s.__dict__ for s in res.segments],
                },
                ensure_ascii=False,
                indent=2,
            )
        elif fmt is Fmt.srt:
            payload = _to_srt(res)
        else:
            payload = res.text

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{f.stem}.{fmt.value if fmt is not Fmt.text else 'txt'}"
            dest.write_text(payload + "\n", encoding="utf-8")
            console.print(f"[green]✓[/] {f.name} -> {dest}")
        else:
            console.rule(f"[bold]{f.name}[/] [dim]{res.audio_seconds:.1f}s audio · {res.elapsed_seconds:.2f}s · RTF {res.rtf:.3f}[/]")
            if timestamps and fmt is Fmt.text:
                for s in res.segments:
                    console.print(f"[cyan][{s.start:6.2f} → {s.end:6.2f}][/] {s.text}")
            else:
                console.print(payload, markup=False, highlight=False)


@app.command("detect-lang")
def detect_lang(
    files: list[Path] = typer.Argument(..., exists=True, dir_okay=False),
    top: int = typer.Option(5, "--top", "-k"),
    model: Optional[str] = ModelOpt,
    device: str = DeviceOpt,
    dtype: str = DtypeOpt,
):
    """Show which Whisper language tokens the model believes it hears."""
    asr = _load(model, device, dtype)
    for f in files:
        t = Table(title=f.name)
        t.add_column("lang")
        t.add_column("prob", justify="right")
        for code, p in asr.detect_language(f, top_k=top):
            t.add_row(code, f"{p:.3f}")
        console.print(t)


@app.command()
def mic(
    seconds: float = typer.Option(5.0, "--seconds", "-s", help="Recording length."),
    loop: bool = typer.Option(False, "--loop", help="Keep recording until Ctrl+C."),
    save: Optional[Path] = typer.Option(None, help="Also save the recording as wav."),
    model: Optional[str] = ModelOpt,
    device: str = DeviceOpt,
    dtype: str = DtypeOpt,
    beams: int = BeamsOpt,
    language: str = LangOpt,
):
    """Record from the microphone and transcribe (needs sounddevice)."""
    from .audio import record, save_wav

    asr = _load(model, device, dtype)
    try:
        while True:
            typer.prompt("Press Enter to start recording", default="", show_default=False)
            with console.status(f"[red]● recording {seconds:.0f}s — сүйлөңүз![/]"):
                audio = record(seconds)
            if save:
                save_wav(save, audio)
            res = asr.transcribe(audio, language=language, num_beams=beams)
            console.print(f"[bold green]»[/] {res.text or '[dim](nothing recognised)[/]'}  [dim]({res.elapsed_seconds:.2f}s)[/]")
            if not loop:
                break
    except KeyboardInterrupt:
        console.print("\nbye")


@app.command()
def evaluate(
    manifest: Path = typer.Argument(..., exists=True, help="TSV with columns path<TAB>text."),
    limit: Optional[int] = typer.Option(None, "--limit", "-n"),
    show: int = typer.Option(5, help="Print this many ref/hyp pairs."),
    report: Optional[Path] = typer.Option(None, help="Write per-utterance results as JSONL."),
    model: Optional[str] = ModelOpt,
    device: str = DeviceOpt,
    dtype: str = DtypeOpt,
    beams: int = BeamsOpt,
    language: str = LangOpt,
):
    """Compute WER/CER on a manifest (e.g. from `make samples`)."""
    from .evaluate import has_digits, normalize, read_manifest, score

    utts = read_manifest(manifest)[:limit]
    asr = _load(model, device, dtype)
    audio_s = elapsed = 0.0
    for u in track(utts, description="Transcribing", console=console):
        r = asr.transcribe(u.path, language=language, num_beams=beams)
        u.hypothesis = r.text
        audio_s += r.audio_seconds
        elapsed += r.elapsed_seconds

    for u in utts[:show]:
        console.print(f"[dim]{u.path.name}[/]\n  REF: {normalize(u.reference)}\n  HYP: {normalize(u.hypothesis)}")

    t = Table(title=f"{len(utts)} utterances · {audio_s / 60:.1f} min audio · {asr.device}/{str(asr.dtype).removeprefix('torch.')} · RTF {elapsed / audio_s:.3f}")
    for col in ("subset", "utts", "WER", "CER"):
        t.add_column(col, justify="right")
    no_digits = [u for u in utts if not has_digits(u.reference)]
    for name, subset in (("all", utts), ("refs without digits", no_digits)):
        if subset:
            m = score([u.reference for u in subset], [u.hypothesis for u in subset])
            t.add_row(name, str(len(subset)), f"{m['wer']:.2%}", f"{m['cer']:.2%}")
    console.print(t)
    console.print("[dim]The model spells numbers out (жети миң) while FLEURS refs use digits (7000); "
                  "the digit-free subset shows recognition quality without that formatting penalty.[/]")

    if report:
        with report.open("w", encoding="utf-8") as fh:
            for u in utts:
                fh.write(json.dumps({"path": str(u.path), "ref": u.reference, "hyp": u.hypothesis}, ensure_ascii=False) + "\n")
        console.print(f"[green]✓[/] report -> {report}")


@app.command()
def bench(
    file: Path = typer.Argument(..., exists=True, dir_okay=False),
    runs: int = typer.Option(3, "--runs", "-r"),
    devices: str = typer.Option("cpu,mps", help="Comma-separated devices to compare."),
    dtypes: str = typer.Option("fp32,fp16", help="Comma-separated dtypes to compare."),
    model: Optional[str] = ModelOpt,
):
    """Compare speed across device/dtype combinations."""
    from .audio import load_audio
    from .transcriber import Transcriber

    audio = load_audio(file)
    t = Table(title=f"{file.name} ({len(audio) / 16000:.1f}s)")
    for col in ("device", "dtype", "load s", "best s", "RTF", "text"):
        t.add_column(col)
    for dev in devices.split(","):
        for dt in dtypes.split(","):
            if dev == "cpu" and dt == "fp16":
                continue
            try:
                t0 = time.perf_counter()
                asr = Transcriber(model=model, device=dev, dtype=dt)
                load_s = time.perf_counter() - t0
                asr.transcribe(audio)
                best = min(asr.transcribe(audio).elapsed_seconds for _ in range(runs))
                text = asr.transcribe(audio).text
                t.add_row(dev, dt, f"{load_s:.1f}", f"{best:.2f}", f"{best / (len(audio) / 16000):.3f}", text[:40] + "…")
                del asr
            except Exception as e:  # noqa: BLE001
                t.add_row(dev, dt, "-", "-", "-", f"[red]{type(e).__name__}: {e}"[:60])
    console.print(t)


if __name__ == "__main__":
    app()
