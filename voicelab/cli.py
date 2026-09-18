"""voicelab command-line interface."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import audio as audio_io
from . import dataset
from .config import (OUT, RVC_MODELS, SAMPLE_RATE, TTS_MODELS,
                     free_disk_gb, have, torch_device)
from .engines.base import Audio, Engine, TTSEngine
from .engines.f5 import F5Engine
from .engines.kokoro import KokoroEngine
from .engines.piper import PiperEngine
from .engines.system import SystemEngine

app = typer.Typer(help="Local TTS + voice conversion toolbox.",
                  no_args_is_help=True, add_completion=False)
console = Console()

RVC_INSTALL_HINT = "install the rvc extra first: uv pip install -e '.[rvc]'"


def _engines() -> dict[str, Engine]:
    """name -> engine instance, for everything importable on this machine."""
    registry: dict[str, Engine] = {}
    for cls in (SystemEngine, KokoroEngine, F5Engine, PiperEngine):
        inst = cls()
        registry[inst.name] = inst
    try:
        from .engines.rvc import RVCEngine  # optional; module may not exist yet
        registry["rvc"] = RVCEngine()
    except ImportError:
        pass
    return registry


def _print_summary(d: dict, title: str) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    for k, v in d.items():
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) if v else "-"
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def doctor() -> None:
    """Check engines, hardware, and installed models."""
    table = Table(title="engines")
    table.add_column("engine")
    table.add_column("kind")
    table.add_column("clones")
    table.add_column("status")
    for name, eng in _engines().items():
        ok, msg = eng.check()
        safe = escape(msg)
        status = f"[green]{safe}[/green]" if ok else f"[red]{safe}[/red]"
        table.add_row(name, eng.kind, "yes" if eng.clones else "no", status)
    console.print(table)

    disk = free_disk_gb()
    disk_str = f"{disk:.1f} GB free"
    if disk < 10:
        disk_str = f"[red]{disk_str} - low! model downloads need headroom[/red]"
    rvc_count = sum(1 for d in RVC_MODELS.iterdir()
                    if d.is_dir() and any(d.glob("*.pth")))
    piper_count = len(list(TTS_MODELS.glob("*.onnx")))

    sys_table = Table(title="system")
    sys_table.add_column("item", style="bold")
    sys_table.add_column("value")
    sys_table.add_row("python", platform.python_version())
    sys_table.add_row("torch device", torch_device())
    sys_table.add_row("disk", disk_str)
    sys_table.add_row("ffmpeg", "[green]present[/green]" if have("ffmpeg")
                      else "[red]missing - brew install ffmpeg[/red]")
    sys_table.add_row("espeak-ng", "[green]present[/green]" if have("espeak-ng")
                      else "[red]missing - brew install espeak-ng[/red]")
    sys_table.add_row("rvc models", str(rvc_count))
    sys_table.add_row("piper voices", str(piper_count))
    console.print(sys_table)


@app.command()
def voices(
    engine: str | None = typer.Option(
        None, "--engine", help="List voices for this engine only."),
) -> None:
    """List available voices, per engine."""
    registry = _engines()
    if engine is not None and engine not in registry:
        console.print(f"[red]unknown engine '{engine}'. "
                      f"Have: {', '.join(registry)}[/red]")
        raise typer.Exit(1)
    targets = {engine: registry[engine]} if engine else registry
    for name, eng in targets.items():
        ok, msg = eng.check()
        if not ok:
            console.print(f"[bold]{name}[/bold]: [dim]unavailable - {escape(msg)}[/dim]")
            continue
        names = eng.voices()
        console.print(f"[bold]{name}[/bold] "
                      f"({len(names)} voice{'s' if len(names) != 1 else ''})")
        for v in names:
            console.print(f"  {v}")


@app.command()
def say(
    text: str = typer.Argument(..., help="Text to speak."),
    engine: str = typer.Option(
        "kokoro", "--engine", help="TTS engine (see `voicelab doctor`)."),
    voice: str | None = typer.Option(
        None, "--voice", help="Voice name (see `voicelab voices`)."),
    ref: Path | None = typer.Option(
        None, "--ref", help="Reference audio for cloning engines."),
    speed: float = typer.Option(1.0, "--speed", help="Speaking-rate multiplier."),
    rvc: str | None = typer.Option(
        None, "--rvc", help="RVC model to convert the result through."),
    pitch: int = typer.Option(0, "--pitch", help="RVC pitch shift in semitones."),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output wav path (default: out/say_<timestamp>.wav)."),
    play: bool = typer.Option(
        False, "--play", help="Play the result with afplay (macOS only)."),
) -> None:
    """Synthesize speech, optionally converting it through an RVC voice."""
    registry = _engines()
    if engine not in registry:
        console.print(f"[red]unknown engine '{engine}'. "
                      f"Have: {', '.join(registry)}[/red]")
        raise typer.Exit(1)
    eng = registry[engine]
    ok, msg = eng.check()
    if not ok and engine == "kokoro":
        console.print(f"[yellow]note:[/yellow] kokoro unavailable ({escape(msg)}) - "
                      "falling back to the system voice")
        eng = registry["system"]
        ok, msg = eng.check()
    if not ok:
        console.print(f"[red]{eng.name} is not usable: {escape(msg)}[/red]")
        raise typer.Exit(1)
    if not isinstance(eng, TTSEngine):
        console.print(f"[red]{eng.name} is voice-conversion only; "
                      "use `voicelab convert` instead[/red]")
        raise typer.Exit(1)

    try:
        result = eng.synth(text, voice=voice, ref=ref, speed=speed)
    except Exception as e:
        console.print(f"[red]{eng.name} synthesis failed: {e}[/red]")
        raise typer.Exit(1)

    if rvc is not None:
        try:
            from .engines.rvc import RVCEngine
        except ImportError:
            console.print(f"[red]--rvc needs RVC - {RVC_INSTALL_HINT}[/red]")
            raise typer.Exit(1)
        try:
            result = RVCEngine().convert(result, model=rvc, pitch=pitch)
        except Exception as e:
            console.print(f"[red]rvc conversion failed: {e}[/red]")
            raise typer.Exit(1)

    if out is None:
        out = OUT / f"say_{datetime.now():%Y%m%d_%H%M%S}.wav"
    try:
        audio_io.save(out, result.wav, result.sr)
    except Exception as e:
        console.print(f"[red]could not write {out}: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{out}[/green]  ({result.duration:.2f}s)")

    if play:
        if sys.platform != "darwin":
            console.print("[yellow]--play uses afplay and only works on macOS[/yellow]")
        else:
            subprocess.run(["afplay", str(out)], check=False)


@app.command()
def convert(
    input_path: Path = typer.Argument(
        ..., metavar="INPUT", help="Existing audio file to convert."),
    model: str = typer.Option(
        ..., "--model", help="RVC model name under models/rvc/."),
    pitch: int = typer.Option(0, "--pitch", help="Pitch shift in semitones."),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output wav path (default: out/<stem>_<model>.wav)."),
) -> None:
    """Convert an existing recording through an RVC voice model."""
    if not input_path.exists():
        console.print(f"[red]{input_path} does not exist[/red]")
        raise typer.Exit(1)
    try:
        from .engines.rvc import RVCEngine
    except ImportError:
        console.print(f"[red]convert needs RVC - {RVC_INSTALL_HINT}[/red]")
        raise typer.Exit(1)
    try:
        wav, sr = audio_io.load(input_path, sr=SAMPLE_RATE)
        result = RVCEngine().convert(Audio(wav, sr), model=model, pitch=pitch)
    except Exception as e:
        console.print(f"[red]conversion failed: {e}[/red]")
        raise typer.Exit(1)
    if out is None:
        out = OUT / f"{input_path.stem}_{model}.wav"
    try:
        audio_io.save(out, result.wav, result.sr)
    except Exception as e:
        console.print(f"[red]could not write {out}: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{out}[/green]  ({result.duration:.2f}s)")


@app.command()
def clone(
    name: str = typer.Option("me", "--name", help="Name to save the voice under (voices/<name>/)."),
    mic: str | None = typer.Option(
        None, "--mic", help="Input device name substring, e.g. 'Wireless Mic Rx'. Default: system input."),
    reuse: bool = typer.Option(
        False, "--reuse", help="Skip recording if voices/<name>/ref.wav already exists."),
    speed: float = typer.Option(1.0, "--speed", help="Speaking-rate multiplier."),
    nfe: int = typer.Option(
        16, "--nfe", help="F5 diffusion steps. 16 is ~2x faster than 32 (the quality default); use 32 for a final take."),
) -> None:
    """Record your voice from a mic, then type lines and hear them in that voice (F5-TTS)."""
    from . import clone as clone_session
    clone_session.run(name=name, mic=mic, reuse=reuse, speed=speed, nfe=nfe)


@app.command("mics")
def mics() -> None:
    """List audio input devices usable by `clone --mic`."""
    try:
        from .clone import input_devices
        devs = input_devices()
    except ImportError:
        console.print("[red]sounddevice missing - uv pip install -e '.[clone]'[/red]")
        raise typer.Exit(1)
    for idx, dev_name, ch in devs:
        console.print(f"  [{idx}] {escape(dev_name)}  ({ch} ch)")


@app.command("dataset-prep")
def dataset_prep(
    src: Path = typer.Argument(..., help="Folder of raw recordings."),
    name: str = typer.Option(..., "--name", help="Dataset name under datasets/."),
    sr: int = typer.Option(
        40000, "--sr", help="Target sample rate (RVC v2: 40000 or 48000)."),
    transcribe: bool = typer.Option(
        False, "--transcribe/--no-transcribe",
        help="Whisper-transcribe clips into manifest.jsonl."),
) -> None:
    """Slice raw recordings into an RVC-training-ready dataset."""
    if not src.is_dir():
        console.print(f"[red]{src} is not a directory[/red]")
        raise typer.Exit(1)
    try:
        summary = dataset.prep(src, name, target_sr=sr, transcribe=transcribe)
    except Exception as e:
        console.print(f"[red]dataset prep failed: {e}[/red]")
        raise typer.Exit(1)
    _print_summary(summary, title=f"dataset '{name}'")


@app.command("dataset-stats")
def dataset_stats(
    name: str = typer.Argument(..., help="Dataset name under datasets/."),
) -> None:
    """Show a dataset's stored summary plus a live recount of its wavs."""
    try:
        info = dataset.stats(name)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    _print_summary(info, title=f"dataset '{name}'")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8080, "--port", help="Port to listen on."),
) -> None:
    """Run the voicelab HTTP API."""
    try:
        import uvicorn
        from voicelab.server import app as server_app
    except ImportError as e:
        console.print(f"[red]web server dependencies missing ({e}) - "
                      "run: uv pip install -e '.[web]'[/red]")
        raise typer.Exit(1)
    uvicorn.run(server_app, host=host, port=port)


if __name__ == "__main__":
    app()
