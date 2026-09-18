"""Interactive clone session: record a reference clip from a mic, then type
lines and hear them spoken in that voice.

Capture goes through PortAudio (sounddevice), not ffmpeg's avfoundation input:
avfoundation mislabels 24-bit USB receivers as pcm_s24le with 4-byte packets
and records pure silence. PortAudio negotiates the format correctly.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.markup import escape

from . import audio as audio_io
from . import outputs
from .config import OUT, SAMPLE_RATE, VOICES
from .engines.f5 import REF_TXT, REF_WAV, F5Engine, saved_voices, voice_dir

console = Console()

# Phonetically broad, ~8 s at a natural pace. F5 wants a 5-10 s reference and
# needs to know exactly what was said, so the user reads this verbatim.
REF_SENTENCE = ("The quick brown fox jumps over the lazy dog, while the morning "
                "sun rises slowly over the quiet hills and the birds begin to sing.")

MAX_REF_SECONDS = 15.0
MIN_REF_SECONDS = 3.0
QUIET_RMS_DB = -35.0


# --------------------------------------------------------------------------- mic

def input_devices() -> list[tuple[int, str, int]]:
    """(index, name, max_input_channels) for every capture-capable device."""
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, d["name"], int(d["max_input_channels"])))
    return out


def pick_device(want: str | None) -> tuple[int | None, str]:
    """Match `want` as a case-insensitive substring; None -> system default."""
    import sounddevice as sd
    devs = input_devices()
    if want:
        for idx, name, _ch in devs:
            if want.lower() in name.lower():
                return idx, name
        names = ", ".join(n for _, n, _ in devs) or "none"
        raise RuntimeError(f"no input device matching '{want}'. Have: {names}")
    idx = sd.default.device[0]
    name = sd.query_devices(idx)["name"] if idx is not None and idx >= 0 else "default"
    return idx, name


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def probe_level(device: int | None, seconds: float = 1.0) -> float:
    """RMS dB of a short capture. Exactly silent (-180) means the device is
    delivering zeros: transmitter off, unpaired, or muted at the receiver."""
    import sounddevice as sd
    info = sd.query_devices(device)
    sr = int(info["default_samplerate"])
    ch = max(1, int(info["max_input_channels"]))
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=ch, dtype="float32", device=device)
    sd.wait()
    return _rms_db(_to_mono(rec))


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Pick the loudest channel. Wireless receivers often carry the transmitter
    on one channel only; averaging with a dead channel halves the level."""
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 1:
        return x[:, 0].astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=0))
    return x[:, int(np.argmax(rms))].astype(np.float32, copy=True)


def record_until_enter(device: int | None, max_seconds: float = MAX_REF_SECONDS,
                       sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record from `device` at its native rate/channels until Enter is pressed
    or max_seconds elapse. Returns mono float32 at `sr`."""
    import sounddevice as sd

    info = sd.query_devices(device)
    dev_sr = int(info["default_samplerate"])
    dev_ch = max(1, int(info["max_input_channels"]))

    chunks: list[np.ndarray] = []
    stop = threading.Event()
    level = {"db": -90.0}

    def cb(indata, _frames, _time, status):
        if status:
            console.print(f"[yellow]{status}[/yellow]")
        block = np.array(indata, dtype=np.float32, copy=True)
        chunks.append(block)
        level["db"] = _rms_db(np.max(np.abs(block), axis=1) if block.ndim > 1 else block)

    def wait_enter():
        try:
            input()
        except EOFError:
            pass
        stop.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    t0 = time.monotonic()
    with sd.InputStream(device=device, channels=dev_ch, samplerate=dev_sr,
                        dtype="float32", callback=cb):
        while not stop.is_set():
            el = time.monotonic() - t0
            if el >= max_seconds:
                console.print(f"\n[yellow]hit {max_seconds:.0f} s limit, stopping[/yellow]")
                break
            bar = "#" * max(0, min(30, int((level["db"] + 60) / 2)))
            sys.stdout.write(f"\r  REC {el:5.1f}s  [{bar:<30}] {level['db']:6.1f} dB   (Enter to stop)")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write("\n")
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    mono = _to_mono(np.concatenate(chunks, axis=0))
    return audio_io.resample(mono, dev_sr, sr) if dev_sr != sr else mono


def play(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        console.print(f"[dim](no player on this platform; file at {path})[/dim]")


# ---------------------------------------------------------------------- session

def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "/q"


def capture_reference(name: str, device: int | None) -> Path:
    """Record, check, play back, and save voices/<name>/ref.wav + ref.txt."""
    vdir = voice_dir(name)
    while True:
        console.print()
        console.print("[bold]Read this out loud, naturally, at a normal pace:[/bold]")
        console.print(f"\n    [cyan]{REF_SENTENCE}[/cyan]\n")
        _ask("Press Enter to START recording, then Enter again when you finish reading... ")
        wav = record_until_enter(device)
        if wav.size == 0:
            console.print("[red]no audio captured[/red]")
            continue

        raw_db = _rms_db(wav)
        wav = audio_io.trim_silence(wav, SAMPLE_RATE)
        dur = len(wav) / SAMPLE_RATE
        console.print(f"  captured {dur:.1f} s, level {raw_db:.1f} dB RMS")
        problems = []
        if dur < MIN_REF_SECONDS:
            problems.append(f"too short ({dur:.1f} s < {MIN_REF_SECONDS:.0f} s)")
        if raw_db < QUIET_RMS_DB:
            problems.append(f"very quiet ({raw_db:.0f} dB) - is the mic on and selected?")
        if problems:
            console.print("[yellow]  " + "; ".join(problems) + "[/yellow]")

        wav = audio_io.peak_normalize(wav)
        vdir.mkdir(parents=True, exist_ok=True)
        ref_path = audio_io.save(vdir / REF_WAV, wav, SAMPLE_RATE)
        (vdir / REF_TXT).write_text(REF_SENTENCE + "\n", encoding="utf-8")

        console.print("  playing back...")
        play(ref_path)
        ans = _ask("Keep this take? [Enter=keep / r=re-record / q=quit] ").lower()
        if ans == "q" or ans == "/q":
            raise SystemExit(0)
        if ans != "r":
            console.print(f"[green]saved voice '{name}' -> {ref_path}[/green]")
            return ref_path


def run(name: str, mic: str | None, reuse: bool, speed: float, nfe: int) -> None:
    engine = F5Engine()
    ok, msg = engine.check()
    if not ok:
        console.print(f"[red]f5 is not usable: {escape(msg)}[/red]")
        raise SystemExit(1)
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        console.print("[red]sounddevice missing - uv pip install -e '.[clone]'[/red]")
        raise SystemExit(1)

    try:
        dev_idx, dev_name = pick_device(mic)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
    console.print(f"mic: [bold]{dev_name}[/bold]  (checking level...)", end=" ")
    try:
        db = probe_level(dev_idx)
    except Exception as e:
        console.print(f"\n[red]cannot open mic: {escape(str(e))}[/red]")
        raise SystemExit(1)
    if db <= -120:
        console.print(f"\n[red]'{dev_name}' is delivering pure silence.[/red] For a wireless "
                      "mic: switch the transmitter on, check it is paired to the receiver, "
                      "and confirm the Input level meter in System Settings > Sound moves "
                      "when you speak. Then run this again.")
        if _ask("Continue anyway? [y/N] ").lower() != "y":
            raise SystemExit(1)
    else:
        console.print(f"room noise {db:.0f} dB, ok")

    ref = voice_dir(name) / REF_WAV
    if ref.exists() and reuse:
        console.print(f"reusing saved voice [bold]{name}[/bold] ({ref})")
    else:
        if ref.exists():
            console.print(f"[dim]voice '{name}' exists; re-recording (use --reuse to keep it)[/dim]")
        ref = capture_reference(name, dev_idx)

    console.print("\nloading F5-TTS (first run downloads ~1.3 GB of weights)...")
    t0 = time.monotonic()
    engine._model()
    console.print(f"ready in {time.monotonic() - t0:.0f} s\n")
    console.print("[bold]Type what you want said. Enter sends it.[/bold]")
    console.print("[dim]  /redo  re-record the reference     /play  replay last output\n"
                  "  /ref   play the reference clip      /q     quit[/dim]\n")

    last: Path | None = None
    while True:
        text = _ask(f"[{name}] > ")
        if not text:
            continue
        if text in ("/q", "/quit", "/exit"):
            break
        if text == "/redo":
            ref = capture_reference(name, dev_idx)
            continue
        if text == "/ref":
            play(ref)
            continue
        if text == "/play":
            if last:
                play(last)
            continue

        t0 = time.monotonic()
        try:
            result = engine.synth(text, voice=name, speed=speed, nfe_step=nfe)
        except Exception as e:
            console.print(f"[red]synthesis failed: {escape(str(e))}[/red]")
            continue
        fname = f"clone_{name}_{datetime.now():%Y%m%d_%H%M%S}.wav"
        last = audio_io.save(OUT / fname, result.wav, result.sr)
        outputs.record(fname, pipeline=f"f5:{name}", text=text)
        console.print(f"  [green]{last.name}[/green]  {result.duration:.1f} s audio "
                      f"in {time.monotonic() - t0:.1f} s")
        play(last)

    console.print(f"\nvoice saved as [bold]{name}[/bold]. Later:  "
                  f"voicelab say \"...\" --engine f5 --voice {name} --play")
