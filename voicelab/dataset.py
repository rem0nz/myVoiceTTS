"""Turn a folder of raw recordings into an RVC-training-ready dataset.

RVC v2 guidance: it wants roughly 10+ minutes of clean single-speaker audio —
no music, no other speakers, no reverb. Clips of 3-12 seconds, with a
consistent mic and room across the whole set. More than ~45 minutes gives
diminishing returns.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console

from .audio import load, peak_normalize, save, split_on_silence, trim_silence
from .config import DATASETS

console = Console()

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".aif", ".ogg",
              ".opus", ".mp4", ".mov"}
RMS_FLOOR_DB = -50.0  # clips quieter than this are silence/noise, not speech


def _rms_db(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return float(20 * np.log10(rms + 1e-12))


def _find_sources(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def prep(src: Path, name: str, target_sr: int = 40000, min_s: float = 3.0,
         max_s: float = 12.0, transcribe: bool = False) -> dict:
    """Slice everything under `src` into normalized clips at DATASETS/name/wavs/."""
    src = Path(src)
    ds_dir = DATASETS / name
    wav_dir = ds_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    files = _find_sources(src)
    if not files:
        raise FileNotFoundError(
            f"no audio files under {src} "
            f"(looked for {', '.join(sorted(AUDIO_EXTS))})"
        )

    clips: list[tuple[Path, float]] = []  # (saved path, duration seconds)
    rejected = 0
    failed: list[str] = []
    counters: dict[str, int] = {}  # per-stem clip index

    for f in files:
        console.print(f"[dim]processing[/dim] {f.name}")
        try:
            wav, sr = load(f, sr=target_sr, mono=True)
            wav = trim_silence(wav, sr)
            pieces = split_on_silence(wav, sr, min_s=min_s, max_s=max_s)
        except Exception as e:
            console.print(f"[yellow]warning:[/yellow] {f}: {e} - skipping file")
            failed.append(str(f))
            continue
        for piece in pieces:
            if _rms_db(piece) < RMS_FLOOR_DB:
                rejected += 1
                continue
            piece = peak_normalize(piece, 0.95)
            idx = counters.get(f.stem, 0)
            counters[f.stem] = idx + 1
            out = save(wav_dir / f"{f.stem}_{idx:04d}.wav", piece, sr)
            clips.append((out, len(piece) / sr))

    transcribed = False
    if transcribe and clips:
        model = None
        try:
            import whisper
            model = whisper.load_model("base")
        except ImportError:
            console.print(
                "[yellow]warning:[/yellow] whisper not installed - skipping "
                "transcripts (`uv pip install openai-whisper`)"
            )
        if model is not None:
            manifest = ds_dir / "manifest.jsonl"
            with manifest.open("w", encoding="utf-8") as fh:
                for path, dur in clips:
                    try:
                        text = model.transcribe(str(path), fp16=False)["text"].strip()
                    except Exception as e:
                        console.print(
                            f"[yellow]warning:[/yellow] transcription failed "
                            f"for {path.name}: {e}"
                        )
                        continue
                    fh.write(json.dumps({
                        "audio": str(path.relative_to(ds_dir)),
                        "text": text,
                        "duration": round(dur, 3),
                    }) + "\n")
            transcribed = True
            console.print(f"[dim]wrote[/dim] {manifest}")

    summary = {
        "name": name,
        "source": str(src),
        "target_sr": target_sr,
        "clip_count": len(clips),
        "total_seconds": round(sum(d for _, d in clips), 2),
        "rejected": rejected,
        "failed": failed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transcribed": transcribed,
    }
    (ds_dir / "dataset.json").write_text(json.dumps(summary, indent=2))
    console.print(
        f"[green]done:[/green] {summary['clip_count']} clips, "
        f"{summary['total_seconds']:.0f}s speech, {rejected} rejected -> {ds_dir}"
    )
    return summary


def stats(name: str) -> dict:
    """Stored summary for DATASETS/name plus a live recount of its wavs."""
    ds_dir = DATASETS / name
    meta_path = ds_dir / "dataset.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"dataset '{name}' not found at {ds_dir} - "
            f"run `voicelab dataset-prep <src> --name {name}` first"
        )
    meta: dict = json.loads(meta_path.read_text())
    wavs = sorted((ds_dir / "wavs").glob("*.wav"))
    total = 0.0
    for w in wavs:  # header-only read; never loads samples
        total += sf.info(str(w)).duration
    meta["live_clip_count"] = len(wavs)
    meta["live_total_seconds"] = round(total, 2)
    return meta
