"""Audio I/O and DSP helpers. Deliberately numpy+soundfile only, no torch."""
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from .config import ffmpeg_or_die


def load(path: Path | str, sr: int | None = None, mono: bool = True):
    """Load audio -> (float32 samples in [-1,1], sample_rate). Resamples via ffmpeg if asked."""
    path = Path(path)
    if sr is None:
        data, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
        if mono and data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        return data, in_sr
    # ffmpeg handles every container + resamples in one pass
    ffmpeg_or_die()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "wav", str(tmp_path)],
            check=True,
        )
        data, out_sr = sf.read(str(tmp_path), dtype="float32", always_2d=True)
        data = data.mean(axis=1) if (mono and data.shape[1] > 1) else data[:, 0]
        return data, out_sr
    finally:
        tmp_path.unlink(missing_ok=True)


def save(path: Path | str, wav: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr)
    return path


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample. Fine for glue between engines; use librosa for training data."""
    if sr_in == sr_out:
        return wav
    n_out = int(round(len(wav) * sr_out / sr_in))
    x_in = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, wav).astype(np.float32)


def peak_normalize(wav: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if len(wav) else 0.0
    return (wav * (peak / m)).astype(np.float32) if m > 1e-9 else wav


def trim_silence(wav: np.ndarray, sr: int, thresh_db: float = -45.0,
                 pad_ms: int = 60) -> np.ndarray:
    """Trim leading/trailing silence using a frame-wise dBFS gate."""
    if len(wav) == 0:
        return wav
    frame = max(1, int(sr * 0.010))
    n = len(wav) // frame
    if n == 0:
        return wav
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    voiced = np.where(db > thresh_db)[0]
    if len(voiced) == 0:
        return wav
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(wav), (voiced[-1] + 1) * frame + pad)
    return wav[start:end]


def split_on_silence(wav: np.ndarray, sr: int, min_s: float = 3.0, max_s: float = 12.0,
                     thresh_db: float = -45.0, min_gap_ms: int = 300) -> list[np.ndarray]:
    """Segment a long recording into training-sized utterances."""
    frame = max(1, int(sr * 0.010))
    n = len(wav) // frame
    if n == 0:
        return [wav]
    frames = wav[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1) + 1e-12) + 1e-12)
    voiced = db > thresh_db
    min_gap = max(1, int(min_gap_ms / 10))

    segs, run_start, silence = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if run_start is None:
                run_start = i
            silence = 0
        elif run_start is not None:
            silence += 1
            if silence >= min_gap:
                segs.append((run_start, i - silence + 1))
                run_start = None
    if run_start is not None:
        segs.append((run_start, len(voiced)))

    out: list[np.ndarray] = []
    for a, b in segs:
        clip = wav[a * frame : b * frame]
        dur = len(clip) / sr
        if dur < min_s:
            continue
        if dur <= max_s:
            out.append(clip)
        else:  # hard-split overlong runs
            step = int(max_s * sr)
            for k in range(0, len(clip), step):
                piece = clip[k : k + step]
                if len(piece) / sr >= min_s:
                    out.append(piece)
    return out
