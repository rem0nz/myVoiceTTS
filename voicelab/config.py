"""Paths and machine-level settings."""
from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path

ROOT = Path(os.environ.get("VOICELAB_HOME", Path.home() / "voicelab")).expanduser()
MODELS = ROOT / "models"
RVC_MODELS = MODELS / "rvc"
TTS_MODELS = MODELS / "tts"
VOICES = ROOT / "voices"
DATASETS = ROOT / "datasets"
OUT = ROOT / "out"
REGISTRY = ROOT / "voices.yaml"

for _p in (MODELS, RVC_MODELS, TTS_MODELS, VOICES, DATASETS, OUT):
    _p.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000  # canonical internal rate; engines resample into this


def torch_device() -> str:
    """Best available torch device. MPS on Apple Silicon, else CUDA, else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def free_disk_gb(path: Path = ROOT) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def ffmpeg_or_die() -> str:
    if not have("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    return "ffmpeg"
