"""Engine contracts. Every backend is optional and lazily imported."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Audio:
    wav: np.ndarray
    sr: int

    @property
    def duration(self) -> float:
        return len(self.wav) / self.sr if self.sr else 0.0


class Engine(ABC):
    name: str = "base"
    kind: str = "tts"           # "tts" | "vc"
    clones: bool = False        # can it match a target voice from a reference?
    extra: str = ""             # pip extra that installs it
    notes: str = ""

    @abstractmethod
    def check(self) -> tuple[bool, str]:
        """(installed_and_usable, human message). Must never raise."""

    def voices(self) -> list[str]:
        return []


class TTSEngine(Engine):
    kind = "tts"

    @abstractmethod
    def synth(self, text: str, voice: str | None = None,
              ref: Path | None = None, speed: float = 1.0, **kw) -> Audio:
        ...


class VCEngine(Engine):
    """Speech-to-speech: keeps timing and prosody, swaps timbre."""
    kind = "vc"

    @abstractmethod
    def convert(self, audio: Audio, model: str, pitch: int = 0, **kw) -> Audio:
        ...


def _missing(extra: str, pkg: str) -> tuple[bool, str]:
    return False, f"not installed - `uv pip install -e '.[{extra}]'` (provides {pkg})"
