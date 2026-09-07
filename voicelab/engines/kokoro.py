"""Kokoro-82M. Apache-2.0, fast, high quality, fixed voice set (no cloning)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .base import TTSEngine, Audio, _missing
from ..audio import resample
from ..config import SAMPLE_RATE, have

VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",   # b* = British
]


class KokoroEngine(TTSEngine):
    name = "kokoro"
    clones = False
    extra = "kokoro"
    notes = "Apache-2.0, 82M params, ~0.3s/sentence on MPS. bf_/bm_ voices are British."

    _pipe = None

    def check(self) -> tuple[bool, str]:
        try:
            import kokoro  # noqa: F401
        except ImportError:
            return _missing(self.extra, "kokoro")
        if not have("espeak-ng"):
            return False, "kokoro installed but espeak-ng missing - `brew install espeak-ng`"
        return True, "ready"

    def voices(self) -> list[str]:
        return VOICES

    def _pipeline(self, lang: str):
        if self._pipe is None:
            from kokoro import KPipeline
            self._pipe = KPipeline(lang_code=lang)
        return self._pipe

    def synth(self, text: str, voice: str | None = None,
              ref: Path | None = None, speed: float = 1.0, **kw) -> Audio:
        if ref is not None:
            raise ValueError("kokoro cannot clone; it has a fixed voice set")
        voice = voice or "bf_emma"
        lang = "b" if voice.startswith(("bf_", "bm_")) else "a"
        pipe = self._pipeline(lang)
        chunks = [audio for _, _, audio in pipe(text, voice=voice, speed=speed)]
        if not chunks:
            raise RuntimeError("kokoro produced no audio")
        wav = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
        return Audio(resample(wav, 24000, SAMPLE_RATE), SAMPLE_RATE)
