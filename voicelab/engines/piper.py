"""Piper. ONNX, CPU-only, tiny. Good for bulk/offline work with no torch install."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .base import TTSEngine, Audio, _missing
from ..audio import resample
from ..config import TTS_MODELS, SAMPLE_RATE


class PiperEngine(TTSEngine):
    name = "piper"
    clones = False
    extra = "piper"
    notes = ("No torch needed. Drop <voice>.onnx + <voice>.onnx.json into models/tts/. "
             "Voices: huggingface.co/rhasspy/piper-voices")

    def check(self) -> tuple[bool, str]:
        try:
            from piper.voice import PiperVoice  # noqa: F401
        except ImportError:
            return _missing(self.extra, "piper-tts")
        n = len(list(TTS_MODELS.glob("*.onnx")))
        return True, f"ready ({n} voice{'s' if n != 1 else ''} in models/tts/)"

    def voices(self) -> list[str]:
        return sorted(p.stem for p in TTS_MODELS.glob("*.onnx"))

    def synth(self, text: str, voice: str | None = None,
              ref: Path | None = None, speed: float = 1.0, **kw) -> Audio:
        if ref is not None:
            raise ValueError("piper cannot zero-shot clone; fine-tune a voice instead")
        from piper.voice import PiperVoice

        available = self.voices()
        if not available:
            raise RuntimeError(
                "no piper voices in models/tts/. Fetch one, e.g.:\n"
                "  curl -L -o ~/voicelab/models/tts/en_GB-alan-medium.onnx "
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx\n"
                "  curl -L -o ~/voicelab/models/tts/en_GB-alan-medium.onnx.json "
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"
            )
        name = voice or available[0]
        model = TTS_MODELS / f"{name}.onnx"
        if not model.exists():
            raise FileNotFoundError(f"{model} not found. Have: {', '.join(available)}")

        v = PiperVoice.load(str(model))
        pcm = b"".join(chunk.audio_int16_bytes for chunk in v.synthesize(text))
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return Audio(resample(wav, v.config.sample_rate, SAMPLE_RATE), SAMPLE_RATE)
