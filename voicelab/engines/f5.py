"""F5-TTS. Zero-shot cloning from a 5-10 s reference clip. No training step.

A saved voice is a folder under voices/<name>/ holding:
    ref.wav   the reference clip (mono, any rate; 5-10 s is the sweet spot)
    ref.txt   the exact words spoken in ref.wav

`voices()` lists those folders, so the web UI's voice picker and `say --voice`
work with no ref path at all. `say --ref some.wav` still works; the transcript
is taken from a sibling .txt if present, otherwise F5 runs Whisper on the clip,
which is slow and pulls a 1.5 GB model on first use - recording with `clone`
avoids that entirely because it knows what you read.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .base import TTSEngine, Audio, _missing
from ..audio import resample
from ..config import SAMPLE_RATE, VOICES, torch_device

REF_WAV = "ref.wav"
REF_TXT = "ref.txt"

# Module-level so the server, which builds a fresh engine per request, does not
# reload 1.3 GB of weights on every synth. One model per process.
_MODEL = None


def _patch_torchaudio_load() -> None:
    """torchaudio>=2.9 delegates load() to torchcodec, which dlopens FFmpeg's
    shared libs and cannot find Homebrew's /opt/homebrew/lib from a uv-managed
    Python (OSError: Could not load libtorchcodec_core*.dylib). F5 calls
    torchaudio.load exactly once, on the reference wav, so serve that call with
    soundfile instead. Same contract: (Tensor[channels, frames], sr)."""
    import torchaudio

    if getattr(torchaudio.load, "_voicelab_patched", False):
        return

    def _load(path, *_a, **_k):
        import soundfile as sf
        import torch
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(wav.T.copy()), sr

    _load._voicelab_patched = True  # type: ignore[attr-defined]
    torchaudio.load = _load


def voice_dir(name: str) -> Path:
    return VOICES / name


def saved_voices() -> list[str]:
    if not VOICES.exists():
        return []
    return sorted(d.name for d in VOICES.iterdir()
                  if d.is_dir() and (d / REF_WAV).exists())


class F5Engine(TTSEngine):
    name = "f5"
    clones = True
    extra = "f5"
    notes = ("Zero-shot cloning, ~335M params, CC-BY-NC weights. Record a voice with "
             "`voicelab clone`, then pick it as the voice. ~10-20 s per sentence on M1.")

    def check(self) -> tuple[bool, str]:
        try:
            import f5_tts  # noqa: F401
        except ImportError:
            return _missing(self.extra, "f5-tts")
        except Exception as e:  # broken install must degrade, not crash
            return False, f"f5-tts import failed: {type(e).__name__}: {e}"
        n = len(saved_voices())
        return True, f"ready ({n} saved voice{'s' if n != 1 else ''} in voices/)"

    def voices(self) -> list[str]:
        return saved_voices()

    def _model(self):
        global _MODEL
        if _MODEL is None:
            _patch_torchaudio_load()
            from f5_tts.api import F5TTS
            _MODEL = F5TTS(model="F5TTS_v1_Base", device=torch_device())
        return _MODEL

    @staticmethod
    def _resolve_ref(voice: str | None, ref: Path | None,
                     ref_text: str | None) -> tuple[Path, str]:
        if ref is None:
            if not voice:
                have = saved_voices()
                raise ValueError(
                    "f5 needs a reference: --voice <saved name> or --ref <wav>. "
                    + (f"Saved voices: {', '.join(have)}" if have
                       else "No saved voices yet - run `voicelab clone`."))
            ref = voice_dir(voice) / REF_WAV
            if not ref.exists():
                raise ValueError(f"no saved voice '{voice}' (looked for {ref})")
        ref = Path(ref)
        if not ref.exists():
            raise ValueError(f"reference audio not found: {ref}")
        if not ref_text:
            sidecar = ref.with_suffix(".txt")
            if sidecar.exists():
                ref_text = sidecar.read_text(encoding="utf-8").strip()
        return ref, ref_text or ""

    def synth(self, text: str, voice: str | None = None,
              ref: Path | None = None, speed: float = 1.0, **kw) -> Audio:
        ref, ref_text = self._resolve_ref(voice, ref, kw.get("ref_text"))
        if not text.strip():
            raise ValueError("nothing to say")
        tts = self._model()
        wav, sr, _spec = tts.infer(
            ref_file=str(ref),
            ref_text=ref_text,          # "" -> F5 transcribes with Whisper (slow)
            gen_text=text,
            speed=speed,
            nfe_step=int(kw.get("nfe_step", 32)),
            cfg_strength=float(kw.get("cfg_strength", 2.0)),
            seed=kw.get("seed"),
            show_info=lambda *_a, **_k: None,
            progress=None,
        )
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            raise RuntimeError("f5 produced no audio")
        if sr != SAMPLE_RATE:
            wav = resample(wav, sr, SAMPLE_RATE)
        return Audio(wav, SAMPLE_RATE)
