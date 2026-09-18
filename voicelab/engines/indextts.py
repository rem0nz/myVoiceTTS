"""IndexTTS-2. Zero-shot cloning with emotion decoupled from timbre.

The reason this engine exists: F5 clones *who* a voice is but has no emotion
conditioning at all, so everything it says lands flat. IndexTTS-2 splits the
two conditioning paths, which is the whole point:

    spk_audio_prompt   ->  WHO   (timbre, accent)
    emo_*              ->  HOW   (delivery)

...and the emotion prompt may come from a *different speaker* than the timbre
prompt. That is what makes a library of expression clips useful: record one
reference of the target voice, then steer it with angry/warm/tired clips read
by anyone at all.

A saved voice is a folder under voices/<name>/ holding:
    ref.wav             timbre reference (mono, any rate; 5-10 s)
    ref.txt             optional; unused here but shared with the f5 engine
    emotions/<tag>.wav  optional expression clips, referenced as --emotion <tag>

Three ways to drive delivery, highest precedence first:

    emotion="angry"                  a clip in emotions/, or a path to a wav
    emotion="sad:0.7,surprised:0.3"  weights over the 8 named emotions
    emo_text="tired and sarcastic"   free text, mapped by the bundled Qwen3

Nothing is sent anywhere; all three run locally.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .base import TTSEngine, Audio, _missing
from ..audio import resample
from ..config import MODELS, SAMPLE_RATE, VOICES, torch_device

REF_WAV = "ref.wav"
EMO_DIR = "emotions"
CKPT = MODELS / "indextts2"

# Fixed order required by IndexTTS2.infer(emo_vector=...). Do not reorder.
EMOTIONS = ("happy", "angry", "sad", "afraid",
            "disgusted", "melancholic", "surprised", "calm")

# One model per process; the server builds a fresh engine per request and this
# is several GB of weights. Same reasoning as f5.py.
_MODEL = None


def voice_dir(name: str) -> Path:
    return VOICES / name


def saved_voices() -> list[str]:
    if not VOICES.exists():
        return []
    return sorted(d.name for d in VOICES.iterdir()
                  if d.is_dir() and (d / REF_WAV).exists())


def emotion_clips(voice: str) -> list[str]:
    """Expression clips saved for a voice, by tag."""
    d = voice_dir(voice) / EMO_DIR
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.wav"))


def parse_emo_vector(spec: str) -> list[float]:
    """'sad:0.7,surprised:0.3' or bare 'sad' -> the 8-float vector.

    Raises ValueError on an unknown name so a typo is a clear error rather
    than silently synthesising a neutral read.
    """
    vec = [0.0] * len(EMOTIONS)
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        name, _, weight = part.partition(":")
        name = name.strip().lower()
        if name not in EMOTIONS:
            raise ValueError(
                f"unknown emotion '{name}'. Have: {', '.join(EMOTIONS)}")
        try:
            w = float(weight) if weight.strip() else 1.0
        except ValueError:
            raise ValueError(f"bad weight in '{part}' - expected a number")
        vec[EMOTIONS.index(name)] = w
    if not any(vec):
        raise ValueError(f"no emotion set from '{spec}'")
    return vec


class IndexTTS2Engine(TTSEngine):
    name = "indextts"
    clones = True
    extra = "indextts"
    notes = ("Zero-shot cloning with emotion decoupled from timbre. Apache-2.0. "
             "Steer with an expression clip, 8 emotion weights, or plain text. "
             "~5.9 GB checkpoints + 1.2 GB for text-driven emotion.")

    def check(self) -> tuple[bool, str]:
        try:
            import indextts  # noqa: F401
        except ImportError:
            return _missing(self.extra, "indextts")
        except Exception as e:      # a broken install degrades, never crashes
            return False, f"indextts import failed: {type(e).__name__}: {e}"
        if not (CKPT / "config.yaml").exists():
            return False, (f"weights missing - run "
                           f"`python -m voicelab.setup_assets --indextts` "
                           f"(~5.9 GB into {CKPT})")
        n = len(saved_voices())
        return True, f"ready ({n} saved voice{'s' if n != 1 else ''} in voices/)"

    def voices(self) -> list[str]:
        return saved_voices()

    def _model(self):
        global _MODEL
        if _MODEL is None:
            from indextts.infer_v2 import IndexTTS2
            # fp16 is unreliable on MPS and use_cuda_kernel/deepspeed are both
            # CUDA-only paths; the plain torch path is what runs on Apple GPUs.
            dev = torch_device()
            _MODEL = IndexTTS2(
                cfg_path=str(CKPT / "config.yaml"),
                model_dir=str(CKPT),
                use_fp16=False,
                use_cuda_kernel=False,
                use_deepspeed=False,
                device=dev,
            )
        return _MODEL

    @staticmethod
    def _resolve_ref(voice: str | None, ref: Path | None) -> Path:
        if ref is None:
            if not voice:
                have = saved_voices()
                raise ValueError(
                    "indextts needs a reference: --voice <saved name> or "
                    "--ref <wav>. "
                    + (f"Saved voices: {', '.join(have)}" if have
                       else "No saved voices yet - run `voicelab clone`."))
            ref = voice_dir(voice) / REF_WAV
            if not ref.exists():
                raise ValueError(f"no saved voice '{voice}' (looked for {ref})")
        ref = Path(ref)
        if not ref.exists():
            raise ValueError(f"reference audio not found: {ref}")
        return ref

    @staticmethod
    def _resolve_emotion(voice: str | None, emotion: str | None) -> dict:
        """-> infer() kwargs for the emotion path. Clip beats vector."""
        if not emotion:
            return {}
        # A tag saved under voices/<name>/emotions/, then a literal path.
        if voice:
            clip = voice_dir(voice) / EMO_DIR / f"{emotion}.wav"
            if clip.exists():
                return {"emo_audio_prompt": str(clip)}
        p = Path(emotion).expanduser()
        if p.suffix.lower() in (".wav", ".flac", ".mp3"):
            if not p.exists():
                raise ValueError(f"emotion clip not found: {p}")
            return {"emo_audio_prompt": str(p)}
        return {"emo_vector": parse_emo_vector(emotion)}

    def synth(self, text: str, voice: str | None = None,
              ref: Path | None = None, speed: float = 1.0, **kw) -> Audio:
        if not text.strip():
            raise ValueError("nothing to say")
        spk = self._resolve_ref(voice, ref)

        emo_text = kw.get("emo_text")
        emo = self._resolve_emotion(voice, kw.get("emotion"))
        if not emo and emo_text:
            emo = {"use_emo_text": True, "emo_text": emo_text}

        # Upstream's default is 1.0 (full strength), which over-drives a cloned
        # voice; their own guidance is <=0.8, and <=0.6 for the text path.
        alpha = kw.get("emo_alpha")
        if alpha is None:
            alpha = 0.6 if emo.get("use_emo_text") else 0.8

        tts = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gen.wav"
            tts.infer(
                spk_audio_prompt=str(spk),
                text=text,
                output_path=str(out),
                emo_alpha=float(alpha),
                use_random=False,      # random sampling costs cloning fidelity
                verbose=False,
                **emo,
            )
            if not out.exists():
                raise RuntimeError("indextts produced no audio")
            import soundfile as sf
            wav, sr = sf.read(str(out), dtype="float32", always_2d=False)

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            raise RuntimeError("indextts produced an empty file")
        if sr != SAMPLE_RATE:
            wav = resample(wav, sr, SAMPLE_RATE)
        return Audio(wav, SAMPLE_RATE)
