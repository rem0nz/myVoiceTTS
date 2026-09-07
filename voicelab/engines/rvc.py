"""RVC v2 inference via infer-rvc-python (fairseq-free).

RVC does not read text. It re-timbres existing speech, preserving timing and
prosody. Text-to-speech in your own voice is therefore a two-stage pipeline:

    text --[Kokoro/Piper/say]--> source speech --[RVC]--> your voice

Source choice matters: a source whose pitch range is near the target's needs
less shifting and produces fewer artifacts.

Backend note: this replaced rvc-python, which depended on fairseq 0.12.2 --
no arm64 wheel, builds only on Python <=3.10, and its pickled Dictionary
collides with torch>=2.6's weights_only default. infer-rvc-python loads
HuBERT through transformers instead, so none of that applies.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .base import VCEngine, Audio, _missing
from ..audio import resample, peak_normalize
from ..config import RVC_MODELS, SAMPLE_RATE, torch_device

# Accepted by infer-rvc-python. "rmvpe+" is rmvpe with extra refinement.
F0_METHODS = ("rmvpe", "rmvpe+", "harvest", "crepe", "pm")

# Loaded once, shared by every voice model.
BASE_DIR = RVC_MODELS / "_base"


class RVCEngine(VCEngine):
    name = "rvc"
    clones = True
    extra = "rvc"
    notes = ("Voice conversion, not TTS. Pair with a TTS engine. HuBERT is "
             "fetched from HF (r3gm/hubert_base) on first use; rmvpe.pt is "
             "reused from models/rvc/_base/ if present.")

    def __init__(self) -> None:
        self._bl = None
        self._configured: set[str] = set()

    # ---------------------------------------------------------------- checks
    def check(self) -> tuple[bool, str]:
        try:
            import infer_rvc_python  # noqa: F401
        except ImportError:
            return _missing(self.extra, "infer-rvc-python")
        n = len(self.voices())
        return True, f"ready ({n} model{'s' if n != 1 else ''})"

    def voices(self) -> list[str]:
        if not RVC_MODELS.exists():
            return []
        return sorted(d.name for d in RVC_MODELS.iterdir()
                      if d.is_dir() and not d.name.startswith("_")
                      and (d / "model.pth").exists())

    # ----------------------------------------------------------------- load
    def _loader(self):
        from infer_rvc_python import BaseLoader

        if self._bl is None:
            rmvpe = BASE_DIR / "rmvpe.pt"
            self._bl = BaseLoader(
                only_cpu=(torch_device() == "cpu"),
                rmvpe_path=str(rmvpe) if rmvpe.exists() else None,
                preload_models=False,   # load lazily on first conversion
            )
        return self._bl

    def _configure(self, model: str, pitch: int, f0_method: str,
                   index_rate: float, protect: float, rms_mix_rate: float,
                   filter_radius: int):
        dest = RVC_MODELS / model
        pth = dest / "model.pth"
        if not pth.exists():
            have = ", ".join(self.voices()) or "none installed"
            raise FileNotFoundError(f"RVC model '{model}' not found. Have: {have}")
        idx = dest / "model.index"

        bl = self._loader()
        # Re-applied every call: parameters change per request, and apply_conf
        # only rewrites a dict entry -- the heavy load is cached separately.
        bl.apply_conf(
            tag=model,
            file_model=str(pth),
            file_index=str(idx) if idx.exists() else "",
            pitch_algo=f0_method,
            pitch_lvl=int(pitch),
            index_influence=float(index_rate),
            respiration_median_filtering=int(filter_radius),
            envelope_ratio=float(rms_mix_rate),
            consonant_breath_protection=float(protect),
        )
        self._configured.add(model)
        return bl

    # -------------------------------------------------------------- convert
    def convert(self, audio: Audio, model: str, pitch: int = 0,
                f0_method: str = "rmvpe", index_rate: float = 0.66,
                protect: float = 0.33, rms_mix_rate: float = 0.25,
                filter_radius: int = 3, **kw) -> Audio:
        """pitch: semitones (+12 = up an octave). Match source to target's range.
        index_rate: 0..1 how strongly to pull toward the trained timbre.
        protect: 0..0.5 guards breath/consonants from over-conversion."""
        if f0_method not in F0_METHODS:
            raise ValueError(f"f0_method must be one of {F0_METHODS}")

        import soundfile as sf

        bl = self._configure(model, pitch, f0_method, index_rate,
                             protect, rms_mix_rate, filter_radius)

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.wav"
            sf.write(str(src), np.asarray(audio.wav, dtype=np.float32), audio.sr)

            results = bl(audio_files=[str(src)], tag_list=[model],
                         overwrite=False, type_output="wav",
                         show_progress=False)

            out = _pick_output(results, src)
            if out is None:
                raise RuntimeError(
                    f"RVC returned no output for '{model}'. Backend result: "
                    f"{results!r}. Check the server log for the traceback.")
            wav, sr = sf.read(str(out), dtype="float32", always_2d=True)

        wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
        wav = peak_normalize(resample(wav, sr, SAMPLE_RATE), 0.97)
        return Audio(wav, SAMPLE_RATE)


def _pick_output(results, src: Path) -> Path | None:
    """__call__ returns a list of produced paths; tolerate shape drift."""
    if not results:
        return None
    if isinstance(results, (str, Path)):
        cand = [Path(results)]
    else:
        cand = [Path(r) for r in results if isinstance(r, (str, Path))]
    cand = [p for p in cand if p.exists() and p.resolve() != src.resolve()]
    if not cand:
        # some versions write in place next to the input
        siblings = [p for p in src.parent.glob("*.wav")
                    if p.resolve() != src.resolve()]
        cand = siblings
    return max(cand, key=lambda p: p.stat().st_mtime) if cand else None
