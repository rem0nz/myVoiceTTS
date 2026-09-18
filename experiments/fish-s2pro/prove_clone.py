"""Proof: cloned voice + emotion tags, together, on MPS.

Same reference clip throughout, so any difference in delivery comes from the
inline tags alone. That is the claim being tested: timbre from `references`,
expression from tags in `text`, composed in one pass.
"""
import os, sys, time
from pathlib import Path

os.environ["EINX_FILTER_TRACEBACK"] = "false"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pyrootutils, torch
pyrootutils.setup_root(Path(__file__).parent / "tools", indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

CKPT = Path("checkpoints/s2-pro")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREC = torch.bfloat16
OUT = Path("clone_out"); OUT.mkdir(exist_ok=True)

REF_WAV, REF_TXT = Path("ref/daniel.wav"), Path("ref/daniel.txt")
ref = ServeReferenceAudio(audio=REF_WAV.read_bytes(), text=REF_TXT.read_text().strip())
print(f"== device={DEV} ref={REF_WAV} ({len(ref.audio)/1024:.0f} KB)", flush=True)

t0 = time.time()
llama_queue = launch_thread_safe_queue(checkpoint_path=CKPT, device=DEV,
                                       precision=PREC, compile=False)
decoder = load_decoder_model(config_name="modded_dac_vq",
                             checkpoint_path=CKPT / "codec.pth", device=DEV)
engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder,
                            precision=PREC, compile=False)
print(f"== models loaded in {time.time()-t0:.1f}s", flush=True)

LINE = "I cannot believe the build broke again right before the demo."
CASES = {
    "clone_neutral": LINE,
    "clone_angry":   f"[super angry] {LINE}",
    "clone_whisper": f"[whispers] {LINE}",
    "clone_sad":     f"[very sad] {LINE}",
}

import soundfile as sf
ok = 0
for tag, text in CASES.items():
    t0 = time.time()
    audio = sr = None
    try:
        for res in engine.inference(ServeTTSRequest(
                text=text, format="wav", references=[ref],
                max_new_tokens=1024, temperature=0.8, top_p=0.8,
                normalize=True, streaming=False, seed=1234)):
            if getattr(res, "code", None) == "final" and res.audio is not None:
                sr, audio = res.audio
    except Exception as e:
        print(f"!! {tag}: {type(e).__name__}: {e}", flush=True)
        continue
    if audio is None or len(audio) == 0:
        print(f"!! {tag}: no audio", flush=True); continue
    dt = time.time() - t0
    p = OUT / f"{tag}.wav"
    sf.write(str(p), audio, sr)
    dur = len(audio) / sr
    print(f"== {tag:15s} {dur:5.2f}s audio in {dt:6.1f}s  RTF {dt/dur:6.2f}  -> {p}", flush=True)
    ok += 1

print(f"\n== RESULT: {ok}/{len(CASES)} cloned generations succeeded on {DEV}", flush=True)
sys.exit(0 if ok == len(CASES) else 1)
