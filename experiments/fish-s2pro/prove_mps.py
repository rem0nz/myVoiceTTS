"""Technical proof: Fish Speech S2-Pro running locally on Apple Silicon (MPS).

Loads the model on MPS, generates neutral and emotion-tagged speech, reports
timing. Nothing leaves the machine.
"""
import os, sys, time, wave
from pathlib import Path

os.environ["EINX_FILTER_TRACEBACK"] = "false"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pyrootutils, torch
pyrootutils.setup_root(Path(__file__).parent / "tools", indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest

CKPT = Path("checkpoints/s2-pro")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREC = torch.bfloat16
OUT = Path("proof_out"); OUT.mkdir(exist_ok=True)

print(f"== device={DEV}  precision={PREC}  torch={torch.__version__}")

t0 = time.time()
llama_queue = launch_thread_safe_queue(
    checkpoint_path=CKPT, device=DEV, precision=PREC, compile=False)
print(f"== llama loaded in {time.time()-t0:.1f}s")

t0 = time.time()
decoder = load_decoder_model(
    config_name="modded_dac_vq",
    checkpoint_path=CKPT / "codec.pth", device=DEV)
print(f"== decoder loaded in {time.time()-t0:.1f}s")

engine = TTSInferenceEngine(
    llama_queue=llama_queue, decoder_model=decoder,
    precision=PREC, compile=False)

CASES = {
    "neutral": "The deployment finished at three in the morning.",
    "expressive": "[whispers] The deployment finished at three in the morning. "
                  "[laugh] Nobody even noticed.",
    "angry": "[super angry] I told you twice already. The build is broken.",
}

ok = 0
for tag, text in CASES.items():
    t0 = time.time()
    audio, sr = None, None
    try:
        for res in engine.inference(ServeTTSRequest(
                text=text, format="wav", references=[],
                max_new_tokens=1024, temperature=0.8, top_p=0.8,
                normalize=True, streaming=False)):
            if getattr(res, "code", None) == "final" and res.audio is not None:
                sr, audio = res.audio
    except Exception as e:
        print(f"!! {tag}: {type(e).__name__}: {e}")
        continue
    dt = time.time() - t0
    if audio is None or len(audio) == 0:
        print(f"!! {tag}: produced no audio")
        continue
    p = OUT / f"{tag}.wav"
    import soundfile as sf
    sf.write(str(p), audio, sr)
    dur = len(audio) / sr
    print(f"== {tag}: {dur:.2f}s audio in {dt:.1f}s  (RTF {dt/dur:.2f})  -> {p}")
    ok += 1

print(f"\n== RESULT: {ok}/{len(CASES)} generations succeeded on {DEV}")
sys.exit(0 if ok else 1)
