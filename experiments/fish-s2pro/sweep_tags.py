"""Canonical-tag sweep. Same reference, same seed, same sentence.
Tags taken verbatim from the repo's documented vocabulary (docs/README.zh.md:117).
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
import soundfile as sf

CKPT = Path("checkpoints/s2-pro"); DEV = "mps"; PREC = torch.bfloat16
OUT = Path("sweep_out"); OUT.mkdir(exist_ok=True)
ref = ServeReferenceAudio(audio=Path("ref/me_clean.wav").read_bytes(),
                          text=Path("ref/me_clean.txt").read_text().strip())
llama_queue = launch_thread_safe_queue(checkpoint_path=CKPT, device=DEV, precision=PREC, compile=False)
decoder = load_decoder_model(config_name="modded_dac_vq", checkpoint_path=CKPT/"codec.pth", device=DEV)
engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder, precision=PREC, compile=False)
print("== loaded", flush=True)

L = "I cannot believe the build broke again right before the demo."
CASES = {
  "00_neutral":      L,
  "01_angry":        f"[angry] {L}",
  "02_shouting":     f"[shouting] {L}",
  "03_screaming":    f"[screaming] {L}",
  "04_whisper":      f"[whisper] {L}",
  "05_whisper_low":  f"[whisper] [low voice] [low volume] {L}",
  "06_sad":          f"[sad] {L}",
  "07_sigh_sad":     f"[sigh] [sad] {L}",
  "08_excited":      f"[excited] {L}",
  "09_excited_laugh":f"[excited tone] {L} [laughing]",
  "10_shocked":      f"[shocked] {L}",
  "11_midsentence":  "I cannot believe [pause] the build broke again [emphasis] right before the demo.",
  "12_accent":       f"[with strong accent] {L}",
}
ok=0
for tag, text in CASES.items():
    t0=time.time(); audio=sr=None
    try:
        for res in engine.inference(ServeTTSRequest(
                text=text, format="wav", references=[ref], max_new_tokens=1024,
                temperature=0.8, top_p=0.8, normalize=True, streaming=False, seed=1234)):
            if getattr(res,"code",None)=="final" and res.audio is not None: sr,audio=res.audio
    except Exception as e:
        print(f"!! {tag}: {type(e).__name__}: {e}", flush=True); continue
    if audio is None or len(audio)==0: print(f"!! {tag}: no audio", flush=True); continue
    sf.write(str(OUT/f"{tag}.wav"), audio, sr); ok+=1
    print(f"== {tag:18s} {len(audio)/sr:5.2f}s in {time.time()-t0:5.1f}s", flush=True)
print(f"\n== RESULT: {ok}/{len(CASES)}", flush=True)
