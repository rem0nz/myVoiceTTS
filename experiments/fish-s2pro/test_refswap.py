"""Controlled test: does the REFERENCE drive delivery?
Same target sentence, NO emotion tags anywhere. Only the reference changes.
"""
import os, time
from pathlib import Path
os.environ["EINX_FILTER_TRACEBACK"]="false"; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
import pyrootutils, torch
pyrootutils.setup_root(Path(__file__).parent/"tools", indicator=".project-root", pythonpath=True)
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
import soundfile as sf

CKPT=Path("checkpoints/s2-pro"); DEV="mps"; PREC=torch.bfloat16
OUT=Path("refswap_out"); OUT.mkdir(exist_ok=True)
SPOKEN="I cannot believe the build broke again right before the demo."
REFS={
 "A_flat_long": ("ref/me_clean.wav", Path("ref/me_clean.txt").read_text().strip()),
 "B_midsent":   ("sweep_out/11_midsentence.wav", SPOKEN),
 "C_excited":   ("sweep_out/09_excited_laugh.wav", SPOKEN),
 "D_screaming": ("sweep_out/03_screaming.wav", SPOKEN),
}
TARGET="We shipped it on Friday and the whole team stayed up to watch the numbers come in."

q=launch_thread_safe_queue(checkpoint_path=CKPT,device=DEV,precision=PREC,compile=False)
dec=load_decoder_model(config_name="modded_dac_vq",checkpoint_path=CKPT/"codec.pth",device=DEV)
eng=TTSInferenceEngine(llama_queue=q,decoder_model=dec,precision=PREC,compile=False)
print("== loaded",flush=True)
for name,(wav,txt) in REFS.items():
    p=Path(wav)
    if not p.exists(): print(f"!! missing {wav}"); continue
    ref=ServeReferenceAudio(audio=p.read_bytes(),text=txt)
    t0=time.time(); audio=sr=None
    try:
        for res in eng.inference(ServeTTSRequest(text=TARGET,format="wav",references=[ref],
                max_new_tokens=1024,temperature=0.8,top_p=0.8,normalize=True,
                streaming=False,seed=4242)):
            if getattr(res,"code",None)=="final" and res.audio is not None: sr,audio=res.audio
    except Exception as e:
        print(f"!! {name}: {type(e).__name__}: {e}",flush=True); continue
    if audio is None or len(audio)==0: print(f"!! {name}: no audio",flush=True); continue
    sf.write(str(OUT/f"{name}.wav"),audio,sr)
    print(f"== {name:12s} ref={Path(wav).name:24s} {len(audio)/sr:5.2f}s in {time.time()-t0:5.1f}s",flush=True)
print("== RESULT: done",flush=True)
