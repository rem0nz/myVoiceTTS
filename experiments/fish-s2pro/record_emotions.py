"""Record one short reference per emotion. S2-Pro has a single reference slot
that carries timbre AND delivery, so the only way to move delivery is to give
it a reference already in that delivery.

Short on purpose (8-12 s): a long reference over-anchors and flattens output.
"""
import sys, time, threading
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf

SR, MAXS = 44100, 20.0
OUT = Path("ref/emo"); OUT.mkdir(parents=True, exist_ok=True)

TAKES = {
 "angry":   ("ANGRY - genuinely irritated, clipped, forceful",
             "I told you twice already. This is completely unacceptable, and I am done explaining it."),
 "excited": ("EXCITED - fast, bright, energised",
             "You will not believe what just happened. This is the best news I have had all year!"),
 "whisper": ("WHISPER - actually whisper, quiet and breathy, close to the mic",
             "Keep your voice down. They are still in the next room, and I do not want them to hear."),
 "sad":     ("SAD - slow, heavy, quiet, falling intonation",
             "I really thought it would work out this time. I suppose there is nothing left to say."),
}
want = [a for a in sys.argv[1:] if not a.startswith("--")] or list(TAKES)
dev = None
for a in sys.argv[1:]:
    if a.startswith("--mic="):
        q=a.split("=",1)[1].lower()
        for i,d in enumerate(sd.query_devices()):
            if d["max_input_channels"]>0 and q in d["name"].lower(): dev=i; print(f"mic [{i}] {d['name']}"); break

for key in want:
    if key not in TAKES: print(f"skip unknown '{key}'"); continue
    mood, line = TAKES[key]
    print("\n"+"="*74); print(f"  [{key}]  {mood}"); print(f"\n  \"{line}\"\n"); print("="*74)
    print("  Perform it - do not read it politely. Exaggerate more than feels natural.")
    input(f"  Enter to record '{key}'... ")
    frames=[]; st=sd.InputStream(device=dev,channels=1,samplerate=SR,dtype="float32")
    st.start(); t0=time.time(); stop=threading.Event()
    threading.Thread(target=lambda:(input(),stop.set()),daemon=True).start()
    print("  RECORDING - Enter when done.")
    while not stop.is_set() and time.time()-t0<MAXS:
        d_,_=st.read(2048); frames.append(d_.copy())
    st.stop(); st.close()
    w=np.concatenate(frames).reshape(-1) if frames else np.zeros(0,np.float32)
    dur=len(w)/SR; peak=float(np.abs(w).max()) if w.size else 0.
    if peak<1e-6: print("  SILENT - not saved."); continue
    if dur<4:     print(f"  too short ({dur:.1f}s) - not saved."); continue
    # trim silence, normalise (whisper kept relatively quieter on purpose)
    win=int(0.05*SR); e=np.sqrt(np.convolve(w**2,np.ones(win)/win,'same'))
    edb=20*np.log10(np.maximum(e,1e-9)); idx=np.where(edb>np.percentile(edb,10)+8)[0]
    a,b=max(0,idx[0]-int(0.1*SR)),min(len(w),idx[-1]+int(0.15*SR)); w=w[a:b]
    w=w*(0.70/max(np.abs(w).max(),1e-9))
    sf.write(str(OUT/f"{key}.wav"),w,SR); (OUT/f"{key}.txt").write_text(line+"\n",encoding="utf-8")
    print(f"  saved ref/emo/{key}.wav  ({len(w)/SR:.1f}s, peak {np.abs(w).max():.2f})")
print("\nDone. Tell Claude: \"emotion refs recorded\"")
