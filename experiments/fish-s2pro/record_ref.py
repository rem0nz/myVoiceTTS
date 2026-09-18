"""Record a 44.1 kHz reference clip for Fish S2-Pro cloning.

PortAudio via sounddevice, NOT ffmpeg avfoundation - that mislabels 24-bit USB
receivers and writes files of exact zeros (see voicelab CLAUDE.md).
"""
import sys, time
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf

SR, TARGET, MAXS = 44100, 22.0, 35.0
SCRIPT = (
    "The old bridge past the harbour was rebuilt last summer, though hardly "
    "anyone noticed the change. I usually walk that way on Thursday evenings, "
    "when the traffic thins out and most of the shops have closed. It takes "
    "about twenty minutes each way, unless I stop to watch the children "
    "chasing pigeons near the fountain."
)
OUT = Path("ref"); OUT.mkdir(exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "me"

if "--list" in sys.argv:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch)")
    sys.exit(0)

dev = None
for a in sys.argv[1:]:
    if a.startswith("--mic="):
        q = a.split("=", 1)[1].lower()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and q in d["name"].lower():
                dev = i; print(f"using mic [{i}] {d['name']}"); break

print("\n" + "=" * 72)
print("READ THIS ALOUD, VERBATIM, IN A NEUTRAL EVEN VOICE:\n")
print("  " + SCRIPT.replace(". ", ".\n  "))
print("\n" + "=" * 72)
print(f"\nNeutral delivery matters: the clip sets your voice AND its emotional")
print(f"baseline. Read it flat - the emotion tags supply expression later.\n")
input("Press Enter to START recording... ")

frames = []
stream = sd.InputStream(device=dev, channels=1, samplerate=SR, dtype="float32")
stream.start(); t0 = time.time()
print("RECORDING - press Enter when you finish the last word.")
import threading
stop = threading.Event()
threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
while not stop.is_set() and time.time() - t0 < MAXS:
    data, _ = stream.read(2048); frames.append(data.copy())
stream.stop(); stream.close()

wav = np.concatenate(frames).reshape(-1) if frames else np.zeros(0, np.float32)
dur, peak = len(wav) / SR, float(np.abs(wav).max()) if wav.size else 0.0
rms = float(np.sqrt((wav ** 2).mean())) if wav.size else 0.0
db = 20 * np.log10(max(rms, 1e-9))
print(f"\n  duration {dur:.1f}s   peak {peak:.3f}   rms {db:.1f} dB")

bad = []
if peak < 1e-6:          bad.append("SILENT - mic off, muted, or permission denied")
elif peak > 0.99:        bad.append("CLIPPING - move back or lower input gain")
if dur < 10:             bad.append(f"too short ({dur:.1f}s) - S2-Pro wants 10-30s")
if db < -35 and peak>1e-6: bad.append(f"very quiet ({db:.0f} dB) - move closer")
if bad:
    print("\n  PROBLEMS:"); [print(f"    - {b}") for b in bad]
    print("\n  Not saved. Run again.")
    sys.exit(1)

sf.write(str(OUT / f"{name}.wav"), wav, SR)
(OUT / f"{name}.txt").write_text(SCRIPT + "\n", encoding="utf-8")
print(f"\n  saved ref/{name}.wav  +  ref/{name}.txt")
print(f"  tell Claude: \"recorded, use {name}\"")
