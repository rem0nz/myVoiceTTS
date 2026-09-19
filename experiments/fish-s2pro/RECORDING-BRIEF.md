# Voice recording brief

**You are being asked to record ~5 short voice clips. That is the deliverable.**
Running the model is optional and depends on your hardware — see the last
section. Works for a human developer or an AI agent; ~10 minutes either way.

---

## Why this exists (30 seconds of context)

We're testing **Fish Audio S2-Pro**, which clones a voice from a short
reference clip and applies emotion through inline tags like `[whisper]`,
`[angry]`, `[excited]`.

It half-works, and we know why. S2-Pro has exactly **one** reference slot
(`ServeReferenceAudio = {audio, text}` — there is no separate emotion-reference
parameter). That single slot carries both *who the voice is* and *how it's
delivered*. Feed it a neutral read and you get a neutral prosodic prior that
the emotion tags then have to fight — and mostly lose to. Measured: structural
tags like `[pause]`/`[emphasis]` land clearly, but `[angry]` moved dynamic
range only +1.3 dB and `[excited]` actually went **backwards** (−0.8 dB).

The untested hypothesis is that a **separate reference per emotion**, with the
speaker genuinely performing it, unlocks real expressiveness. These recordings
are that experiment.

---

## 1. Setup (small — no model download)

```bash
git clone -b rem0nz/snaggletooth https://github.com/rem0nz/myVoiceTTS.git
cd myVoiceTTS/experiments/fish-s2pro
pip install numpy soundfile sounddevice
```

Platform extras if the install complains:

| OS | command |
|---|---|
| macOS | `brew install portaudio` |
| Debian/Ubuntu | `sudo apt install portaudio19-dev libsndfile1` |

Then work somewhere the scripts can write `./ref/`:

```bash
mkdir -p ~/fishwork && cp *.py ~/fishwork/ && cd ~/fishwork
```

## 2. Record

```bash
python record_ref.py --list          # list microphones
python record_ref.py me              # neutral baseline, ~20 s
python record_emotions.py            # angry / excited / whisper / sad, ~10 s each
```

Append `--mic=<substring>` to either script if the default input is wrong
(e.g. `--mic=Yeti`). To redo just one emotion:
`python record_emotions.py whisper`

Outputs:

```
ref/me.wav          ref/me.txt
ref/emo/angry.wav   ref/emo/angry.txt
ref/emo/excited.wav ref/emo/excited.txt
ref/emo/whisper.wav ref/emo/whisper.txt
ref/emo/sad.wav     ref/emo/sad.txt
```

Each script prints the exact line to read and validates the take. It **refuses
to save** anything silent, clipping, too short or too quiet. If a take is
rejected, fix the cause and rerun — don't work around the check.

## 3. The one thing that actually matters

**On the emotion takes: perform them. Do not read them politely.**

Whatever restraint is in the reference becomes the hard **ceiling** on the
generated output. A politely-read "angry" line produces polite speech and the
experiment tells us nothing. Overdo it past the point where it feels silly.
For the whisper take, genuinely whisper and move closer to the mic.

The neutral take (`record_ref.py`) is the opposite — read that one **flat and
even**. It's the control.

## 4. Recording quality

The model clones room tone along with the voice, so:

- Quiet room — no music, TV, fan, or air conditioning.
- ~20 cm from the mic, consistent distance throughout.
- Read the neutral script **verbatim**. The transcript is saved alongside and
  the model aligns audio to text; improvising degrades the clone.
- Short is correct. 10–30 s neutral, 8–12 s per emotion. A 34 s first attempt
  over-anchored the model and flattened everything.
- **macOS: do not record via ffmpeg's `avfoundation` input.** It mislabels
  24-bit USB receivers and writes files of exact zeros. The scripts use
  PortAudio specifically to avoid this.

## 5. Sending the files back

Send `ref/me.wav`, `ref/me.txt`, and everything in `ref/emo/`.

> ### Do not commit or push the audio.
>
> This repo is **public**. A voice clip plus its exact transcript is a
> ready-made cloning kit for whoever's voice is on it, and publishing is
> irreversible — it can be mirrored and cloned by anyone.
>
> `.gitignore` already blocks audio under `experiments/`. Please don't
> override it. Transfer privately: direct upload, encrypted share, AirDrop.

Model weights don't belong in git either — ~10 GB, over GitHub's 100 MB
per-file limit, and non-commercially licensed.

## 6. Optional — run the model yourself

Only worth it if you have **≥16 GB RAM and ≥15 GB free disk**. Apple Silicon
(MPS) and NVIDIA both work; MPS is verified on an M5 Pro. Don't attempt it
CPU-only or on a low-RAM machine — it will thrash or OOM, and recordings alone
are genuinely useful because the requester has a machine that runs it.

```bash
git clone https://github.com/fishaudio/fish-speech.git ~/fishlab
cd ~/fishlab && uv venv --python 3.12
uv pip install -e .
uv pip install 'tensorboard<2.20' sounddevice
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro   # ~10 GB
cp -r ~/fishwork/*.py ~/fishwork/ref .
.venv/bin/python -u sweep_tags.py
```

**Two traps that will stop you cold** (both documented in `README.md`):

1. `pyaudio` build fails with `portaudio.h not found` → install portaudio first.
2. `import audiotools` raises a protobuf `VersionError`. `descript-audiotools`
   pins `protobuf<3.20`; `tensorboard 2.21` needs `>=6.31.1`. Irreconcilable.
   Fix by **downgrading tensorboard** to `<2.20`. Do **not** try upgrading
   protobuf — the audiotools pin wins and you get the identical error.

Expect ~60 s model load, then RTF ~2.8–2.9 (roughly 3× slower than realtime).
Output is 44.1 kHz.

Use the **documented** tag vocabulary — invented variants are read as literal
text and ignored (`[whispers]` did nothing; the correct `[whisper]` moved RMS
−11.4%). Full list and measured results are in `README.md`.

## 7. Report back

- Whether you recorded only, or also ran the model (and your machine specs).
- Each take's duration and peak, and whether the validator passed.
- If you ran the model: **did the emotional references produce audibly more
  expressive output than the neutral one?** That single question is the entire
  point of this exercise.

---

*Licence note: Fish S2-Pro is under the FISH AUDIO RESEARCH LICENSE —
research and non-commercial use free, commercial use requires a separate
licence. It is open-weight, not open-source; third-party claims of MIT are
incorrect.*
