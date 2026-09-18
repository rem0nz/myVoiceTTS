# CLAUDE.md — voicelab

Read this before changing anything. Several decisions here look arbitrary and
are not; the reasons cost real debugging time to find.

## What this is

A local voice workbench: take an RVC voice model, register it, and speak text
through it. Web UI + CLI. Nothing leaves the machine — no API calls, no cloud.

Built for Apple Silicon (M-series, MPS). Should work elsewhere but is only
verified on macOS arm64.

## The one thing to understand first

**RVC is voice *conversion*, not text-to-speech.** It re-timbres existing
speech; it cannot read text. Everything about the architecture follows from
this. TTS in a target voice is two stages:

```
text ──[Kokoro / Piper / macOS say]──> source speech ──[RVC + model]──> target voice
        stage 1: TTS engine                             stage 2: conversion
```

If someone asks "why can't I just give RVC text", that's why. The user's
trained model plugs into stage 2 only.

## Running it

```bash
cd ~/voicelab
.venv/bin/python -m uvicorn voicelab.server:app --host 127.0.0.1 --port 8791
# → http://127.0.0.1:8791
```

CLI:

```bash
.venv/bin/python -m voicelab.cli doctor          # engine + system status, start here
.venv/bin/python -m voicelab.cli voices          # list source voices
.venv/bin/python -m voicelab.cli say "hello" --engine kokoro --voice bm_george \
    --rvc <model> --pitch 0 --play
.venv/bin/python -m voicelab.cli convert in.wav --model <model> -o out.wav
.venv/bin/python -m voicelab.cli dataset-prep ~/recordings --name myvoice
.venv/bin/python -m voicelab.cli clone --name me --mic "Wireless Mic Rx"   # record, then type
```

`doctor` is the first thing to run when anything looks wrong. It reports each
engine's availability with the exact install command, plus torch device, free
disk, ffmpeg/espeak-ng, and model counts.

## Fresh clone setup

A clone does **not** run as-is — model weights and the venv are deliberately
not in git:

```bash
uv venv --python 3.11
uv pip install -e ".[web]"
./install-heavy.sh                              # Kokoro + RVC, ~4.3 GB
.venv/bin/python -m voicelab.setup_assets       # rmvpe.pt, ~180 MB
```

HuBERT is *not* fetched by that script — `infer-rvc-python` pulls it through
`transformers` (`r3gm/hubert_base`) on the first conversion.

## Layout

```
voicelab/
├── __init__.py      OpenMP env vars — READ THE WARNING BELOW
├── config.py        paths, torch device detection, disk checks
├── audio.py         io/resample/trim/silence-split (numpy + soundfile only, no torch)
├── rvcpack.py       RVC .zip inspect / validate / install
├── presets.py       named settings, atomic JSON store
├── outputs.py       generated-audio library, zip bundling, path hardening
├── dataset.py       raw recordings → training clips
├── clone.py         interactive record-a-reference → type → speak loop (F5)
├── server.py        FastAPI
├── cli.py           typer CLI
├── setup_assets.py  downloads rmvpe.pt
├── web/index.html   entire frontend, single file, no build step
└── engines/
    ├── base.py      Engine / TTSEngine / VCEngine contracts
    ├── system.py    macOS `say`   — no install, the sanity baseline
    ├── kokoro.py    Kokoro-82M    [kokoro]
    ├── piper.py     Piper ONNX    [piper]
    ├── f5.py        F5-TTS zero-shot cloning from voices/<name>/ref.wav [f5]
    └── rvc.py       RVC v2        [rvc]
```

Engines are lazily imported and independently optional. A missing engine must
degrade to a status message from `check()`, never an import crash. Preserve
that when adding one: `check()` returns `(bool, message)` and **must not
raise**.

## Traps — do not undo these

### 1. OpenMP env vars in `__init__.py`

`torch`, `faiss` and `sklearn` each vendor their own `libomp.dylib`. macOS
loads all three into one process, their OpenMP runtimes share thread state and
collide, and the interpreter dies with `SIGSEGV` in
`__kmp_suspend_initialize_thread`. No Python traceback — the process just
vanishes and the web UI reports `TypeError: Load failed` (a dropped socket).
RVC conversion touches all three libraries at once, so it hits this every time.

The env block at the top of `voicelab/__init__.py` fixes it. **It must run
before torch/faiss/sklearn import**, which is why it lives there and not in
`config.py` or `server.py`. Do not move it, do not "tidy" it into a function
that gets called later.

### 2. `infer-rvc-python`, not `rvc-python`

The obvious package, `rvc-python`, is a trap on this platform:

- depends on `fairseq==0.12.2` — no macOS-arm64 wheel, builds only on ≤3.10
- pins `numpy<=1.23.5`, irreconcilable with torch and Kokoro (its runtime code
  is actually fine on numpy 2.x — the pin is stale)
- fairseq's pickled `Dictionary` in `hubert_base.pt` is refused by torch
  ≥2.6's `weights_only=True` default, and `rvc-python` surfaces that failure
  as `AttributeError: 'tuple' object has no attribute 'dtype'` because it
  passes `vc_single`'s failure tuple straight to `scipy.io.wavfile.write`

`infer-rvc-python` loads HuBERT via `transformers` and has none of this. Don't
"upgrade" back.

Also: `rvc-inferpy`'s PyPI metadata advertises `fairseq2`, but its source does
`from fairseq import checkpoint_utils` (unrelated legacy package). It imports
fine and only fails at conversion time. Avoid.

### 3. `setuptools<81` is pinned on purpose

The resolved `librosa` still does `from pkg_resources import resource_filename`.
setuptools dropped `pkg_resources` in 81. Unpinning breaks imports.

### 4. `weights_only=True` on uploaded checkpoints

`rvcpack.read_checkpoint` loads user-supplied `.pth` files with
`weights_only=True`. A `.pth` is a pickle; without that flag an uploaded
checkpoint executes arbitrary code. Never relax it for user input.

### 5. HuBERT format

The old fairseq-format `hubert_base.pt` is **not** loadable by the current
backend (`from_pretrained` needs transformers format). If you see code trying
to reuse it, that's stale. `rmvpe.pt` *is* shared and reusable.

## Repo policy on voice models

This tool clones voices. **No trained voice model is published from this
repo** — `.gitignore` excludes `models/rvc/` entirely. That is deliberate, not
an oversight:

- a model of a real, identifiable person is not ours to distribute
- a public model of *your own* voice lets anyone synthesise speech as you,
  and publishing is not reversible — it can be cloned and mirrored

Users bring their own model and load it through the web UI. Don't add models
to the repo, and don't "helpfully" remove that ignore rule.

## Git

`.gitattributes` routes `*.pth`, `*.index`, `*.pt` through **Git LFS** if any
ever are committed — they're 50–75 MB binaries and plain git would grow the
packfile permanently per revision. Today none are: `models/` is ignored
wholesale (see above). Leave it that way.

Never commit: `models/rvc/_base/` (`rmvpe.pt` is 173 MB, over GitHub's hard
100 MB per-file cap — the push is *rejected*, not warned), `.venv/`, `out/`.

## Verifying a change

Cheapest → most thorough. Run at least the first three after touching
anything:

```bash
# 1. everything imports and engines report honestly
.venv/bin/python -m voicelab.cli doctor

# 2. the libomp fix still holds (this segfaults if trap #1 is broken)
.venv/bin/python -c "import voicelab, torch, faiss; \
  from voicelab.engines.rvc import RVCEngine; \
  print(RVCEngine()._loader().config.device)"

# 3. TTS end-to-end through the API
curl -s -X POST http://127.0.0.1:8791/api/synth \
  -F "text=test" -F "engine=kokoro" -F "voice=bm_george"

# 4. output-library path hardening (all must be rejected)
.venv/bin/python -c "
from voicelab import outputs as O
for bad in ['../etc/passwd','server.log','x/y.wav','..']:
    try: O.resolve(bad); print('FAIL', bad)
    except Exception as e: print('ok', bad, type(e).__name__)"
```

Stage 2 (actual RVC conversion) can only be verified with a real model
installed. If none is present, say the conversion path is unverified rather
than implying it was tested.

## Gotchas that waste time

- **Don't record with ffmpeg's avfoundation input.** It mislabels 24-bit USB
  receivers (e.g. `Wireless Mic Rx`) as `pcm_s24le`, logs "Invalid PCM packet"
  per frame and writes a file of exact zeros. `clone.py` uses PortAudio via
  `sounddevice` for that reason. Exact digital silence from a mic that opens
  fine is the transmitter being off, not a permission problem — permission
  denial on macOS also yields zeros, so compare against the built-in mic.
- **F5 needs the reference transcript.** `voices/<name>/ref.txt` must hold
  the exact words in `ref.wav`. With an empty transcript F5 runs Whisper
  (1.5 GB download, slow on 8 GB RAM). `clone` avoids that by having the user
  read a fixed sentence.

- **`say`-engine voice names** contain a locale in parens in `voices()` output
  (`Daniel  (en_GB)`); the CLI/API want just `Daniel`. The frontend splits on
  the double space.
- **rich eats `[extra]` as markup.** Any engine status message containing
  `.[kokoro]` must be passed through `rich.markup.escape` or it prints the
  wrong install command.
- **`urlretrieve` progress hooks** must be TTY-guarded — a `\r` spinner into a
  pipe emits megabytes of output.
- **The audio player uses `?inline=1`.** Downloads use the same route without
  it, which sets `Content-Disposition: attachment`. Don't serve the player the
  attachment variant.
- **`out/server.log` and `out/history.json`** live beside the wavs and must
  never be listable, downloadable or deletable through the outputs API.
