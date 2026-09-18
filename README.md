# voicelab

Local voice-synthesis workbench for Apple Silicon. Load an RVC voice model,
register it, and drive text-to-speech through it — web UI or CLI. Nothing
leaves the machine.

## The one thing to understand first

**RVC is voice *conversion*, not text-to-speech.** It re-timbres existing
speech; it cannot read text. So TTS in your own voice is two stages:

```
text ──[Kokoro]──> source speech ──[RVC + your model]──> your voice
        stage 1                      stage 2
```

Your trained model plugs into stage 2. Stage 1 can be any TTS engine.

## Recommended stack for this machine

Measured on the target host: Apple M5 Pro, 48 GB RAM, MPS available,
**~20 GB free disk** — disk is the binding constraint, not compute.

| Stage | Pick | Why |
|---|---|---|
| TTS | **Kokoro-82M** | Apache-2.0, ~0.3 s/sentence on MPS, genuinely good. Needs torch — which RVC needs anyway, so its marginal cost is only ~350 MB. Ships British voices (`bf_emma`, `bm_george`). |
| Conversion | **RVC v2** (`infer-rvc-python`) | Matches the format you're training. 40 kHz, f0-conditioned. Runs on MPS with half precision. |

Rejected on disk grounds: F5-TTS, XTTS-v2 and Chatterbox each cost
1.5–4 GB and do their *own* zero-shot cloning — redundant once you have a
properly trained RVC model, and they'd eat a third of your free space.

Fallbacks with no torch at all: **Piper** (ONNX, ~60 MB/voice) and
**system** (macOS `say`, zero install — the sanity baseline).

Budget for the full install: **~4.5 GB.**

## Install

```bash
cd ~/voicelab
uv venv --python 3.11
uv pip install -e ".[web]"      # core + web UI, ~100 MB
./install-heavy.sh              # Kokoro + RVC + rmvpe, ~4.3 GB
```

`install-heavy.sh` refuses to run under 8 GB free, installs `espeak-ng` via
brew if missing, and fetches `rmvpe.pt` (the shared pitch model). HuBERT is
**not** pre-fetched: `infer-rvc-python` pulls it through `transformers`
(`r3gm/hubert_base`) on the first conversion.

### Backend history — why it looks like this

The obvious choice, `rvc-python`, is a trap on this machine and the packaging
encodes three fixes worth keeping:

1. It depends on **`fairseq==0.12.2`**, which has no macOS-arm64 wheel and
   only builds on Python ≤3.10.
2. Its pins are stale — it demands `numpy<=1.23.5`, irreconcilable with torch
   and Kokoro, though its runtime code is fine on numpy 2.x.
3. Even once installed, fairseq's pickled `Dictionary` inside `hubert_base.pt`
   is rejected by **torch ≥2.6's `weights_only=True` default**, and
   `rvc-python` reports that failure as
   `AttributeError: 'tuple' object has no attribute 'dtype'` — it hands
   `vc_single`'s `(info, (None, None))` failure tuple straight to
   `scipy.io.wavfile.write`.

`infer-rvc-python` loads HuBERT via `transformers` and has none of these
problems. It also picks up **MPS with half precision**, where the fairseq
path fell back to CPU.

Two adjacent gotchas that still apply:

- `setuptools<81` is pinned — the resolved `librosa` still imports
  `pkg_resources`, which setuptools dropped in 81.
- `rvc-inferpy`'s PyPI metadata advertises `fairseq2`, but its source does
  `from fairseq import checkpoint_utils` (the unrelated legacy package). It
  imports fine and only fails at conversion time. Avoid.

## Run

```bash
.venv/bin/python -m uvicorn voicelab.server:app --port 8791
# → http://127.0.0.1:8791
```

Web UI: drag your `.zip` in (it's inspected before anything is written),
name it, install, then type text and hit Generate.

CLI:

```bash
.venv/bin/python -m voicelab.cli doctor              # what's installed
.venv/bin/python -m voicelab.cli voices              # list source voices
.venv/bin/python -m voicelab.cli say "hello" --engine kokoro --voice bm_george \
    --rvc myvoice --pitch 0 --play
.venv/bin/python -m voicelab.cli convert in.wav --model myvoice -o out.wav
```

## Presets

Every synth setting (engine, voice, RVC model, pitch, index rate, protect,
F0 method, speed, and optionally the text) saves as a named preset.

- **Save as…** names the current form state.
- **Export** downloads `voicelab-presets.json`.
- **Import** merges a file in — existing names are updated, new ones added,
  malformed entries skipped and reported. Nothing is silently dropped.

Stored at `~/voicelab/presets.json`, written atomically via a temp file, so an
interrupted save can't corrupt the store. Values are clamped on the way in
(pitch to ±24, index rate 0–1, protect 0–0.5, speed 0.5–2), so a hand-edited
or foreign file can't push the UI into an invalid state. The file is plain
JSON — safe to hand-edit or keep in version control.

## Generated audio

Everything you generate lands in `~/voicelab/out/` and appears in the
**Generated audio** panel: play, download, or delete any past take, plus
**Download all (.zip)** and **Clear all**.

The zip carries a `manifest.json` listing each included file with its
duration, size, pipeline, and source text — so a bundle stays self-describing
once it leaves the machine.

Filenames are validated server-side before any file is opened: no `/`, `\`,
or `..`, `.wav` only, and the resolved path's parent must be `out/` (which
also blocks symlink escapes). `server.log` and `history.json` live in the same
directory and are never listable, downloadable, or deletable through the API.

Per-file metadata (which pipeline produced it, the text) is recorded in
`out/history.json` on a best-effort basis — a missing or corrupt file degrades
to an empty pipeline label rather than breaking the listing, and files
generated before that existed simply show their filename.

## Model package format

What the loader expects — names are globbed, so trainer defaults are fine:

```
YourVoice.zip
├── <any folder>/<any name>.pth     generator weights + embedded config
├── <any folder>/<any name>.index   FAISS retrieval index (optional)
└── metadata.json                   optional descriptor
```

The `.pth` is a torch archive whose top-level dict carries `weight`,
`config` (18 hyperparameters), `sr`, `f0`, `version`, `info`. The installer
reads those, writes `models/rvc/<name>/voicelab.json`, and warns you about:

- no `.index` → conversion works, timbre fidelity is noticeably worse
- `f0=0` → not pitch-conditioned, `--pitch` will be ignored
- no `weight` key → you exported a training checkpoint (`G_*.pth`) rather
  than an extracted model

**Security:** uploaded `.pth` files are loaded with `weights_only=True`.
A `.pth` is a pickle; without that flag a malicious checkpoint executes
arbitrary code on load. Don't relax it.

## Cloning without training (F5-TTS)

```bash
uv pip install -e ".[f5,clone]"
.venv/bin/python -m voicelab.cli mics                       # find your input device
.venv/bin/python -m voicelab.cli clone --name me --mic "Wireless Mic Rx"
```

`clone` records you reading one sentence (5-10 s), saves it as
`voices/me/ref.wav` + `ref.txt`, then loops: type a line, hear it in your voice.
Afterwards the voice is available everywhere as `--engine f5 --voice me`, and in
the web UI's voice picker. Quality is below a trained RVC model but there is no
training step. F5 weights are CC-BY-NC.

## Training your own

```bash
.venv/bin/python -m voicelab.cli dataset-prep ~/recordings --name myvoice
.venv/bin/python -m voicelab.cli dataset-stats myvoice
```

Segments recordings into 3–12 s clips, trims silence, normalizes, drops
near-silent takes. Want ~10+ minutes of clean single-speaker audio: one mic,
one room, no music, no reverb, no second voice. Past ~45 minutes returns
diminish sharply.

Then train with upstream RVC (this repo deliberately doesn't reimplement the
trainer) and drop the resulting zip into the web UI.

### Getting good conversions

- **Match the source to the target.** A male target converts better from
  `am_adam`/`bm_george` than from a female source. Less pitch shifting =
  fewer artifacts. This matters more than any slider.
- `index_rate` (0–1) — how hard to pull toward the trained timbre. ~0.66 is
  a good start; too high gets muddy.
- `protect` (0–0.5) — shields breath and consonants from over-conversion.
- `f0_method` — `rmvpe` is the best general choice; `rmvpe+` refines it,
  `pm` is fastest.

## Layout

```
voicelab/
├── config.py        paths, device detection, disk checks
├── audio.py         io, resample, trim, silence-splitting (numpy only)
├── rvcpack.py       RVC package inspect / validate / install
├── dataset.py       raw recordings → training clips
├── server.py        FastAPI backend
├── cli.py           typer CLI
├── web/index.html   single-file UI, no build step
└── engines/
    ├── base.py      Engine / TTSEngine / VCEngine contracts
    ├── system.py    macOS say      (no install)
    ├── kokoro.py    Kokoro-82M     [kokoro]
    ├── piper.py     Piper ONNX     [piper]
    └── rvc.py       RVC v2         [rvc]  infer-rvc-python, fairseq-free
```

Engines are lazily imported and independently optional — a missing engine
degrades to a status message, never an import crash.
