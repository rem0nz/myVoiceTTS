# Fish Audio S2-Pro on Apple Silicon — evaluation

Dated 2026-09-18. Machine: Apple M5 Pro, 48 GB unified memory, macOS 25.6.

## Why this exists

voicelab's TTS is a two-stage pipeline: a TTS engine produces speech, then RVC
re-timbres it. **RVC preserves the source prosody.** If stage 1 reads flat, the
output is flat in someone else's voice — no RVC parameter can fix that, because
RVC has no concept of emotion. F5-TTS has the same problem from the other side:
it clones timbre with no emotion conditioning at all (`nfe_step` and
`cfg_strength` are sampling-quality knobs, not expression).

So the question was whether a single model could do cloning *and* expression in
one pass. Fish Audio S2-Pro claims exactly that.

## Verdict

| claim | status |
|---|---|
| Runs locally on Apple Silicon (MPS) | **CONFIRMED** |
| Clones a voice from a short reference | **CONFIRMED** — sounds like the speaker |
| Emotion via inline tags | **PARTIAL** — structural tags work, emotion words mostly do not |
| Emotion decoupled from timbre | **NOT SUPPORTED** — architectural limit, see below |

Fish's own docs list supported platforms as "Linux, WSL" and never mention
macOS. That is a documentation gap, not a capability gap: MPS support is in the
code.

## Licence — read before using this commercially

`FISH AUDIO RESEARCH LICENSE`, confirmed in the repo's `LICENSE` and on the
HuggingFace model card. Research and non-commercial use free; **commercial use
requires a separate licence from Fish Audio** (business@fish.audio). It is
open-weight, not open-source. Third-party blogs claiming MIT are wrong.

## Setup

Fish pins `torch==2.8.0`; voicelab runs `torch 2.14.0`. **They cannot share a
venv.** Keep fish-speech isolated.

```bash
git clone https://github.com/fishaudio/fish-speech.git ~/fishlab
cd ~/fishlab
uv venv --python 3.12
brew install portaudio                 # else the pyaudio build fails
uv pip install -e .
uv pip install 'tensorboard<2.20'      # see "protobuf" below
uv pip install sounddevice             # for the recording scripts
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro   # ~10 GB
```

Then copy the scripts from this directory into `~/fishlab/`.

### Two install traps

1. **`portaudio.h` not found** — `pyaudio` needs headers. `brew install
   portaudio`. (Linux docs say `apt install portaudio19-dev`.)
2. **protobuf version clash** — `descript-audiotools` pins `protobuf<3.20`,
   `tensorboard 2.21` needs `protobuf>=6.31.1`. Irreconcilable; the resolver
   lands on 5.29.6 and `import audiotools` dies with a `VersionError`.
   Downgrading to `tensorboard<2.20` fixes it. tensorboard is training-only and
   irrelevant to inference. Do not "fix" this by upgrading protobuf — the
   audiotools pin wins and you get the same error.

## Evidence that MPS is supported

Grepped from the source, not inferred:

- `fish_speech/utils/context.py` defines `autocast_exclude_mps()`, which
  disables autocast on MPS. It is wired into the live inference path at
  `fish_speech/inference_engine/__init__.py:185`. Someone hit MPS autocast
  failures and fixed them deliberately.
- `tools/run_webui.py:49` and `tools/server/model_manager.py:31` both probe
  **MPS first**, ahead of XPU/CUDA/CPU.
- No `flash_attn`, `triton`, `sglang` or `vllm` in the core inference path —
  the usual Apple Silicon blockers are absent.
- Every `torch.cuda.*` call is guarded by `is_available()`;
  `text2semantic/inference.py:408` falls back to `aot_eager` compile off-CUDA.
- The one unguarded `.cuda()` (`models/dac/modded_dac.py:1018`) is inside
  `if __name__ == "__main__":` — a developer scratch script, not the inference
  path.

Confirmed at runtime: torch 2.8.0, `mps.is_available() == True`, **bfloat16 on
MPS works** (that matters — `run_webui.py` defaults to bf16).

## Results

### Performance

RTF ~2.8–2.9 once warm (about 3x slower than realtime). Model load ~60 s.
First one or two generations after load are slower (cold cache); discount them.
Output is **44.1 kHz** — voicelab's internal rate is 24 kHz, so integration
needs a resample.

### Cloning

A 24 s reference at 21 dB SNR produced output the speaker identified as
recognisably their own voice. Long-term spectral overlap with the reference was
a consistent 0.59–0.66 across all generations.

Reference guidance, learned the hard way:

- **10–30 s.** A first attempt at 34 s over-anchored and flattened everything.
- **Quiet room.** S2-Pro clones room tone along with the voice. 21 dB SNR was
  workable; aim higher.
- **Transcript must match the audio exactly.** The model aligns them.
- 44.1 kHz mono. Record via PortAudio, **not** ffmpeg `avfoundation` — that
  mislabels 24-bit USB receivers and writes files of exact zeros (see the main
  CLAUDE.md).

### Emotion tags

**The documented tag vocabulary is the one that works.** Invented variants are
read as literal text and largely ignored — `[whispers]` moved RMS +0.5% (i.e.
nothing) while the correct `[whisper]` moved it −11.4%.

Supported tags, from `docs/README.zh.md:117`:

```
[pause] [emphasis] [laughing] [inhale] [chuckle] [tsk] [singing] [excited]
[laughing tone] [interrupting] [chuckling] [excited tone] [volume up] [echo]
[angry] [low volume] [sigh] [low voice] [whisper] [screaming] [shouting]
[loud] [surprised] [short pause] [exhale] [delight] [panting]
[audience laughter] [with strong accent] [volume down] [clearing throat]
[sad] [moaning] [shocked]
```

Free-form descriptions are also claimed to work (`[whisper in small voice]`,
`[professional broadcast tone]`, `[pitch up]`) — untested here.

Sweep of 13 cases, identical reference / sentence / seed, ranked by change in
dynamic range (a proxy for expressive vs flat):

| case | Δdyn | Δduration | Δrms |
|---|---|---|---|
| `[pause]` + `[emphasis]` mid-sentence | **+6.2 dB** | +20.7% | −5.8% |
| `[excited tone]` … `[laughing]` | **+5.6 dB** | +28.7% | −14.4% |
| `[screaming]` | **+5.1 dB** | +17.2% | −0.4% |
| `[shouting]` | +3.6 dB | +2.3% | +9.1% |
| `[whisper][low voice][low volume]` | +2.9 dB | 0.0% | −4.1% |
| `[shocked]` | +2.4 dB | +4.6% | +3.0% |
| `[sigh][sad]` | +1.9 dB | +10.3% | +7.9% |
| `[whisper]` | +1.5 dB | +3.4% | −11.4% |
| `[angry]` | +1.3 dB | −3.4% | −6.0% |
| `[with strong accent]` | +1.3 dB | +12.6% | −17.8% |
| `[sad]` | +0.6 dB | +2.3% | −8.5% |
| `[excited]` | **−0.8 dB** | −1.1% | −2.7% |

Three conclusions:

1. **Structural tags beat emotion words.** `[pause]` and `[emphasis]` placed
   mid-sentence were the single largest effect. They change timing and stress
   rather than trying to override delivery, so the reference's prosody doesn't
   suppress them. `[laughing]` also reliably inserts a real laugh.
2. **Intensity escalates:** `[angry]` +1.3 dB → `[shouting]` +3.6 →
   `[screaming]` +5.1. Mild emotion words are weak; strong physical ones bite.
3. **Listener verdict was still "mostly flat."** The measured deltas are real
   but subtle to the ear. Only `[pause]`/`[emphasis]` and `[laughing]` were
   judged clearly effective.

### The architectural limit

`ServeReferenceAudio` has exactly two fields, `audio` and `text`. There is no
separate emotion-reference parameter — grep for `emo_audio`, `emotion_ref`,
`style_ref` returns nothing.

**One reference slot carries timbre and delivery baseline together.** A neutral
reference therefore sets a neutral prosodic prior that the tags have to fight,
which is why emotion words land weakly while structural tags get through.

The implied workaround is one reference *per emotion* — record yourself angry,
excited, whispering — and clone from the matching one. **This is untested**;
it's the next experiment (`record_emotions.py`).

An attempt to shortcut it by re-using expressive *generated* clips as
references failed (+0.2 to +0.5 dB only), but that test was flawed: those clips
were themselves generated from a flat reference, so they had no real
expressiveness to transfer. It does not disprove the hypothesis.

## If per-emotion references don't work either

Then S2-Pro's expressiveness is bounded by its single reference slot, and the
right tool is **IndexTTS-2** (Apache-2.0, genuinely open source), which takes
`spk_audio_prompt` and `emo_audio_prompt` as *separate* arguments — the emotion
prompt may even come from a different speaker than the timbre prompt. It also
offers an 8-float emotion vector and text mood descriptions, and has a native
MLX port for Apple Silicon.

A voicelab engine for it is already written at `voicelab/engines/indextts.py`
(not yet wired into the registry). Weights are ~7 GB.

## Scripts

Copy into `~/fishlab/` and run with `.venv/bin/python -u <script>`.

| script | purpose |
|---|---|
| `record_ref.py` | Record a 44.1 kHz neutral reference. Prints the script to read, validates level/length/silence, refuses to save bad takes. |
| `record_emotions.py` | Record one short performed reference per emotion (angry/excited/whisper/sad). |
| `prove_mps.py` | Minimal proof: load on MPS, generate un-cloned speech. |
| `prove_clone.py` | Cloned voice + emotion tags, fixed seed. |
| `prove_me.py` | Same, against `ref/me_clean.wav`. |
| `sweep_tags.py` | 13-case canonical tag sweep. |
| `test_refswap.py` | Does the reference drive delivery? Same text, no tags, different references. |

## Not included in this repo, deliberately

No audio of anyone's voice — no reference recordings, no generated clips. This
repo is public, and a reference clip plus its matching transcript is a
ready-made cloning kit for that person. Same reasoning as the main CLAUDE.md
policy on voice models. Record your own with `record_ref.py`; it takes two
minutes.

Model weights (~10 GB) are not here either — they exceed GitHub's 100 MB
per-file limit and carry a non-commercial licence. Download them from
HuggingFace as shown above.
