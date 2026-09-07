"""Fetch the shared RVC base models (needed once, by every RVC voice model)."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from .config import RVC_MODELS, free_disk_gb

# Only rmvpe is pre-fetched. HuBERT is NOT here on purpose: infer-rvc-python
# loads it through transformers (r3gm/hubert_base) on first conversion, and
# the old fairseq-format hubert_base.pt is not loadable by from_pretrained.
ASSETS = {
    "rmvpe.pt": (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        181_184_272),
}


def _hook(name: str):
    """Progress only on a TTY; a \r spinner into a pipe emits megabytes."""
    if not sys.stdout.isatty():
        return None
    last = [-1]

    def report(blocks: int, bs: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, blocks * bs * 100 // total)
        if pct != last[0]:
            last[0] = pct
            sys.stdout.write(f"\r  {name}: {pct}%")
            sys.stdout.flush()
    return report


def fetch(force: bool = False) -> list[Path]:
    dest = RVC_MODELS / "_base"
    dest.mkdir(parents=True, exist_ok=True)
    need = [n for n, _ in ASSETS.items() if force or not (dest / n).exists()]
    if not need:
        print("Base models already present.")
        return [dest / n for n in ASSETS]

    want_gb = sum(ASSETS[n][1] for n in need) / 1024**3
    if free_disk_gb() < want_gb + 1:
        raise RuntimeError(f"need ~{want_gb:.1f} GB free, "
                           f"have {free_disk_gb():.1f} GB")

    out = []
    for name in need:
        url, _ = ASSETS[name]
        tmp = dest / (name + ".part")
        print(f"Fetching {name}…")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_hook(name))
            tmp.replace(dest / name)
            print(f"\r  {name}: done      ")
            out.append(dest / name)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    return out


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
