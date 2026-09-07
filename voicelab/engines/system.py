"""macOS `say`. Zero dependencies, zero disk, always works. The sanity baseline."""
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path

from .base import TTSEngine, Audio
from ..audio import load
from ..config import have, SAMPLE_RATE


class SystemEngine(TTSEngine):
    name = "system"
    clones = False
    extra = ""
    notes = "Built-in macOS voices. No cloning, no install, instant."

    def check(self) -> tuple[bool, str]:
        if not have("say"):
            return False, "macOS `say` not found (non-Darwin host?)"
        return True, "ready (built in)"

    def voices(self) -> list[str]:
        if not have("say"):
            return []
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                                 check=True, timeout=15).stdout
        except Exception:
            return []
        names = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and "_" in parts[1]:
                names.append(f"{parts[0]}  ({parts[1]})")
        return names

    def synth(self, text: str, voice: str | None = None,
              ref: Path | None = None, speed: float = 1.0, **kw) -> Audio:
        if ref is not None:
            raise ValueError("system engine cannot clone; use f5/chatterbox/xtts with --ref")
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as t:
            tmp = Path(t.name)
        try:
            cmd = ["say", "-o", str(tmp)]
            if voice:
                cmd += ["-v", voice]
            if speed != 1.0:
                cmd += ["-r", str(int(180 * speed))]
            cmd += ["--", text]
            subprocess.run(cmd, check=True)
            wav, sr = load(tmp, sr=SAMPLE_RATE)
            return Audio(wav, sr)
        finally:
            tmp.unlink(missing_ok=True)
