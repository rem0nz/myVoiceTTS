"""Generated-audio management for OUT/: best-effort history metadata,
safe filename resolution, listings, deletion, and zip bundles."""
from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import OUT, free_disk_gb

try:
    import soundfile as _sf
except ImportError:  # header probe degrades to duration/sr = None
    _sf = None

HISTORY = OUT / "history.json"
SCHEMA = 1
TEXT_LIMIT = 500
BUNDLE_DIR = OUT / ".bundle"
BUNDLE_MAX_AGE_S = 3600


class OutputError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_history() -> dict[str, Any]:
    """Best-effort read; a corrupt or missing history degrades to empty."""
    try:
        data = json.loads(HISTORY.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"version": SCHEMA, "items": {}}
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        return {"version": SCHEMA, "items": {}}
    return data


def _save_history(data: dict[str, Any]) -> None:
    """Atomic write; history is metadata, so write failures are swallowed."""
    data["version"] = SCHEMA
    tmp = HISTORY.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(HISTORY)    # atomic; never leaves a half-written store
    except OSError:
        pass


def record(fname: str, pipeline: str = "", text: str = "") -> None:
    """Add/update the history entry for a generated file. Never raises."""
    data = _load_history()
    data["items"][fname] = {
        "pipeline": pipeline,
        "text": (text or "")[:TEXT_LIMIT],
        "created": _now_iso(),
    }
    _save_history(data)


def _probe(path: Path) -> tuple[float | None, int | None]:
    """Header-only duration/samplerate; any failure -> (None, None)."""
    if _sf is None:
        return None, None
    try:
        info = _sf.info(str(path))
        return round(float(info.duration), 2), int(info.samplerate)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None, None       # truncated/partial file must not break listing


def listing(limit: int = 200) -> list[dict[str, Any]]:
    """All wavs in OUT, newest first, enriched with history metadata."""
    items = _load_history()["items"]
    rows: list[tuple[float, dict[str, Any]]] = []
    for path in OUT.glob("*.wav"):
        try:
            st = path.stat()
        except OSError:
            continue            # vanished mid-scan
        duration, sr = _probe(path)
        meta = items.get(path.name)
        if not isinstance(meta, dict):
            meta = {}
        created = meta.get("created") or datetime.fromtimestamp(
            st.st_mtime, timezone.utc).isoformat(timespec="seconds")
        rows.append((st.st_mtime, {
            "file": path.name,
            "size": st.st_size,
            "size_mb": round(st.st_size / 1024**2, 2),
            "duration": duration,
            "sr": sr,
            "created": created,
            "pipeline": str(meta.get("pipeline", "")),
            "text": str(meta.get("text", "")),
        }))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [row for _, row in rows[:limit]]


def resolve(fname: str) -> Path:
    """Resolve a bare wav filename inside OUT, rejecting traversal escapes."""
    if "/" in fname or "\\" in fname or ".." in fname:
        raise OutputError(f"invalid file name: {fname!r}")
    if not fname.lower().endswith(".wav"):
        raise OutputError(f"not a .wav file: {fname!r}")
    path = (OUT / fname).resolve()
    if path.parent != OUT.resolve():
        raise OutputError(f"path escapes the output dir: {fname!r}")
    if not path.is_file():
        raise FileNotFoundError(fname)
    return path


def remove(fname: str) -> bool:
    """Delete one wav and prune its history entry. False if already gone."""
    try:
        path = resolve(fname)
        path.unlink()
    except FileNotFoundError:
        return False
    data = _load_history()
    if data["items"].pop(fname, None) is not None:
        _save_history(data)
    return True


def clear() -> int:
    """Delete every wav in OUT (never server.log or history.json)."""
    count = 0
    for path in OUT.glob("*.wav"):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        count += 1
    data = _load_history()
    data["items"] = {}
    _save_history(data)
    return count


def _sweep_old_bundles() -> None:
    cutoff = time.time() - BUNDLE_MAX_AGE_S
    for old in BUNDLE_DIR.glob("*.zip"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            continue


def bundle(names: list[str] | None = None) -> Path:
    """Zip the requested wavs (or all of them) plus a manifest.json."""
    if names:
        paths = list(dict.fromkeys(resolve(n) for n in names))
    else:
        paths = sorted(OUT.glob("*.wav"))
    if not paths:
        raise OutputError("no audio files to bundle")
    if free_disk_gb() < 1:
        raise OutputError("low disk: under 1 GB free, refusing to bundle")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    _sweep_old_bundles()
    wanted = {p.name for p in paths}
    manifest = [row for row in listing(limit=1_000_000) if row["file"] in wanted]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zpath = BUNDLE_DIR / f"voicelab-audio-{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=path.name)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return zpath


def total_bytes() -> int:
    total = 0
    for path in OUT.glob("*.wav"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total
