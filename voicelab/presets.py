"""Named synthesis presets, stored as one JSON file so they can be
exported, version-controlled, or hand-edited."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .config import ROOT

STORE = ROOT / "presets.json"
SCHEMA = 1

# field -> (type, default, validator)
FIELDS: dict[str, tuple[type, Any]] = {
    "engine": (str, "kokoro"),
    "voice": (str, ""),
    "rvc_model": (str, ""),
    "pitch": (int, 0),
    "index_rate": (float, 0.66),
    "protect": (float, 0.33),
    "f0_method": (str, "rmvpe"),
    "speed": (float, 1.0),
    "text": (str, ""),          # optional; handy for repeatable test lines
}

_BOUNDS = {
    "pitch": (-24, 24),
    "index_rate": (0.0, 1.0),
    "protect": (0.0, 0.5),
    "speed": (0.5, 2.0),
}


class PresetError(ValueError):
    pass


def _load() -> dict:
    if not STORE.exists():
        return {"version": SCHEMA, "presets": {}}
    try:
        data = json.loads(STORE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise PresetError(f"presets.json is unreadable: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("presets"), dict):
        raise PresetError("presets.json has the wrong shape")
    return data


def _save(data: dict) -> None:
    data["version"] = SCHEMA
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(STORE)          # atomic; never leaves a half-written store


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name:
        raise PresetError("preset needs a name")
    if len(name) > 60:
        raise PresetError("preset name is too long (60 char limit)")
    return name


def coerce(raw: dict) -> dict:
    """Accept a loose dict (e.g. form data or an imported file) -> valid preset."""
    out: dict[str, Any] = {}
    for key, (typ, default) in FIELDS.items():
        val = raw.get(key, default)
        try:
            val = typ(val) if val != "" or typ is str else default
        except (TypeError, ValueError):
            val = default
        if key in _BOUNDS:
            lo, hi = _BOUNDS[key]
            val = max(lo, min(hi, val))
        out[key] = val
    return out


def listing() -> dict:
    return _load()["presets"]


def save(name: str, values: dict) -> dict:
    name = clean_name(name)
    data = _load()
    preset = coerce(values)
    preset["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["presets"][name] = preset
    _save(data)
    return {name: preset}


def delete(name: str) -> bool:
    data = _load()
    if name not in data["presets"]:
        return False
    del data["presets"][name]
    _save(data)
    return True


def export_blob() -> str:
    return json.dumps(_load(), indent=2, sort_keys=True)


def import_blob(blob: str | bytes, replace: bool = False) -> dict:
    """Merge presets from an exported file. Returns a summary."""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", errors="replace")
    try:
        incoming = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise PresetError(f"not valid JSON: {exc}") from exc

    # tolerate both {"presets": {...}} and a bare {name: {...}} mapping
    block = incoming.get("presets") if isinstance(incoming, dict) else None
    if block is None:
        block = incoming
    if not isinstance(block, dict):
        raise PresetError("expected an object of presets")

    data = _load()
    if replace:
        data["presets"] = {}

    added, updated, skipped = [], [], []
    for raw_name, raw_val in block.items():
        if not isinstance(raw_val, dict):
            skipped.append(str(raw_name))
            continue
        try:
            name = clean_name(str(raw_name))
        except PresetError:
            skipped.append(str(raw_name))
            continue
        (updated if name in data["presets"] else added).append(name)
        preset = coerce(raw_val)
        preset["saved_at"] = raw_val.get("saved_at") or datetime.now(
            timezone.utc).isoformat(timespec="seconds")
        data["presets"][name] = preset

    _save(data)
    return {"added": added, "updated": updated, "skipped": skipped,
            "total": len(data["presets"])}
