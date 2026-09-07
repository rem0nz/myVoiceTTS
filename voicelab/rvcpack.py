r"""RVC model-package format: inspect, validate, install.

Layout produced by weights.gg exports and by standard RVC training runs:

    Whatever.zip
    |- <folder>/<name>.pth      generator weights + embedded config
    |- <folder>/<name>.index    FAISS retrieval index (optional)
    \- metadata.json            descriptor (optional, weights.gg only)

The .pth is a torch archive whose top-level dict carries:
    weight   state dict
    config   18-item hyperparameter list
    sr       40000 | 48000 | 32000   (or "40k" style string in extra_info)
    f0       1 if pitch-conditioned
    version  "v1" | "v2"
    info     free text, e.g. "400epoch"

SECURITY: .pth files are pickles. We load with weights_only=True so an uploaded
checkpoint cannot execute code. Never relax that for user-supplied files.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from .config import RVC_MODELS

AUDIO_SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}
CONFIG_FIELDS = [
    "spec_channels", "segment_size", "inter_channels", "hidden_channels",
    "filter_channels", "n_heads", "n_layers", "kernel_size", "p_dropout",
    "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
    "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes",
    "spk_embed_dim", "gin_channels", "sr",
]


class PackError(ValueError):
    """Archive is not a usable RVC package."""


@dataclass
class ModelInfo:
    name: str
    version: str = "v2"
    sr: int = 40000
    f0: int = 1
    epochs: str = ""
    spk_embed_dim: int | None = None
    gin_channels: int | None = None
    pth_bytes: int = 0
    index_bytes: int = 0
    has_index: bool = False
    md5: str = ""
    title: str = ""
    source: str = ""
    installed_at: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_checkpoint(pth: Path) -> dict:
    """Pull the descriptor out of an RVC .pth without instantiating the model."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise PackError("torch is required to inspect a .pth "
                        "- uv pip install -e '.[rvc]'") from exc

    try:
        ckpt = torch.load(str(pth), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PackError(f"could not read checkpoint: {exc}") from exc

    if not isinstance(ckpt, dict):
        raise PackError("checkpoint is not a dict - not an RVC model")

    cfg_list = ckpt.get("config")
    cfg: dict = {}
    if isinstance(cfg_list, (list, tuple)) and len(cfg_list) >= 18:
        cfg = dict(zip(CONFIG_FIELDS, cfg_list))

    sr = ckpt.get("sr", cfg.get("sr", 40000))
    if isinstance(sr, str):
        sr = AUDIO_SR_MAP.get(sr.lower(), 40000)

    return {
        "version": str(ckpt.get("version", "v2")),
        "sr": int(sr),
        "f0": int(ckpt.get("f0", 1)),
        "epochs": str(ckpt.get("info", "")),
        "spk_embed_dim": cfg.get("spk_embed_dim"),
        "gin_channels": cfg.get("gin_channels"),
        "has_weight": "weight" in ckpt,
    }


def inspect_zip(zip_path: Path) -> dict:
    """Look inside an archive without extracting it. Returns a plan dict."""
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise PackError(f"{zip_path.name} is not a zip archive")

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist()
                 if not n.startswith("__MACOSX/")
                 and not Path(n).name.startswith("._")
                 and not n.endswith("/")]

    pths = [n for n in names if n.lower().endswith(".pth")]
    indexes = [n for n in names if n.lower().endswith(".index")]
    metas = [n for n in names if Path(n).name.lower() == "metadata.json"]

    if not pths:
        raise PackError("no .pth found in archive - not an RVC model package")
    if len(pths) > 1:
        pths.sort(key=lambda n: (("g_" in Path(n).name.lower()) or
                                 ("d_" in Path(n).name.lower()), len(n)))

    return {
        "pth": pths[0],
        "index": indexes[0] if indexes else None,
        "metadata": metas[0] if metas else None,
        "extra_pth": pths[1:],
        "all_files": names,
    }


def install(zip_path: Path, name: str | None = None,
            overwrite: bool = False) -> ModelInfo:
    """Extract + validate + register an RVC package into models/rvc/<name>/."""
    zip_path = Path(zip_path)
    plan = inspect_zip(zip_path)

    meta: dict = {}
    with zipfile.ZipFile(zip_path) as zf:
        if plan["metadata"]:
            try:
                meta = json.loads(zf.read(plan["metadata"]).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                meta = {}

        slug = name or _derive_name(meta, plan, zip_path)
        dest = RVC_MODELS / slug
        if dest.exists():
            if not overwrite:
                raise PackError(f"model '{slug}' already exists - pass overwrite")
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        try:
            (dest / "model.pth").write_bytes(zf.read(plan["pth"]))
            if plan["index"]:
                (dest / "model.index").write_bytes(zf.read(plan["index"]))
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise

    pth = dest / "model.pth"
    idx = dest / "model.index"

    try:
        desc = read_checkpoint(pth)
    except PackError:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    warnings: list[str] = []
    if not desc["has_weight"]:
        warnings.append("checkpoint has no 'weight' key - may be a training "
                        "checkpoint (G_*.pth) rather than an extracted model")
    if not idx.exists():
        warnings.append("no .index file - conversion still works but timbre "
                        "fidelity is noticeably lower")
    if desc["f0"] == 0:
        warnings.append("model is not pitch-conditioned (f0=0); --pitch is ignored")
    if plan["extra_pth"]:
        warnings.append(f"archive had {len(plan['extra_pth'])} extra .pth files; "
                        f"used {Path(plan['pth']).name}")

    info = ModelInfo(
        name=slug,
        version=desc["version"],
        sr=desc["sr"],
        f0=desc["f0"],
        epochs=desc["epochs"],
        spk_embed_dim=desc["spk_embed_dim"],
        gin_channels=desc["gin_channels"],
        pth_bytes=pth.stat().st_size,
        index_bytes=idx.stat().st_size if idx.exists() else 0,
        has_index=idx.exists(),
        md5=_md5(pth),
        title=str(meta.get("title", "")),
        source=str(meta.get("weightsLink") or meta.get("url") or zip_path.name),
        installed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        warnings=warnings,
    )
    (dest / "voicelab.json").write_text(json.dumps(info.to_dict(), indent=2))
    return info


def _derive_name(meta: dict, plan: dict, zip_path: Path) -> str:
    raw = (meta.get("title") or Path(plan["pth"]).parent.name
           or zip_path.stem or "model")
    slug = "".join(c if (c.isalnum() or c in "-_") else "-" for c in raw.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "model"


def listing() -> list[dict]:
    out = []
    for d in sorted(RVC_MODELS.iterdir()) if RVC_MODELS.exists() else []:
        if not d.is_dir() or not (d / "model.pth").exists():
            continue
        card = d / "voicelab.json"
        if card.exists():
            try:
                out.append(json.loads(card.read_text()))
                continue
            except json.JSONDecodeError:
                pass
        out.append({"name": d.name, "sr": 40000, "version": "v2",
                    "has_index": (d / "model.index").exists(), "warnings": []})
    return out


def remove(name: str) -> bool:
    dest = RVC_MODELS / name
    if not dest.is_dir():
        return False
    shutil.rmtree(dest)
    return True
