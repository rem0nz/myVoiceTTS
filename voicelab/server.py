"""FastAPI backend: upload an RVC package, register it, synthesize with it."""
from __future__ import annotations

import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse)

from . import outputs, presets, rvcpack
from .audio import save
from .config import OUT, RVC_MODELS, free_disk_gb, have, torch_device
from .engines import tts_engines

app = FastAPI(title="voicelab", docs_url="/api/docs")
WEB = Path(__file__).parent / "web"
MAX_UPLOAD_MB = 600


# --------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text()


# -------------------------------------------------------------------- status
@app.get("/api/status")
def status() -> dict:
    engines = []
    for name, eng in tts_engines().items():
        ok, msg = eng.check()
        engines.append({"name": name, "kind": eng.kind, "clones": eng.clones,
                        "ok": ok, "msg": msg, "voices": eng.voices() if ok else [],
                        "notes": eng.notes})
    rvc_ok, rvc_msg = False, "not installed"
    try:
        from .engines.rvc import RVCEngine
        rvc_ok, rvc_msg = RVCEngine().check()
    except ImportError:
        pass
    return {
        "engines": engines,
        "rvc": {"ok": rvc_ok, "msg": rvc_msg},
        "device": torch_device(),
        "free_gb": round(free_disk_gb(), 1),
        "ffmpeg": have("ffmpeg"),
        "espeak": have("espeak-ng"),
    }


# -------------------------------------------------------------------- models
@app.get("/api/models")
def list_models() -> list[dict]:
    return rvcpack.listing()


def _spool(upload: UploadFile) -> Path:
    """Stream an upload to disk in chunks; never buffer 100+ MB in memory."""
    tmp = Path(tempfile.mkdtemp(prefix="voicelab-up-")) / (upload.filename or "up.zip")
    total = 0
    with tmp.open("wb") as fh:
        while chunk := upload.file.read(1 << 20):
            total += len(chunk)
            if total > MAX_UPLOAD_MB * 1024 * 1024:
                fh.close()
                shutil.rmtree(tmp.parent, ignore_errors=True)
                raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB} MB")
            fh.write(chunk)
    return tmp


@app.post("/api/models/inspect")
async def inspect_model(file: UploadFile = File(...)) -> dict:
    """Dry run: report what's inside without writing anything to the library."""
    tmp = _spool(file)
    try:
        plan = rvcpack.inspect_zip(tmp)
        return {"ok": True, "pth": plan["pth"], "index": plan["index"],
                "metadata": plan["metadata"], "files": plan["all_files"][:50],
                "size_mb": round(tmp.stat().st_size / 1024**2, 1)}
    except rvcpack.PackError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


@app.post("/api/models")
async def add_model(file: UploadFile = File(...), name: str = Form(""),
                    overwrite: bool = Form(False)) -> dict:
    if free_disk_gb() < 2:
        raise HTTPException(507, "under 2 GB free - free space before installing")
    tmp = _spool(file)
    try:
        info = rvcpack.install(tmp, name=name.strip() or None, overwrite=overwrite)
        return {"ok": True, "model": info.to_dict()}
    except rvcpack.PackError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


@app.delete("/api/models/{name}")
def delete_model(name: str) -> dict:
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad model name")
    return {"ok": rvcpack.remove(name)}


# ------------------------------------------------------------------- presets
@app.get("/api/presets")
def list_presets() -> dict:
    try:
        return {"ok": True, "presets": presets.listing()}
    except presets.PresetError as exc:
        return {"ok": False, "error": str(exc), "presets": {}}


@app.post("/api/presets")
async def save_preset(name: str = Form(...), engine: str = Form("kokoro"),
                      voice: str = Form(""), rvc_model: str = Form(""),
                      pitch: int = Form(0), index_rate: float = Form(0.66),
                      protect: float = Form(0.33), f0_method: str = Form("rmvpe"),
                      speed: float = Form(1.0), text: str = Form("")) -> dict:
    try:
        saved = presets.save(name, {
            "engine": engine, "voice": voice, "rvc_model": rvc_model,
            "pitch": pitch, "index_rate": index_rate, "protect": protect,
            "f0_method": f0_method, "speed": speed, "text": text})
        return {"ok": True, "saved": saved}
    except presets.PresetError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.delete("/api/presets/{name}")
def delete_preset(name: str) -> dict:
    try:
        return {"ok": presets.delete(name)}
    except presets.PresetError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/presets/export")
def export_presets():
    return PlainTextResponse(
        presets.export_blob(), media_type="application/json",
        headers={"Content-Disposition":
                 'attachment; filename="voicelab-presets.json"'})


@app.post("/api/presets/import")
async def import_presets(file: UploadFile = File(...),
                         replace: bool = Form(False)) -> dict:
    blob = await file.read()
    if len(blob) > 1_000_000:
        raise HTTPException(413, "preset file too large")
    try:
        return {"ok": True, "result": presets.import_blob(blob, replace=replace)}
    except presets.PresetError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


# --------------------------------------------------------------------- synth
@app.post("/api/synth")
async def synth(text: str = Form(...), engine: str = Form("kokoro"),
                voice: str = Form(""), speed: float = Form(1.0),
                rvc_model: str = Form(""), pitch: int = Form(0),
                index_rate: float = Form(0.66), protect: float = Form(0.33),
                f0_method: str = Form("rmvpe")) -> dict:
    text = text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > 5000:
        raise HTTPException(400, "text too long (5000 char limit)")

    engines = tts_engines()
    if engine not in engines:
        raise HTTPException(400, f"unknown engine '{engine}'")
    eng = engines[engine]
    ok, msg = eng.check()
    if not ok:
        raise HTTPException(400, f"engine '{engine}' unavailable: {msg}")

    try:
        audio = eng.synth(text, voice=voice or None, speed=speed)
        stage = f"{engine}"
        if rvc_model:
            try:
                from .engines.rvc import RVCEngine
            except ImportError:
                raise HTTPException(400, "RVC not installed - "
                                         "uv pip install -e '.[rvc]'")
            rvc = RVCEngine()
            rok, rmsg = rvc.check()
            if not rok:
                raise HTTPException(400, f"RVC unavailable: {rmsg}")
            audio = rvc.convert(audio, model=rvc_model, pitch=pitch,
                                index_rate=index_rate, protect=protect,
                                f0_method=f0_method)
            stage += f" -> rvc:{rvc_model}"
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"synth_{stamp}.wav"
    save(OUT / fname, audio.wav, audio.sr)
    outputs.record(fname, pipeline=stage, text=text)
    return {"ok": True, "file": fname, "url": f"/api/out/{fname}",
            "duration": round(audio.duration, 2), "pipeline": stage}


# ------------------------------------------------------------------- outputs
@app.get("/api/out")
def list_outputs(limit: int = 200) -> dict:
    items = outputs.listing(limit=limit)
    return {"ok": True, "items": items, "count": len(items),
            "total_mb": round(outputs.total_bytes() / 1024**2, 2)}


@app.get("/api/out.zip")
def bundle_outputs(names: str = ""):
    """names: optional comma-separated list; empty means everything."""
    wanted = [n for n in (names.split(",") if names else []) if n.strip()]
    try:
        z = outputs.bundle(wanted or None)
    except outputs.OutputError as exc:
        raise HTTPException(400, str(exc))
    return FileResponse(z, media_type="application/zip", filename=z.name)


@app.get("/api/out/{fname}")
def get_out(fname: str, inline: bool = False):
    """inline=1 serves without attachment disposition, for the <audio> player."""
    try:
        p = outputs.resolve(fname)
    except outputs.OutputError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError:
        raise HTTPException(404, "not found")
    if inline:
        return FileResponse(p, media_type="audio/wav")
    return FileResponse(p, media_type="audio/wav", filename=fname)


@app.delete("/api/out/{fname}")
def delete_out(fname: str) -> dict:
    try:
        return {"ok": outputs.remove(fname)}
    except outputs.OutputError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/out")
def clear_outputs() -> dict:
    return {"ok": True, "deleted": outputs.clear()}
