#!/usr/bin/env bash
# Installs the ML stack. ~4.5 GB. Run from ~/voicelab.
set -euo pipefail
cd "$(dirname "$0")"

free=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
echo "Free disk: ${free} GB"
[ "$free" -lt 8 ] && { echo "Need at least 8 GB free. Aborting."; exit 1; }

command -v espeak-ng >/dev/null || { echo "==> brew install espeak-ng"; brew install espeak-ng; }

echo "==> Kokoro TTS + torch (~2.9 GB)"
uv pip install -e ".[kokoro]"

echo "==> RVC conversion stack (~1.2 GB)"
uv pip install -e ".[rvc]"

echo "==> RVC pitch model (~180 MB; HuBERT downloads on first use)"
.venv/bin/python -m voicelab.setup_assets

echo
echo "Done. Start the app:  .venv/bin/python -m uvicorn voicelab.server:app --port 8791"
