#!/usr/bin/env bash
# Quick runner for AutoClipper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv .venv
    source .venv/bin/activate
    uv pip install yt-dlp faster-whisper numpy scipy "opencv-python-headless<5" fastapi uvicorn[standard] python-multipart requests edge-tts
else
    source .venv/bin/activate
fi

# Ensure ffmpeg in PATH
export PATH="/home/agent/.local/bin:$PATH"

if [ "$1" == "--web" ] || [ "$1" == "-w" ]; then
    echo "Starting AutoClipper Web Studio on http://localhost:8000..."
    exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
else
    exec python3 autoclipper.py "$@"
fi
