#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
echo "======================================================================"
echo "🚀 Menjalankan Vocal-Guard Streaming Demo Server (CPU Local)"
echo "👉 Buka di browser: http://127.0.0.1:8000"
echo "======================================================================"
./.venv/bin/python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
