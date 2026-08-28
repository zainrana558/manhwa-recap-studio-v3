#!/bin/bash
# Start the PaddleOCR service
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi
# C16 FIX: Bind to localhost only — external access should go through Caddy
exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 3002 --workers 1
