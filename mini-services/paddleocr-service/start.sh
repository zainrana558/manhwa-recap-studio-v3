#!/bin/bash
# Start the PaddleOCR service
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/.venv/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 3002 --workers 1
