#!/bin/bash
# Start the PaddleOCR service
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# Look for a venv in three places, in order:
#   1. A venv local to this service directory (mini-services/paddleocr-service/.venv)
#   2. The project-root venv that setup.sh actually creates (../../.venv) —
#      setup.sh installs paddleocr/paddlepaddle/rapidocr into
#      $PROJECT_DIR/.venv, not a subdirectory-local one, so without this
#      check this script silently fell through to bare system `python3`
#      on any box that only ran setup.sh, missing every dependency it had
#      just installed.
#   3. Bare `python3` as a last resort (e.g. dependencies installed
#      system-wide, or a venv already active in the parent environment).
ROOT_VENV="$SCRIPT_DIR/../../.venv"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$ROOT_VENV/bin/python" ]; then
    PYTHON_BIN="$ROOT_VENV/bin/python"
else
    PYTHON_BIN="python3"
fi
# C16 FIX: Bind to localhost only — external access should go through Caddy
exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 3002 --workers 1
