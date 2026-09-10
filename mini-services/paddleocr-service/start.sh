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
# Load the project .env so per-box tuning + optional keys reach THIS service.
# bun/uvicorn's callers read .env on their own; this pure-Python launcher never
# did, so anything set ONLY in .env (GEMINI_API_KEY for the OCR VLM fallback
# tier, OCR_* / RAPIDOCR_* knobs) was invisible here. Export every assignment;
# `.` runs the file so comments/blanks are handled by the shell. PATH is
# preserved — .env ships its own PATH= line for the systemd-spawned pipeline
# and it must not shadow the venv/bin this script's parent prepended.
ENV_FILE="$SCRIPT_DIR/../../.env"
if [ -f "$ENV_FILE" ]; then
    _SAVED_PATH="$PATH"
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
    export PATH="$_SAVED_PATH"
    unset _SAVED_PATH
fi

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
