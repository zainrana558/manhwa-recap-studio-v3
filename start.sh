#!/bin/bash
# start.sh — Starts all services on Oracle Cloud VM.
# Next.js (port 3000) + pipeline-service (port 3001) + Caddy (port 80)

set -e
cd "$(dirname "$0")"

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# Make setup.sh's venv (created at $(pwd)/.venv) the one every process
# this script launches actually uses, regardless of whether the calling
# shell happened to have it activated already — see the matching block
# in start-services.sh for the full reasoning (paddleocr-service's start.sh
# and pipeline-service's spawned python3 calls both silently fall back to
# bare `python3` otherwise). Safe no-op if this venv doesn't exist yet.
if [ -x "$(pwd)/.venv/bin/python3" ]; then
    export PATH="$(pwd)/.venv/bin:$PATH"
    export VIRTUAL_ENV="$(pwd)/.venv"
    echo "  (using venv: $(pwd)/.venv)"
fi

echo "🚀 Starting Manhwa Recap Studio..."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# DEPENDENCY BOOTSTRAP
#
# Production-mode TTS (pipeline/master_pipeline.py) refuses to silently
# substitute silence when Piper and eSpeak-NG are both missing — it raises
# and fails the chapter/job instead, by design (see production plan: "never
# turn TTS failure into success by writing silence"). That's the correct
# behavior, but it means a box that never had these installed will fail
# every real job with a confusing runtime error instead of failing loudly
# and obviously at startup. Everything below is idempotent (checks before
# installing) so re-running start.sh is fast and side-effect-free once
# everything is already present.
# ═══════════════════════════════════════════════════════════════════════════
echo "🔧 Checking pipeline dependencies..."

PIPER_VOICE_DIR="$(pwd)/pipeline/voices"
PIPER_VOICE_NAME="en_US-ryan-high"
PIPER_VOICE_MODEL_PATH="$PIPER_VOICE_DIR/${PIPER_VOICE_NAME}.onnx"

# --- System packages: eSpeak-NG (TTS fallback) + Tesseract (OCR fallback) ---
_missing_pkgs=()
command -v espeak-ng >/dev/null 2>&1 || _missing_pkgs+=("espeak-ng")
command -v tesseract >/dev/null 2>&1 || _missing_pkgs+=("tesseract-ocr")

if [ ${#_missing_pkgs[@]} -gt 0 ]; then
    echo "  Installing missing system packages: ${_missing_pkgs[*]}"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq "${_missing_pkgs[@]}" \
            || echo "  ⚠️  apt-get install failed for one or more packages — TTS/OCR fallback tiers may be degraded"
    elif command -v dnf >/dev/null 2>&1; then
        # RHEL/Oracle Linux package names differ slightly for tesseract.
        _rhel_pkgs=()
        for p in "${_missing_pkgs[@]}"; do
            [ "$p" = "tesseract-ocr" ] && _rhel_pkgs+=("tesseract") || _rhel_pkgs+=("$p")
        done
        sudo dnf install -y -q "${_rhel_pkgs[@]}" \
            || echo "  ⚠️  dnf install failed for one or more packages — TTS/OCR fallback tiers may be degraded"
    else
        echo "  ⚠️  No supported package manager found (apt/dnf) — install manually: ${_missing_pkgs[*]}"
    fi
else
    echo "  ✅ espeak-ng, tesseract-ocr already installed"
fi

# --- Piper TTS (primary production TTS engine) ---
# master_pipeline.py's _synthesize_with_piper() does `shutil.which("piper")`
# and reads the PIPER_VOICE_MODEL env var — both must be set correctly in
# THIS script's environment before pipeline-service is launched below,
# since pipeline-service spawns master_pipeline.py with `env: {...process.env}`
# (a straight passthrough of whatever it itself was started with).
#
# NOTE: this deliberately does NOT `pip install piper-tts`. piper-tts's
# native dependency (piper-phonemize) only ships wheels for Python 3.9+ —
# there is no cp38 wheel at all. On a box with an older root venv (e.g.
# Ubuntu 20.04's stock Python 3.8, as opposed to a separately-built 3.11
# venv some services here use), pip's resolver falls back to ancient
# piper-tts 1.1.0/1.2.0 releases and then fails with a version conflict
# between them, not a solvable dependency issue. Piper also publishes a
# fully self-contained prebuilt binary (bundles its own espeak-ng,
# libpiper_phonemize, and libonnxruntime — zero Python involved), which
# sidesteps this entirely and works regardless of which Python the box
# happens to have. That's what this installs instead.
PIPER_DIR="$(pwd)/pipeline/piper"
PIPER_BIN="$PIPER_DIR/piper/piper"
PIPER_ARCH="$(uname -m)"
case "$PIPER_ARCH" in
    x86_64)  PIPER_ASSET="piper_linux_x86_64.tar.gz" ;;
    aarch64) PIPER_ASSET="piper_linux_aarch64.tar.gz" ;;
    armv7l)  PIPER_ASSET="piper_linux_armv7l.tar.gz" ;;
    *)       PIPER_ASSET="" ;;
esac

if [ -x "$PIPER_BIN" ]; then
    echo "  ✅ piper binary already installed at $PIPER_BIN"
elif [ -z "$PIPER_ASSET" ]; then
    echo "  ⚠️  No prebuilt Piper release for architecture '$PIPER_ARCH' — will fall back to eSpeak-NG at render time"
else
    echo "  Downloading Piper prebuilt binary ($PIPER_ASSET)..."
    mkdir -p "$PIPER_DIR"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/${PIPER_ASSET}"
    if curl -fsSL -o "$PIPER_DIR/piper.tar.gz" "$PIPER_URL" \
        && tar -xzf "$PIPER_DIR/piper.tar.gz" -C "$PIPER_DIR" \
        && rm -f "$PIPER_DIR/piper.tar.gz"; then
        echo "  ✅ Piper binary installed"
    else
        echo "  ⚠️  Piper binary download/extract failed — will fall back to eSpeak-NG at render time"
    fi
fi

if [ ! -f "$PIPER_VOICE_MODEL_PATH" ]; then
    echo "  Downloading Piper voice model ($PIPER_VOICE_NAME)..."
    mkdir -p "$PIPER_VOICE_DIR"
    PIPER_VOICE_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
    if curl -fsSL -o "$PIPER_VOICE_MODEL_PATH" "$PIPER_VOICE_BASE_URL/${PIPER_VOICE_NAME}.onnx" \
        && curl -fsSL -o "$PIPER_VOICE_MODEL_PATH.json" "$PIPER_VOICE_BASE_URL/${PIPER_VOICE_NAME}.onnx.json"; then
        echo "  ✅ Piper voice model downloaded"
    else
        echo "  ⚠️  Piper voice model download failed — will fall back to eSpeak-NG at render time"
        rm -f "$PIPER_VOICE_MODEL_PATH" "$PIPER_VOICE_MODEL_PATH.json"
    fi
else
    echo "  ✅ Piper voice model already present: $PIPER_VOICE_MODEL_PATH"
fi

# Make piper's binary reachable and tell master_pipeline.py which model to
# use, for every process this script launches from here on. piper's own
# bundled shared libraries (libespeak-ng.so, libonnxruntime.so, etc.) live
# alongside the binary in the same directory, so no LD_LIBRARY_PATH needed
# — the binary was built with a relative rpath for exactly this layout.
if [ -x "$PIPER_BIN" ]; then
    export PATH="$(dirname "$PIPER_BIN"):$PATH"
fi
if [ -f "$PIPER_VOICE_MODEL_PATH" ]; then
    export PIPER_VOICE_MODEL="$PIPER_VOICE_MODEL_PATH"
fi

echo ""

# Kill any existing processes
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 3002/tcp 2>/dev/null || true
pkill -f "pipeline-service" 2>/dev/null || true
pkill -f "index.ts" 2>/dev/null || true
sleep 2

# C8/C9/C16 FIX: Use dynamic project dir instead of hardcoded path
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# SHARED PIPELINE_SECRET
#
# pipeline-service's /internal/* endpoints and its socket.io connection both
# require a PIPELINE_SECRET (see mini-services/pipeline-service/index.ts,
# checkAuth). If unset, pipeline-service generates a random one at startup
# that nothing else knows — the Next.js API routes (jobs POST/cancel) and
# the browser's socket.io client both need the SAME value or every request
# is silently rejected with 401 (which the Next.js side then reported as
# "Pipeline service is not running", even though the service was up).
# Generate once and persist so it's stable across restarts, and export it
# for every process this script launches from here on: PIPELINE_SECRET for
# the two Node/Python server processes, and NEXT_PUBLIC_PIPELINE_SECRET
# (same value) so it gets baked into the browser bundle at build time below.
# ═══════════════════════════════════════════════════════════════════════════
SECRET_FILE="$PROJECT_DIR/.pipeline-secret"
if [ ! -s "$SECRET_FILE" ]; then
    echo "🔑 Generating pipeline secret (first run)..."
    head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32 > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
fi
export PIPELINE_SECRET="$(cat "$SECRET_FILE")"
export NEXT_PUBLIC_PIPELINE_SECRET="$PIPELINE_SECRET"

# Start paddleocr-service (port 3002)
echo "▶ Starting paddleocr-service (port 3002)..."
cd mini-services/paddleocr-service
nohup bash start.sh > "$LOG_DIR/paddleocr.log" 2>&1 &
PADDLEOCR_PID=$!
cd ../..
sleep 3

# Start pipeline-service (port 3001)
echo "▶ Starting pipeline-service (port 3001)..."
cd mini-services/pipeline-service
nohup bun run start > "$LOG_DIR/pipeline.log" 2>&1 &
PIPELINE_PID=$!
cd ../..

# Start Next.js (port 3000)
echo "▶ Starting Next.js (port 3000)..."
# Clear stale .next build cache to fix "Failed to find Server Action" errors.
# The production build may contain action IDs from a previous code version;
# rebuilding ensures the action registry matches current source. Checking
# only "does server.js exist at all" misses the common case here — a
# `git pull` that changed source without rebuilding — so compare against
# the checked-out commit instead: any commit change means the build must
# be regenerated, not just "does a build exist at all".
CURRENT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo "")"
BUILD_STAMP_FILE=".next/standalone/.build-commit"
NEED_BUILD=0
if [ ! -f ".next/standalone/server.js" ]; then
    NEED_BUILD=1
    echo "  ⚠️  .next/standalone/server.js missing — building..."
elif [ -n "$CURRENT_COMMIT" ] && [ "$(cat "$BUILD_STAMP_FILE" 2>/dev/null)" != "$CURRENT_COMMIT" ]; then
    NEED_BUILD=1
    echo "  ⚠️  .next build is stale (source changed since last build) — rebuilding..."
fi
if [ "$NEED_BUILD" = "1" ]; then
    rm -rf .next
    bun run build > "$LOG_DIR/nextjs-build.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "  ❌ Next.js build failed — check $LOG_DIR/nextjs-build.log"
    else
        echo "  ✅ Next.js build succeeded"
        [ -n "$CURRENT_COMMIT" ] && mkdir -p "$(dirname "$BUILD_STAMP_FILE")" && echo "$CURRENT_COMMIT" > "$BUILD_STAMP_FILE"
    fi
fi
# H2 FIX: Bind Next.js to localhost only (Caddy proxies externally)
HOSTNAME=127.0.0.1 nohup bun .next/standalone/server.js > "$LOG_DIR/nextjs.log" 2>&1 &
NEXT_PID=$!

# Wait for services to start
sleep 5

# Check if they're running
if curl -s http://localhost:3001/internal/health | grep -q "ok"; then
    echo "✅ Pipeline-service is running (PID: $PIPELINE_PID)"
else
    echo "⚠️  Pipeline-service may not be ready yet (check pipeline.log)"
fi

if curl -s http://localhost:3002/health | grep -q "\"status\":\"ok\""; then
    echo "✅ PaddleOCR-service is running (PID: $PADDLEOCR_PID)"
else
    echo "⚠️  PaddleOCR-service may not be ready yet (check paddleocr.log)"
fi

if curl -s -o /dev/null -w "" http://localhost:3000/ 2>/dev/null; then
    echo "✅ Next.js is running (PID: $NEXT_PID)"
else
    echo "⚠️  Next.js may not be ready yet (check nextjs.log)"
fi

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_VM_IP")
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎉  Manhwa Recap Studio is running!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  🌐 Website:  http://$PUBLIC_IP"
echo "  📊 API:      http://$PUBLIC_IP/api/stats"
echo "  🔧 Pipeline: http://$PUBLIC_IP:3001/internal/health"
# Watchdog: everything above only starts each service once — nothing was
# watching them afterward, so a native crash (the PaddleOCR PIR-interpreter
# SIGSEGV in particular — see mini-services/paddleocr-service/main.py) took
# OCR down permanently until someone noticed and reran this script by hand.
# Restart it in place so OCR — and the other two services — recover
# automatically within seconds instead.
pkill -f "watchdog.sh" 2>/dev/null
WATCHDOG_LOG_DIR="$LOG_DIR" setsid bash "$PROJECT_DIR/watchdog.sh" > "$LOG_DIR/watchdog.log" 2>&1 < /dev/null &
echo "✅ Watchdog started — auto-restarts any of the 3 services if it dies (see $LOG_DIR/watchdog.log)"

echo ""
echo "  To stop:     fuser -k 3000/tcp; pkill -f 'index.ts'; pkill -f 'watchdog.sh'"
echo "  To restart:  bash start.sh"
echo "  Logs:        tail -f $LOG_DIR/nextjs.log $LOG_DIR/pipeline.log $LOG_DIR/paddleocr.log"
echo ""
echo "═══════════════════════════════════════════════════════════════"
