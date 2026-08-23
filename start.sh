#!/bin/bash
# start.sh — Starts all services on Oracle Cloud VM.
# Next.js (port 3000) + pipeline-service (port 3001) + Caddy (port 80)

set -e
cd "$(dirname "$0")"

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

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
PIPER_VOICE_NAME="en_US-lessac-medium"
PIPER_VOICE_MODEL_PATH="$PIPER_VOICE_DIR/${PIPER_VOICE_NAME}.onnx"
ROOT_VENV="$HOME/.venv"

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
if [ -x "$ROOT_VENV/bin/python3" ]; then
    if [ -x "$ROOT_VENV/bin/piper" ]; then
        echo "  ✅ piper-tts already installed in $ROOT_VENV"
    else
        echo "  Installing piper-tts into $ROOT_VENV..."
        "$ROOT_VENV/bin/pip" install --quiet piper-tts \
            || echo "  ⚠️  piper-tts pip install failed — will fall back to eSpeak-NG at render time"
    fi

    if [ ! -f "$PIPER_VOICE_MODEL_PATH" ]; then
        echo "  Downloading Piper voice model ($PIPER_VOICE_NAME)..."
        mkdir -p "$PIPER_VOICE_DIR"
        "$ROOT_VENV/bin/python3" -m piper.download_voices "$PIPER_VOICE_NAME" \
            --download-dir "$PIPER_VOICE_DIR" \
            || echo "  ⚠️  Piper voice download failed — will fall back to eSpeak-NG at render time"
    else
        echo "  ✅ Piper voice model already present: $PIPER_VOICE_MODEL_PATH"
    fi

    # Make piper's binary reachable and tell master_pipeline.py which model
    # to use, for every process this script launches from here on.
    export PATH="$ROOT_VENV/bin:$PATH"
    if [ -f "$PIPER_VOICE_MODEL_PATH" ]; then
        export PIPER_VOICE_MODEL="$PIPER_VOICE_MODEL_PATH"
    fi
else
    echo "  ⚠️  Root Python venv not found at $ROOT_VENV — skipping Piper setup."
    echo "     Run setup.sh first, or TTS will fall back to eSpeak-NG only."
fi

echo ""

# Kill any existing processes
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 3002/tcp 2>/dev/null || true
pkill -f "pipeline-service" 2>/dev/null || true
pkill -f "index.ts" 2>/dev/null || true
sleep 2

# Start paddleocr-service (port 3002)
echo "▶ Starting paddleocr-service (port 3002)..."
cd mini-services/paddleocr-service
nohup bash start.sh > /home/ubuntu/manhwa-recap-studio-v3/paddleocr.log 2>&1 &
PADDLEOCR_PID=$!
cd ../..
sleep 3

# Start pipeline-service (port 3001)
echo "▶ Starting pipeline-service (port 3001)..."
cd mini-services/pipeline-service
nohup bun run start > /home/ubuntu/manhwa-recap-studio-v3/pipeline.log 2>&1 &
PIPELINE_PID=$!
cd ../..

# Start Next.js (port 3000)
echo "▶ Starting Next.js (port 3000)..."
nohup bun .next/standalone/server.js > /home/ubuntu/manhwa-recap-studio-v3/nextjs.log 2>&1 &
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
echo ""
echo "  To stop:     fuser -k 3000/tcp; pkill -f 'index.ts'"
echo "  To restart:  bash start.sh"
echo "  Logs:        tail -f nextjs.log pipeline.log paddleocr.log"
echo ""
echo "═══════════════════════════════════════════════════════════════"
