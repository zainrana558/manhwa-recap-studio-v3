#!/bin/bash
# =============================================================================
# start-hf.sh — Entry point for the Hugging Face Spaces Docker container.
# Launches all three services and keeps the container alive.
# =============================================================================

set -e

# Graceful shutdown: kill all child processes when the container stops.
cleanup() {
    echo "[start-hf] Shutting down services..."
    kill "$OCR_PID" "$PIPELINE_PID" "$NEXT_PID" "$CADDY_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

cd /app

# ---- 1. Initialize database (creates tables if they don't exist) -----------
echo "[start-hf] Initializing database..."
bun run db:push 2>&1 || echo "[start-hf] db:push warning (tables may already exist)"

# ---- 2. Start PaddleOCR service (port 3002, PP-OCRv5) --------------------
echo "[start-hf] Starting PaddleOCR service on port 3002..."
(cd mini-services/paddleocr-service && python3 main.py) &
OCR_PID=$!

# ---- 3. Start pipeline-service (port 3001, Python + socket.io) -------------
echo "[start-hf] Starting pipeline-service on port 3001..."
(cd mini-services/pipeline-service && bun run start) &
PIPELINE_PID=$!

# ---- 4. Start Next.js production server (port 3000) ------------------------
echo "[start-hf] Starting Next.js on port 3000..."
bun .next/standalone/server.js &
NEXT_PID=$!

# ---- 5. Wait for services to be ready --------------------------------------
echo "[start-hf] Waiting for services to initialize..."
sleep 5

# Quick health check
if curl -s -o /dev/null -w "" "http://localhost:3000/" 2>/dev/null; then
    echo "[start-hf] Next.js is responding on port 3000 ✓"
else
    echo "[start-hf] WARNING: Next.js not yet responding (may still be starting)"
fi

# ---- 6. Start Caddy reverse proxy (port 7860, main process) ----------------
echo "[start-hf] Starting Caddy on port 7860 (main process)..."
echo "[start-hf] App will be available at the Hugging Face Space URL."
caddy run --config Caddyfile.docker --adapter caddyfile &
CADDY_PID=$!

# ---- 7. Keep the container alive -------------------------------------------
# If any service crashes, the container stops and HF Spaces restarts it.
echo "[start-hf] All services started. Container is live."
echo "  - PaddleOCR:       PID $OCR_PID (port 3002)"
echo "  - Pipeline-service: PID $PIPELINE_PID (port 3001)"
echo "  - Next.js:        PID $NEXT_PID (port 3000)"
echo "  - Caddy:          PID $CADDY_PID (port 7860)"

wait
