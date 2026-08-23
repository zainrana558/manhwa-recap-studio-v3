#!/usr/bin/env bash
# Start all 3 services for Manhwa Recap Studio
# Usage: bash start-services.sh

PROJECT_DIR="$HOME/manhwa-recap-studio-v3"
LOG_DIR="/tmp"

kill_service() {
  pkill -f "$1" 2>/dev/null
  sleep 1
}

echo "=== Starting Manhwa Recap Studio services ==="

# 1. PaddleOCR (port 3002)
kill_service "python3 main.py"
echo "[1/3] Starting PaddleOCR on port 3002..."
cd "$PROJECT_DIR/mini-services/paddleocr-service"
setsid python3 main.py > "$LOG_DIR/paddleocr.log" 2>&1 < /dev/null &
sleep 8
if curl -s http://localhost:3002/health | grep -q ready; then
  echo "  ✅ PaddleOCR ready"
else
  echo "  ❌ PaddleOCR failed - check $LOG_DIR/paddleocr.log"
fi

# 2. Pipeline service (port 3001)
kill_service "bun index.ts"
echo "[2/3] Starting Pipeline on port 3001..."
cd "$PROJECT_DIR/mini-services/pipeline-service"
setsid bun run start > "$LOG_DIR/pipeline.log" 2>&1 < /dev/null &
sleep 3
if pgrep -f "bun index.ts" > /dev/null; then
  echo "  ✅ Pipeline running"
else
  echo "  ❌ Pipeline failed - check $LOG_DIR/pipeline.log"
fi

# 3. Next.js (port 3000)
kill_service "server.js"
echo "[3/3] Starting Next.js on port 3000..."
cd "$PROJECT_DIR"
setsid bun .next/standalone/server.js > "$LOG_DIR/nextjs.log" 2>&1 < /dev/null &
sleep 3
if curl -s http://localhost:3000/ | head -c 5 | grep -q DOCTYPE; then
  echo "  ✅ Next.js running"
else
  echo "  ❌ Next.js failed - check $LOG_DIR/nextjs.log"
fi

echo ""
echo "=== All services started ==="
echo "Logs:  tail -f $LOG_DIR/pipeline.log"
echo "        tail -f $LOG_DIR/paddleocr.log"
