#!/usr/bin/env bash
# Start all 3 services for Manhwa Recap Studio
# Usage: bash start-services.sh

PROJECT_DIR="$HOME/manhwa-recap-studio-v3"
LOG_DIR="/tmp"

kill_service() {
  # Plain `sleep 1` after pkill isn't reliable: uvicorn installs a SIGTERM
  # handler for graceful shutdown, so a process mid-request (or mid-init,
  # once past module load) can take longer than 1s to actually release its
  # port -- the next step then binds too early, the new process dies on
  # "address already in use", and the OLD process silently keeps serving
  # (exactly what happened on this box: health checks kept answering from
  # a stale process while the new one's log showed nothing but a bind
  # error). Poll until the process is actually gone instead of guessing.
  local pattern="$1"
  pkill -f "$pattern" 2>/dev/null
  for _ in $(seq 1 20); do
    pgrep -f "$pattern" > /dev/null || return 0
    sleep 0.5
  done
  # Still alive after 10s of SIGTERM -- force it so we don't proceed with
  # two processes fighting over the same port.
  pkill -9 -f "$pattern" 2>/dev/null
  sleep 1
}

kill_port() {
  # kill_service matches by process command-line pattern, which silently
  # misses anything started with a slightly different invocation (a
  # different interpreter path, an old checkout, a leftover from before a
  # refactor). Whatever is actually bound to the port is unambiguous --
  # use that as the ground truth instead of guessing a name pattern.
  local port="$1"
  local pids
  pids=$(ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" | grep -oP 'pid=\K[0-9]+' | sort -u)
  if [ -n "$pids" ]; then
    echo "  (killing stale process(es) on port $port: $pids)"
    kill -9 $pids 2>/dev/null
  fi
  for _ in $(seq 1 20); do
    ss -ltn 2>/dev/null | grep -qE "[:.]${port}[[:space:]]" || return 0
    sleep 0.5
  done
}

echo "=== Starting Manhwa Recap Studio services ==="

# 1. PaddleOCR (port 3002)
kill_service "python3 main.py"
kill_port 3002
echo "[1/3] Starting PaddleOCR on port 3002..."
cd "$PROJECT_DIR/mini-services/paddleocr-service"
setsid python3 main.py > "$LOG_DIR/paddleocr.log" 2>&1 < /dev/null &
# Poll for real readiness instead of a fixed sleep + a substring grep that
# matches "ready":false as happily as "ready":true. Model init (with the
# self-healing kwarg retries, or a cold model download) can legitimately
# take longer than a fixed 8s, so give it a real timeout instead of a guess.
ready=0
for _ in $(seq 1 60); do
  if curl -s http://localhost:3002/health | grep -q '"ready":[[:space:]]*true'; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" = "1" ]; then
  echo "  ✅ PaddleOCR ready"
else
  echo "  ❌ PaddleOCR failed - check $LOG_DIR/paddleocr.log"
fi

# 2. Pipeline service (port 3001)
kill_service "bun index.ts"
kill_port 3001
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
kill_port 3000
echo "[3/3] Starting Next.js on port 3000..."
cd "$PROJECT_DIR"
setsid bun .next/standalone/server.js > "$LOG_DIR/nextjs.log" 2>&1 < /dev/null &
# Old check did `head -c 5 | grep -q DOCTYPE`: "<!DOCTYPE html>" truncated to
# 5 bytes is "<!DOC" -- grep can never find the 7-byte string "DOCTYPE"
# inside a 5-byte string. This check was reporting failure unconditionally,
# regardless of whether Next.js was actually up. Check the real HTTP status
# instead, with a poll loop since a cold Turbopack start can take a few
# seconds.
ready=0
for _ in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/ 2>/dev/null)
  if [ "$code" = "200" ]; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" = "1" ]; then
  echo "  ✅ Next.js running"
else
  echo "  ❌ Next.js failed - check $LOG_DIR/nextjs.log"
fi

echo ""
echo "=== All services started ==="
echo "Logs:  tail -f $LOG_DIR/pipeline.log"
echo "        tail -f $LOG_DIR/paddleocr.log"
