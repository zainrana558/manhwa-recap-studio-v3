#!/usr/bin/env bash
# Start all 3 services for Manhwa Recap Studio
# Usage: bash start-services.sh

PROJECT_DIR="$HOME/manhwa-recap-studio-v3"
LOG_DIR="/tmp"

# See the matching block in start.sh for why this exists: pipeline-service's
# /internal/* endpoints and its socket.io connection both require
# PIPELINE_SECRET (checkAuth in mini-services/pipeline-service/index.ts).
# Generate once and persist so the Next.js API routes and browser socket
# client (NEXT_PUBLIC_PIPELINE_SECRET, baked in at build time below) agree
# with pipeline-service on the same value across restarts.
SECRET_FILE="$PROJECT_DIR/.pipeline-secret"
if [ ! -s "$SECRET_FILE" ]; then
  echo "🔑 Generating pipeline secret (first run)..."
  head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32 > "$SECRET_FILE"
  chmod 600 "$SECRET_FILE"
fi
export PIPELINE_SECRET="$(cat "$SECRET_FILE")"
export NEXT_PUBLIC_PIPELINE_SECRET="$PIPELINE_SECRET"

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
kill_service "uvicorn main:app"
kill_service "python3 main.py"
kill_port 3002
echo "[1/3] Starting PaddleOCR on port 3002..."
cd "$PROJECT_DIR/mini-services/paddleocr-service"
# Go through this service's own start.sh, not `python3 main.py` directly —
# that nested start.sh binds uvicorn to 127.0.0.1 only ("C16 FIX": external
# access should go through Caddy). Calling main.py directly binds 0.0.0.0
# (see its own `if __name__ == "__main__"` block), exposing OCR publicly
# with no auth.
setsid bash start.sh > "$LOG_DIR/paddleocr.log" 2>&1 < /dev/null &
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

# Clear stale .next build cache to fix "Failed to find Server Action" errors.
# This happens when the production build has a server-action registry that
# no longer matches the current source (actions added/removed since last
# build). The old check only looked at whether server.js existed at all, so
# a `git pull` that changed source without rebuilding (the exact scenario
# in the reported error log) left a stale-but-present build in place and
# this check never fired. Compare against the checked-out commit instead —
# any commit change means source may have moved and the build must be
# regenerated, not just "does a build exist at all".
CURRENT_COMMIT="$(cd "$PROJECT_DIR" && git rev-parse HEAD 2>/dev/null || echo "")"
BUILD_STAMP_FILE=".next/standalone/.build-commit"
NEED_BUILD=0
if [ ! -f ".next/standalone/server.js" ]; then
  NEED_BUILD=1
  echo "  .next/standalone/server.js not found — building..."
elif [ -n "$CURRENT_COMMIT" ] && [ "$(cat "$BUILD_STAMP_FILE" 2>/dev/null)" != "$CURRENT_COMMIT" ]; then
  NEED_BUILD=1
  echo "  .next build is stale (source changed since last build) — rebuilding..."
fi
if [ "$NEED_BUILD" = "1" ]; then
  rm -rf .next
  bun run build > "$LOG_DIR/nextjs-build.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "  ⚠️  Build failed — check $LOG_DIR/nextjs-build.log"
  else
    echo "  ✅ Build succeeded"
    [ -n "$CURRENT_COMMIT" ] && mkdir -p "$(dirname "$BUILD_STAMP_FILE")" && echo "$CURRENT_COMMIT" > "$BUILD_STAMP_FILE"
  fi
fi

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

# Watchdog: this script only starts each service once and exits — nothing
# was watching them afterward, so a native crash (the PaddleOCR PIR SIGSEGV
# in particular — see mini-services/paddleocr-service/main.py) took OCR
# down permanently until someone noticed and reran this script by hand.
# Restart it in place (it self-restarts if already running) so OCR — and
# the other two services — recover automatically within seconds.
pkill -f "watchdog.sh" 2>/dev/null
WATCHDOG_LOG_DIR="$LOG_DIR" setsid bash "$PROJECT_DIR/watchdog.sh" > "$LOG_DIR/watchdog.log" 2>&1 < /dev/null &
echo "  ✅ Watchdog started — auto-restarts any of the 3 services if it dies (see $LOG_DIR/watchdog.log)"

echo ""
echo "=== All services started ==="
echo "Logs:  tail -f $LOG_DIR/pipeline.log"
echo "        tail -f $LOG_DIR/paddleocr.log"
