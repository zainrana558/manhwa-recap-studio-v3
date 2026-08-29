#!/usr/bin/env bash
# Lightweight process watchdog for the 3 Manhwa Recap Studio services.
#
# start-services.sh only does a one-shot start with no ongoing supervision,
# so a native crash — most notably the known PaddleOCR PIR-interpreter
# SIGSEGV (see the crash-mitigation comment at the top of
# mini-services/paddleocr-service/main.py) — killed OCR permanently until a
# person noticed the pipeline was stuck and reran start-services.sh by hand.
#
# This loop polls each service and restarts it in place when it's down,
# with per-service exponential backoff so a service that's crash-looping on
# every request doesn't get restarted in a tight, CPU-burning loop.
#
# Usage: setsid bash watchdog.sh > /tmp/watchdog.log 2>&1 &
# (start-services.sh launches this automatically after all 3 services are
# confirmed up.)

set -u
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp"
CHECK_INTERVAL_SEC=15
MIN_BACKOFF_SEC=15
MAX_BACKOFF_SEC=300

declare -A backoff_sec
declare -A next_check_at

log() {
  echo "[watchdog] $(date -u +%FT%TZ) $*"
}

restart_paddleocr() {
  log "PaddleOCR (port 3002) is down — restarting"
  pkill -9 -f "python3 main.py" 2>/dev/null
  sleep 1
  (cd "$PROJECT_DIR/mini-services/paddleocr-service" && setsid python3 main.py > "$LOG_DIR/paddleocr.log" 2>&1 < /dev/null &)
}

restart_pipeline() {
  log "Pipeline service (port 3001) is down — restarting"
  pkill -9 -f "bun index.ts" 2>/dev/null
  sleep 1
  (cd "$PROJECT_DIR/mini-services/pipeline-service" && setsid bun run start > "$LOG_DIR/pipeline.log" 2>&1 < /dev/null &)
}

restart_nextjs() {
  log "Next.js (port 3000) is down — restarting"
  pkill -9 -f "server.js" 2>/dev/null
  sleep 1
  (cd "$PROJECT_DIR" && setsid bun .next/standalone/server.js > "$LOG_DIR/nextjs.log" 2>&1 < /dev/null &)
}

port_open() {
  ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]"
}

paddleocr_healthy() {
  port_open 3002 && curl -s -m 5 http://localhost:3002/health 2>/dev/null | grep -q '"ready":[[:space:]]*true'
}

pipeline_healthy() {
  # index.ts's HTTP server is a bare socket.io/internal-route handler with
  # no dedicated /health endpoint — a listening port is the available
  # liveness signal here.
  port_open 3001
}

nextjs_healthy() {
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:3000/ 2>/dev/null)" = "200" ]
}

check_and_restart() {
  local name="$1" healthy_fn="$2" restart_fn="$3"
  local now
  now=$(date +%s)
  if [ "${next_check_at[$name]:-0}" -gt "$now" ]; then
    return
  fi
  if "$healthy_fn"; then
    if [ "${backoff_sec[$name]:-0}" -gt 0 ]; then
      log "$name recovered"
    fi
    backoff_sec[$name]=0
    return
  fi
  local b="${backoff_sec[$name]:-$MIN_BACKOFF_SEC}"
  "$restart_fn"
  if [ "$b" -eq 0 ]; then b=$MIN_BACKOFF_SEC; fi
  local next_b=$(( b * 2 ))
  if [ "$next_b" -gt "$MAX_BACKOFF_SEC" ]; then next_b=$MAX_BACKOFF_SEC; fi
  backoff_sec[$name]=$next_b
  next_check_at[$name]=$(( now + b ))
}

log "started — checking every ${CHECK_INTERVAL_SEC}s (backoff ${MIN_BACKOFF_SEC}s-${MAX_BACKOFF_SEC}s per service)"

while true; do
  check_and_restart "paddleocr" paddleocr_healthy restart_paddleocr
  check_and_restart "pipeline" pipeline_healthy restart_pipeline
  check_and_restart "nextjs" nextjs_healthy restart_nextjs
  sleep "$CHECK_INTERVAL_SEC"
done
