#!/usr/bin/env bash
# Compact status snapshot for the newest job, every INTERVAL seconds.
# One line per tick -> becomes one live report in chat.
set -u
cd /home/ubuntu/manhwa-recap-studio-v3
DB=db/custom.db
INTERVAL="${HEARTBEAT_INTERVAL:-300}"
JID=$(sqlite3 "$DB" "SELECT id FROM Job ORDER BY createdAt DESC LIMIT 1;")
prev_prog=""; prev_stage=""; prev_ct=0; t0=$(date +%s)

while true; do
  read -r status stage prog msg < <(sqlite3 -separator '|' "$DB" \
    "SELECT status||' '||COALESCE(stage,'?')||' '||COALESCE(progress,'?')||' '||REPLACE(COALESCE(substr(message,1,70),''),' ','_') FROM Job WHERE id='$JID';" 2>/dev/null | tr '|' ' ')

  # slice/transcribe/render progress: count distinct chapter dirs done
  sliced=$(ls -d data/jobs/$JID/work/*/ 2>/dev/null | wc -l)
  panels=$(find data/jobs/$JID/work -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)
  logct=$(sqlite3 "$DB" "SELECT COUNT(*) FROM JobLog WHERE jobId='$JID';" 2>/dev/null)
  warns=$(sqlite3 "$DB" "SELECT COUNT(*) FROM JobLog WHERE jobId='$JID' AND level IN ('warn','error');" 2>/dev/null)
  findings=$(grep -c '^### \[' logs/findings-report.md 2>/dev/null); findings=${findings:-0}

  last_log_age=$(( ($(date +%s)*1000 - $(sqlite3 "$DB" "SELECT COALESCE(MAX(createdAt),0) FROM JobLog WHERE jobId='$JID';" 2>/dev/null | cut -d. -f1)) / 1000 ))
  disk=$(df --output=pcent / | tail -1 | tr -dc 0-9)
  load=$(cut -d' ' -f1 /proc/loadavg)
  memfree=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  el=$(( ($(date +%s) - t0) / 60 ))

  delta=""
  [ -n "$prev_prog" ] && [ "$logct" != "$prev_ct" ] && delta=" (+$((logct - prev_ct)) log lines/${INTERVAL}s)"
  prev_ct=$logct; prev_prog=$prog; prev_stage=$stage

  echo "$(date -u +%H:%M)Z | ${status}/${stage} ${prog}% | work-dirs:${sliced} panels:${panels} | logs:${logct} warn/err:${warns} findings:${findings} | lastlog ${last_log_age}s ago | disk ${disk}% load ${load} memfree ${memfree}G | +${el}m${delta}"

  st=$(sqlite3 "$DB" "SELECT status FROM Job WHERE id='$JID';" 2>/dev/null || true)
  case "$st" in done|complete|completed|error|failed|cancelled) echo "### TERMINAL: $st"; exit 0 ;; esac
  sleep "$INTERVAL"
done
