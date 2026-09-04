#!/usr/bin/env bash
# Emits one line per *notable* JobLog event for the newest job:
# stage transitions, real errors/fallbacks, chapter-level milestones, and the
# terminal state. Deliberately ignores routine "scraped: N images" chatter.
set -u
cd /home/ubuntu/manhwa-recap-studio-v3
DB=db/custom.db
JID=$(sqlite3 "$DB" "SELECT id FROM Job ORDER BY createdAt DESC LIMIT 1;")
echo "live_watch: job $JID"
last=$(sqlite3 "$DB" "SELECT COALESCE(MAX(createdAt),0) FROM JobLog WHERE jobId='$JID';")
laststage=""

ERR='Traceback|Exception|SegFault|SIGSEGV|Segmentation fault|\bKilled\b|FATAL|std::bad_alloc|MemoryError|moov atom not found|Invalid data found|Conversion failed|no valid frames|0 panels|no panels|faithfulness|reverted to source|Tesseract fallback|VLM fallback|PaddleOCR (failed|unavailable)|edge-tts.*(429|failed)|piper.*failed|could not (fetch|download|open)|failed to (spawn|render|write|open)|render (failed|aborted)|\b0 images\b'
MILESTONE='Chapter [0-9]+/[0-9]+ (transcribed|sliced|rendered)|master_recap|Pipeline complete|Merging|merged|Rendering chapter|BGM|writing final|intro card'

while true; do
  rows=$(sqlite3 -separator $'\t' "$DB" \
    "SELECT createdAt,level,stage,message FROM JobLog WHERE jobId='$JID' AND createdAt>$last ORDER BY createdAt ASC;" 2>/dev/null || true)
  if [ -n "$rows" ]; then
    while IFS=$'\t' read -r ts lvl stg msg; do
      [ -z "$ts" ] && continue
      last=$ts
      if [ "$lvl" = "error" ] || printf '%s' "$msg" | grep -qE "$ERR"; then
        echo "[!] ${stg}/${lvl}: ${msg:0:220}"
      elif [ -n "$stg" ] && [ "$stg" != "$laststage" ]; then
        [ -n "$laststage" ] && echo "== stage ${laststage} -> ${stg} :: ${msg:0:120}"
        laststage=$stg
      elif printf '%s' "$msg" | grep -qE "$MILESTONE"; then
        echo "${stg}: ${msg:0:180}"
      fi
    done <<< "$rows"
  fi
  st=$(sqlite3 "$DB" "SELECT status FROM Job WHERE id='$JID';" 2>/dev/null || true)
  case "$st" in
    done|complete|completed) echo "### JOB DONE (status=$st)"; exit 0 ;;
    error|failed)            echo "### JOB FAILED (status=$st)"; exit 1 ;;
    cancelled)               echo "### JOB CANCELLED"; exit 0 ;;
  esac
  sleep 60
done
