#!/usr/bin/env bash
# Unattended health + job monitor for a long recap run.
#
# Observes (does NOT own recovery — watchdog.sh + systemd do that):
#   - the 3 app services, caddy, and the public URL
#   - the newest DB job: status / stage / progress / chapter / staleness
#   - disk, load, memory
#   - watchdog.sh liveness (re-launches it if it died)
#
# Writes one status line per cycle to logs/monitor.log and anything abnormal
# to logs/monitor-alerts.log (which is what a check-in should read first).
#
#   setsid bash pipeline/monitor/job_monitor.sh > /dev/null 2>&1 &

set -u
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/monitor.log"
ALERTS="$LOG_DIR/monitor-alerts.log"
DB="$PROJECT_DIR/db/custom.db"
INTERVAL="${MONITOR_INTERVAL:-120}"
PUBLIC_URL="${MONITOR_PUBLIC_URL:-http://80.225.248.230/}"
STALL_SECS="${MONITOR_STALL_SECS:-1800}"   # job updated_at older than this = stalled

ts() { date -u +%FT%TZ; }
note() { echo "$(ts) $*" >> "$LOG"; }
alert() { echo "$(ts) ALERT $*" | tee -a "$ALERTS" >> "$LOG"; }

declare -A down_streak
prev_job_stage=""
prev_stall_flagged=""

http() { curl -s -m 8 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null; }

check_svc() {  # name url expected
  local name="$1" code
  code="$(http "$2")"
  if [ "$code" = "$3" ]; then
    if [ "${down_streak[$name]:-0}" -ge 3 ]; then alert "$name RECOVERED (code $code)"; fi
    down_streak[$name]=0
  else
    down_streak[$name]=$(( ${down_streak[$name]:-0} + 1 ))
    if [ "${down_streak[$name]}" -ge 2 ]; then
      alert "$name DOWN (code $code, streak ${down_streak[$name]})"
    fi
  fi
  echo "$code"
}

note "monitor started (interval ${INTERVAL}s, stall ${STALL_SECS}s, public ${PUBLIC_URL})"

while true; do
  c_next="$(check_svc nextjs   http://localhost:3000/ 200)"
  c_pipe="$(check_svc pipeline http://localhost:3001/internal/health 200)"
  c_ocr="$(check_svc  ocr      http://localhost:3002/health 200)"
  c_pub="$(check_svc  public   "$PUBLIC_URL" 200)"

  # watchdog.sh liveness
  if ! pgrep -f "watchdog.sh" >/dev/null 2>&1; then
    alert "watchdog.sh not running — relaunching"
    (cd "$PROJECT_DIR" && setsid bash watchdog.sh > "$LOG_DIR/watchdog.log" 2>&1 < /dev/null &)
  fi

  # newest job from the DB
  job_line=""
  if [ -f "$DB" ]; then
    job_line="$(sqlite3 "$DB" "SELECT id||'|'||status||'|'||COALESCE(mangaTitle,'?') FROM Job ORDER BY createdAt DESC LIMIT 1;" 2>/dev/null)"
  fi
  jid="${job_line%%|*}"
  jrest="${job_line#*|}"; jstatus="${jrest%%|*}"; jtitle="${jrest#*|}"
  pstr="no-job"
  if [ -n "$jid" ]; then
    pj="$PROJECT_DIR/data/jobs/$jid/progress.json"
    if [ -f "$pj" ]; then
      read -r stage prog ci tc upd msg < <(python3 - "$pj" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("stage","?"), d.get("progress","?"), d.get("chapter_index","?"),
      d.get("total_chapters","?"), d.get("updated_at",0), str(d.get("message",""))[:60].replace(" ","_"))
PY
)
      now=$(date +%s)
      age=$(( now - ${upd%.*} ))
      pstr="job=$jid $jstatus stage=$stage ${prog}% ch=${ci}/${tc} age=${age}s"
      # stall detection: only while the job looks active
      case "$jstatus" in
        running|processing|active|in_progress|queued)
          if [ "$age" -gt "$STALL_SECS" ]; then
            if [ "$prev_stall_flagged" != "$jid:$upd" ]; then
              alert "job $jid appears STALLED — status=$jstatus stage=$stage progress=${prog}% no update for ${age}s (msg: $msg)"
              prev_stall_flagged="$jid:$upd"
            fi
          fi ;;
      esac
      if [ "$stage" != "$prev_job_stage" ] && [ -n "$prev_job_stage" ]; then
        note "job $jid stage: $prev_job_stage -> $stage (${prog}%, ch ${ci}/${tc})"
      fi
      prev_job_stage="$stage"
    else
      pstr="job=$jid $jstatus (no progress.json yet)"
    fi
  fi

  # resources
  disk_pct="$(df --output=pcent / | tail -1 | tr -dc 0-9)"
  disk_free="$(df -h --output=avail / | tail -1 | tr -d ' ')"
  load1="$(cut -d' ' -f1 /proc/loadavg)"
  mem_avail_mb="$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"
  if [ "${disk_pct:-0}" -ge 90 ]; then alert "disk ${disk_pct}% full (${disk_free} free)"; fi
  if [ "${mem_avail_mb:-9999}" -lt 400 ]; then alert "low memory: ${mem_avail_mb}MB available"; fi

  note "svc next=$c_next pipe=$c_pipe ocr=$c_ocr pub=$c_pub | $pstr | disk ${disk_pct}% (${disk_free}) load ${load1} memAvail ${mem_avail_mb}MB"

  sleep "$INTERVAL"
done
