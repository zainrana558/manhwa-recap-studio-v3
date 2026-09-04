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
prev_err_flagged=""

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

  # newest job from the DB (+ its latest JobLog line & recency)
  jid=""; jstatus=""; jstage=""; jprog=""; jmsg=""; last_log_age=""
  if [ -f "$DB" ]; then
    row="$(sqlite3 -separator '|' "$DB" \
      "SELECT id,status,COALESCE(stage,'?'),COALESCE(progress,'?'),COALESCE(substr(message,1,70),'') FROM Job ORDER BY createdAt DESC LIMIT 1;" 2>/dev/null)"
    IFS='|' read -r jid jstatus jstage jprog jmsg <<<"$row"
    if [ -n "$jid" ]; then
      # ms-epoch of the most recent JobLog row for this job
      last_ms="$(sqlite3 "$DB" "SELECT COALESCE(MAX(createdAt),0) FROM JobLog WHERE jobId='$jid';" 2>/dev/null)"
      now_ms=$(( $(date +%s) * 1000 ))
      last_log_age=$(( (now_ms - ${last_ms%.*}) / 1000 ))
    fi
  fi

  pstr="no-job"
  if [ -n "$jid" ]; then
    pstr="job=$jid $jstatus stage=$jstage ${jprog}% logAge=${last_log_age}s"
    if [ "$jstage" != "$prev_job_stage" ] && [ -n "$prev_job_stage" ]; then
      note "job $jid stage: $prev_job_stage -> $jstage (${jprog}%) — $jmsg"
    fi
    prev_job_stage="$jstage"

    case "$jstatus" in
      done|complete|completed|error|failed|cancelled) : ;;   # terminal, no stall check
      *)
        if [ "${last_log_age:-0}" -gt "$STALL_SECS" ]; then
          key="$jid:$last_ms"
          if [ "$prev_stall_flagged" != "$key" ]; then
            alert "job $jid appears STALLED — status=$jstatus stage=$jstage ${jprog}% — no JobLog update for ${last_log_age}s (last msg: $jmsg)"
            prev_stall_flagged="$key"
          fi
        fi ;;
    esac
    if [ "$jstatus" = "error" ] || [ "$jstatus" = "failed" ]; then
      if [ "$prev_err_flagged" != "$jid" ]; then
        alert "job $jid FAILED — stage=$jstage msg: $jmsg"
        prev_err_flagged="$jid"
      fi
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
