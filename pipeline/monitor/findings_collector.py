#!/usr/bin/env python3
"""Durable findings collector for an unattended recap run.

Mines the JobLog table (every master_pipeline.py stdout/stderr line lands there
via emitLog) + logs/monitor.log, classifies anything that looks like an error,
bug, fallback, or limitation, and keeps a de-duplicated running report at
logs/findings-report.md. On job completion it probes the output video and
extracts sample frames so the visual analysis + final summary can be produced
later even if nothing interactive was watching at the moment it finished.

Runs as recap-findings.service (systemd, Restart=always). Safe to restart:
state is checkpointed in logs/.findings_state.json.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "db" / "custom.db"
LOGS = ROOT / "logs"
STATE = LOGS / ".findings_state.json"
JSONL = LOGS / "findings.jsonl"
REPORT = LOGS / "findings-report.md"
MONITOR_LOG = LOGS / "monitor.log"
DONE_MARK = LOGS / "JOB_DONE.json"
VIDEO_SAMPLE = LOGS / "video-sample"
INTERVAL = int(os.environ.get("FINDINGS_INTERVAL", "60"))

# ---- classification rules: (category, severity, compiled regex) ------------- #
RULES = [
    ("crash",          "high", re.compile(r"Traceback \(most recent call last\)|Segmentation fault|SIGSEGV|core dumped|MemoryError|\bKilled\b|OOMKilled|CUDA out of memory|std::bad_alloc", re.I)),
    ("exception",      "high", re.compile(r"\b\w*(Error|Exception)\b: \S| raise \w+Error|unhandled exception|\bpanic:", re.I)),
    ("render_fail",    "high", re.compile(r"ffmpeg.*(error|failed)|moov atom not found|Invalid data found|Conversion failed|non-monotonous|render (failed|aborted)|Output file .* empty", re.I)),
    ("scrape_fail",    "high", re.compile(r"scrape.*(failed|error|abort)|\bno images\b|\b0 images\b|download.*(failed|403|404|timeout)|could not (fetch|download)|giving up on chapter", re.I)),
    ("ocr_fallback",   "med",  re.compile(r"Tesseract fallback|VLM fallback|PaddleOCR (failed|unavailable|down|error)|OCR.*(failed|low quality|empty|retry)|falling back to", re.I)),
    ("narration_guard","med",  re.compile(r"faithfulness|drift(ed)?|hallucinat|expansion ratio|dropped content|reverted to (cleaned )?source|narration.*(fallback|rejected)", re.I)),
    ("slice_issue",    "med",  re.compile(r"no panels|0 panels|panel detection.*(failed|empty)|split.*failed|slice.*(failed|skipped)|degenerate box|no valid frames", re.I)),
    ("tts_issue",      "med",  re.compile(r"piper.*(failed|missing)|espeak.*(failed|error)|edge-tts.*(failed|429|timeout)|synthesi[sz].*(failed|empty)|silence fallback|voice.*not found", re.I)),
    ("empty_output",   "med",  re.compile(r"empty narration|no narration|nothing to render|no segments|skipping chapter|chapter.*(empty|no content)", re.I)),
    ("retry",          "low",  re.compile(r"\bretry(ing)?\b|attempt \d+/|re-?attempt|backing off", re.I)),
    ("deprecation",    "low",  re.compile(r"DeprecationWarning|FutureWarning|will be removed in a future", re.I)),
    ("perf",           "info", re.compile(r"(complete|finished|rendered|sliced|transcribed).{0,40}in\s+[\d.]+\s*(s|sec|min|m)\b|took\s+[\d.]+\s*(s|min)", re.I)),
]

# "Chapter N/M scraped: K images" — parsed in code, not by the rules above.
SCRAPED_RE = re.compile(r"Chapter\s+(\d+)/(\d+)\s+scraped:\s+(\d+)\s+images", re.I)

BENIGN = re.compile(
    r"%\|[\s#=>-]*\||\r|it/s\]|MB/s|"
    r"onnxruntime.*(CPUExecutionProvider|provider)|"
    r"Some weights of|You are using the default|"
    r"matplotlib|findfont|fontTools|"
    r"UserWarning: (Torch|The parameter|Named tensors)|"
    r"\[mem\] rss=",
    re.I,
)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"job_id": None, "last_ms": 0, "mon_pos": 0}


def save_state(s):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    tmp.replace(STATE)


def db_query(sql, params=()):
    for _ in range(5):
        try:
            con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
            con.row_factory = sqlite3.Row
            try:
                return con.execute(sql, params).fetchall()
            finally:
                con.close()
        except sqlite3.OperationalError:
            time.sleep(0.5)
    return []


def newest_job():
    r = db_query("SELECT id,status,stage,progress,mangaTitle,createdAt FROM Job ORDER BY createdAt DESC LIMIT 1")
    return dict(r[0]) if r else None


def sig_of(msg):
    s = re.sub(r"\d+", "N", msg.strip())
    s = re.sub(r"\s+", " ", s)
    return s[:180]


def classify(level, msg):
    m = SCRAPED_RE.search(msg)
    if m:
        k = int(m.group(3))
        # AsuraScans serves some chapters as a handful of very tall long-strip
        # images and others pre-sliced into ~20 chunks, so a low image count is
        # not itself a failure. Only a true zero is.
        if k == 0:
            return "scrape_fail", "high"
        return None
    if BENIGN.search(msg):
        if level != "error":       # 'error' level always passes the benign filter
            return None
    for cat, sev, rx in RULES:
        if rx.search(msg):
            return cat, sev
    if level == "error":
        return "error", "high"
    if level == "warn":
        return "warn_other", "low"
    return None


FIND = {}   # sig -> record


def ingest_row(job_id, ts_ms, level, stage, msg):
    c = classify(level, msg)
    if not c:
        return
    cat, sev = c
    sig = f"{cat}:{sig_of(msg)}"
    rec = FIND.get(sig)
    if not rec:
        rec = FIND[sig] = {
            "category": cat, "severity": sev, "signature": sig_of(msg),
            "count": 0, "first": ts_ms, "last": ts_ms,
            "stages": defaultdict(int), "examples": [],
        }
    rec["count"] += 1
    rec["last"] = ts_ms
    rec["stages"][stage or "?"] += 1
    if len(rec["examples"]) < 4 and msg not in rec["examples"]:
        rec["examples"].append(msg[:400])
    with JSONL.open("a") as f:
        f.write(json.dumps({"t": now_iso(), "job": job_id, "cat": cat, "sev": sev,
                            "stage": stage, "level": level, "msg": msg[:600]}) + "\n")


def ms(v):
    try:
        return int(float(v))
    except Exception:
        # sqlite may hand back an ISO string for DATETIME columns
        try:
            return int(datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            return 0


def scan_joblog(state, job):
    rows = db_query(
        "SELECT createdAt,level,stage,message FROM JobLog WHERE jobId=? AND createdAt>? ORDER BY createdAt ASC",
        (job["id"], state["last_ms"]),
    )
    for r in rows:
        t = ms(r["createdAt"])
        ingest_row(job["id"], t, r["level"], r["stage"], r["message"])
        state["last_ms"] = max(state["last_ms"], t)
    return len(rows)


def scan_monitor(state):
    try:
        data = MONITOR_LOG.read_text(errors="replace")
    except Exception:
        return
    for line in data[state.get("mon_pos", 0):].splitlines():
        if "ALERT" in line:
            sig = f"monitor:{sig_of(line)}"
            rec = FIND.get(sig)
            if not rec:
                rec = FIND[sig] = {"category": "resource/health", "severity": "high",
                                   "signature": sig_of(line), "count": 0, "first": 0, "last": 0,
                                   "stages": defaultdict(int), "examples": []}
            rec["count"] += 1
            if len(rec["examples"]) < 4:
                rec["examples"].append(line[:400])
    state["mon_pos"] = len(data)


def scrape_stats(job_id):
    rows = db_query("SELECT message FROM JobLog WHERE jobId=? AND message LIKE '%scraped:%images%'", (job_id,))
    counts = []
    for r in rows:
        m = re.search(r"scraped:\s*(\d+)\s+images", r["message"])
        if m:
            counts.append(int(m.group(1)))
    return counts


def stage_timeline(job_id):
    rows = db_query("SELECT stage,MIN(createdAt) a,MAX(createdAt) b,COUNT(*) c FROM JobLog WHERE jobId=? GROUP BY stage ORDER BY a", (job_id,))
    out = []
    for r in rows:
        a, b = ms(r["a"]), ms(r["b"])
        out.append((r["stage"], a, b, r["c"], (b - a) / 1000.0))
    return out


def write_report(job):
    sev_order = {"high": 0, "med": 1, "low": 2, "info": 3}
    recs = sorted(FIND.values(), key=lambda r: (sev_order.get(r["severity"], 9), -r["count"]))
    counts = scrape_stats(job["id"])
    tl = stage_timeline(job["id"])
    L = []
    L.append(f"# Recap run — findings report")
    L.append("")
    L.append(f"_updated {now_iso()}_  ")
    L.append(f"**job** `{job['id']}` · **{job.get('mangaTitle','?')}** · "
             f"status **{job['status']}** · stage **{job.get('stage','?')}** · {job.get('progress','?')}%")
    L.append("")
    if DONE_MARK.exists():
        try:
            d = json.loads(DONE_MARK.read_text())
            L.append("**output video:** " + (", ".join(
                f"{k} {d[k]}" for k in ("duration_hms", "size_mb", "bitrate_kbps", "video", "audio")
                if k in d) or d.get("error") or d.get("status", "?")))
            if d.get("video_path"):
                L.append(f"`{d['video_path']}`  ")
            if d.get("frames_extracted"):
                L.append(f"{d['frames_extracted']} sample frames in `logs/video-sample/` (awaiting visual review)")
            L.append("")
        except Exception:
            pass
    if counts:
        zero = counts.count(0)
        thin = sum(1 for c in counts if 0 < c < 8)
        L.append(f"**scrape:** {len(counts)} chapters · images/chapter "
                 f"min {min(counts)} / median {sorted(counts)[len(counts)//2]} / max {max(counts)} · "
                 f"total ~{sum(counts)} source images"
                 + (f" · {thin} chapters have <8 images (often a long-strip source serving a few very tall images, not a failure)" if thin else "")
                 + (f" · ⚠ **{zero} chapters scraped 0 images**" if zero else ""))
        L.append("")
    if tl:
        L.append("**stage timeline:**")
        for stg, a, b, c, dur in tl:
            L.append(f"- `{stg}` — {c} log lines, {dur/60:.1f} min "
                     f"({datetime.fromtimestamp(a/1000, timezone.utc):%H:%M:%S}→"
                     f"{datetime.fromtimestamp(b/1000, timezone.utc):%H:%M:%S})")
        L.append("")
    tot = sum(r["count"] for r in recs if r["severity"] != "info")
    L.append(f"## {len(recs)} distinct findings · {tot} total non-info hits")
    L.append("")
    for r in recs:
        first = datetime.fromtimestamp(r["first"]/1000, timezone.utc).strftime("%H:%M:%S") if r["first"] else "?"
        last = datetime.fromtimestamp(r["last"]/1000, timezone.utc).strftime("%H:%M:%S") if r["last"] else "?"
        stgs = ", ".join(f"{k}×{v}" for k, v in sorted(r["stages"].items(), key=lambda x: -x[1]))
        L.append(f"### [{r['severity'].upper()}] {r['category']} ×{r['count']}")
        L.append(f"`{r['signature']}`  ")
        L.append(f"first {first} · last {last}" + (f" · stages: {stgs}" if stgs else ""))
        for ex in r["examples"][:3]:
            L.append(f"> {ex}")
        L.append("")
    if not recs:
        L.append("_no errors, fallbacks or anomalies detected yet._")
    REPORT.write_text("\n".join(L))


def probe_video(job_id):
    jd = ROOT / "data" / "jobs" / job_id / "output"
    mp4 = None
    for cand in ("master_recap.mp4", "recap.mp4"):
        if (jd / cand).exists():
            mp4 = jd / cand
            break
    if not mp4:
        for p in jd.glob("*.mp4"):
            mp4 = p
            break
    info = {"job_id": job_id, "ended_at": now_iso(), "video": str(mp4) if mp4 else None}
    if not mp4:
        info["error"] = "no output mp4 found"
        DONE_MARK.write_text(json.dumps(info, indent=1))
        return
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(mp4)],
            capture_output=True, text=True, timeout=60)
        meta = json.loads(pr.stdout or "{}")
        fmt = meta.get("format", {})
        vs = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
        as_ = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), {})
        dur = float(fmt.get("duration", 0))
        info.update({
            "duration_s": round(dur, 1), "duration_hms": time.strftime("%H:%M:%S", time.gmtime(dur)),
            "size_mb": round(int(fmt.get("size", 0)) / 1e6, 1),
            "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000),
            "video": f"{vs.get('codec_name')} {vs.get('width')}x{vs.get('height')} {vs.get('r_frame_rate')}",
            "audio": f"{as_.get('codec_name')} {as_.get('sample_rate')}Hz {as_.get('channels')}ch",
            "video_path": str(mp4),
        })
        # sample frames for later visual analysis
        if VIDEO_SAMPLE.exists():
            shutil.rmtree(VIDEO_SAMPLE)
        VIDEO_SAMPLE.mkdir(parents=True)
        n = 30
        for i in range(n):
            t = dur * (i + 0.5) / n
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.1f}", "-i", str(mp4),
                            "-frames:v", "1", "-q:v", "3",
                            str(VIDEO_SAMPLE / f"f{i:02d}_{int(t)}s.jpg")],
                           capture_output=True, timeout=60)
        info["frames_extracted"] = len(list(VIDEO_SAMPLE.glob("*.jpg")))
    except Exception as e:
        info["probe_error"] = str(e)
    DONE_MARK.write_text(json.dumps(info, indent=1))


def main():
    LOGS.mkdir(exist_ok=True)
    state = load_state()
    print(f"[findings] started {now_iso()} — interval {INTERVAL}s", flush=True)
    handled_done = set()
    while True:
        try:
            job = newest_job()
            if job:
                if job["id"] != state.get("job_id"):
                    # new job — reset cursors, keep FIND across for continuity if same id
                    state = {"job_id": job["id"], "last_ms": 0, "mon_pos": state.get("mon_pos", 0)}
                    FIND.clear()
                n = scan_joblog(state, job)
                scan_monitor(state)
                write_report(job)
                save_state(state)
                if job["status"] in ("done", "complete", "completed", "error", "failed", "cancelled") \
                        and job["id"] not in handled_done:
                    print(f"[findings] job {job['id']} terminal: {job['status']} — probing output", flush=True)
                    if job["status"] in ("done", "complete", "completed"):
                        probe_video(job["id"])
                    else:
                        DONE_MARK.write_text(json.dumps(
                            {"job_id": job["id"], "status": job["status"], "ended_at": now_iso()}, indent=1))
                    write_report(job)
                    handled_done.add(job["id"])
                if n:
                    print(f"[findings] +{n} log rows, {len(FIND)} distinct findings", flush=True)
        except Exception as e:
            print(f"[findings] loop error: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
