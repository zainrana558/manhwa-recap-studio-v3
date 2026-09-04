#!/usr/bin/env bash
# Ad-hoc inspection helpers for the running job. Usage: inspect.sh <cmd> [args]
set -u
cd /home/ubuntu/manhwa-recap-studio-v3
J="${JOB_ID:-$(sqlite3 db/custom.db "SELECT id FROM Job ORDER BY createdAt DESC LIMIT 1;")}"
W="data/jobs/$J/work"
PY=.venv/bin/python3
cmd="${1:-status}"; shift || true

case "$cmd" in
status)
  sqlite3 -separator ' | ' db/custom.db "SELECT id,status,stage,progress,substr(message,1,70) FROM Job WHERE id='$J';"
  echo "slice-only proc: $(pgrep -af 'master_pipeline' | grep -c slice-only) running"
  echo "sliced chapters: $(ls -d $W/temp_slices/chap_* 2>/dev/null | wc -l)/100"
  echo "rendered chapters: $(ls $W/temp_chapters/*.mp4 2>/dev/null | wc -l)/100"
  echo "narration.json: $(find $W -name 'narration*.json' 2>/dev/null | wc -l)"
  df -h / | tail -1
  ps -o etimes,%cpu,rss,cmd -p "$(pgrep -f 'master_pipeline' | head -1)" 2>/dev/null | tail -1 | awk '{printf "proc: %ss %s%% %.0fMB\n",$1,$2,$3/1024}'
  ;;

ocrfill)  # per-chapter OCR fill rate from manifests
  $PY - "$W" <<'PY'
import json,sys,glob,os
W=sys.argv[1]
rows=[]
for mp in sorted(glob.glob(f"{W}/temp_slices/chap_*/manifest.json")):
    m=json.load(open(mp)); fr=m["frames"]
    st={}
    for f in fr: st[f.get("ocr_status","?")]=st.get(f.get("ocr_status","?"),0)+1
    wt=sum(1 for f in fr if f.get("ocr_text","").strip())
    rows.append((os.path.basename(os.path.dirname(mp)), len(fr), wt, st))
for name,n,wt,st in rows:
    print(f"{name}: {n:3d} frames  {wt:3d} w/text ({100*wt//max(n,1)}%)  {st}")
print(f"--- {len(rows)} chapters, {sum(r[1] for r in rows)} frames total, "
      f"{sum(r[2] for r in rows)} with text ({100*sum(r[2] for r in rows)//max(sum(r[1] for r in rows),1)}%)")
PY
  ;;

sheet)  # contact sheet of sliced frames for chapter $1 -> logs/inspect/
  ch=$(printf 'chap_%03d' "${1:?chapter number}")
  d="$W/temp_slices/$ch"
  out="logs/inspect"; mkdir -p "$out"
  n=$(ls "$d"/frame_*.jpg 2>/dev/null | wc -l)
  [ "$n" = 0 ] && { echo "no frames in $d"; exit 1; }
  # up to 25 frames evenly sampled, montage 5-wide
  files=$(ls "$d"/frame_*.jpg | $PY -c "import sys;L=sys.stdin.read().split();
step=max(1,len(L)//25);print(' '.join(L[::step][:25]))")
  montage $files -tile 5x -geometry 220x330+2+2 -background '#222' "$out/${ch}_sheet.jpg" 2>/dev/null \
    && echo "wrote $out/${ch}_sheet.jpg ($n frames, sampled $(echo $files|wc -w))"
  # also dump the manifest text for those frames
  $PY - "$d" <<'PY'
import json,sys,os
d=sys.argv[1]
m=json.load(open(f"{d}/manifest.json"))
for f in m["frames"]:
    t=f.get("ocr_text","").replace("\n"," ").strip()
    print(f"  {f['filename']}  [{f.get('ocr_status','?'):9s}] {t[:110]}")
PY
  ;;

frametext)  # just the OCR text stream for a chapter (what will be spoken)
  ch=$(printf 'chap_%03d' "${1:?}")
  $PY - "$W/temp_slices/$ch" <<'PY'
import json,sys
d=sys.argv[1]
m=json.load(open(f"{d}/manifest.json"))
for f in m["frames"]:
    t=(f.get("narration_text") or f.get("ocr_text") or "").replace("\n"," ").strip()
    if t: print(t)
PY
  ;;

*) echo "cmds: status | ocrfill | sheet <ch> | frametext <ch>";;
esac
