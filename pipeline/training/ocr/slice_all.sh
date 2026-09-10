#!/usr/bin/env bash
# Run master_pipeline --slice-only on every scraped title -> frames + narration.json
# (RapidOCR + per-series correction dict) under slice_work/<title>/temp_slices/.
set -u
ROOT=/home/azureuser/manhwa-recap-studio
PY=$ROOT/.venv/bin/python3
SP=/tmp/claude-1000/-home-azureuser/88ec5773-0abc-4c6b-afe1-a01b0a9ee495/scratchpad
WORK=$SP/slice_work
mkdir -p "$WORK"
export RECAP_ORT_THREADS=3

declare -A TITLE=(
  [mounthua]="Return of the Mount Hua Sect"
  [northernblade]="Legend of the Northern Blade"
  [orv]="Omniscient Reader's Viewpoint"
  [sssclass]="SSS-Class Suicide Hunter"
  [damnreinc]="Damn Reincarnation"
  [reaper]="Reaper of the Drifting Moon"
  [greatmage]="The Great Mage Returns After 4000 Years"
  [breaker]="The Breaker: Eternal Force"
  [tbate]="The Beginning After the End"
  [lookism]="Lookism"
  [eleceed]="Eleceed"
)

for t in "${!TITLE[@]}"; do
  src="$ROOT/data/ocr_train_raw/$t"
  [ -d "$src" ] || { echo "SKIP $t (not scraped)"; continue; }
  nch=$(find "$src" -maxdepth 1 -type d -name 'chapter_*' | wc -l)
  [ "$nch" -eq 0 ] && { echo "SKIP $t (0 chapters)"; continue; }
  echo "=== SLICE $t ($nch chapters) — ${TITLE[$t]} ==="
  mkdir -p "$WORK/$t"
  timeout 5400 "$PY" "$ROOT/pipeline/master_pipeline.py" \
    --input-dir "$src" \
    --output "$WORK/$t/out.mp4" \
    --work-dir "$WORK/$t" \
    --voice en-US-AndrewNeural \
    --narration-provider none \
    --recap-title "${TITLE[$t]}" \
    --production-mode --slice-only --keep-temp \
    > "$WORK/$t.log" 2>&1
  echo "  exit $? — frames: $(find "$WORK/$t/temp_slices" -name 'frame_*.jpg' 2>/dev/null | wc -l)"
done
echo "ALL SLICING DONE"
find "$WORK" -name 'frame_*.jpg' | wc -l
