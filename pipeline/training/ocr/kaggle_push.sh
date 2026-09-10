#!/usr/bin/env bash
# Push the OCR-training corpora to private Kaggle datasets.
# Huge image dirs (synth) are pre-zipped; the rest upload as dirs
# (kagglehub auto-zips >50 files). Training notebooks unzip on Kaggle.
set -u
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token 2>/dev/null)"
SP=/tmp/claude-1000/-home-azureuser/88ec5773-0abc-4c6b-afe1-a01b0a9ee495/scratchpad
STAGE=$SP/_kaggle_stage
PY=/home/azureuser/manhwa-recap-studio/.venv/bin/python
USER=zainrana1122
mkdir -p "$STAGE"

up() {  # slug  dir  notes
  local slug="$1" dir="$2" notes="$3"
  echo "=== $USER/$slug  ($(du -sh "$dir" 2>/dev/null|cut -f1)) ==="
  "$PY" - "$USER/$slug" "$dir" "$notes" <<'EOF'
import sys, json, os, kagglehub
handle, d, notes = sys.argv[1:4]
mf = os.path.join(d, "dataset-metadata.json")
json.dump({"title": handle.split("/")[1][:50], "id": handle,
           "licenses": [{"name": "other"}]}, open(mf, "w"), indent=1)
try:
    kagglehub.dataset_upload(handle, d, version_notes=notes)
    print("  OK", handle)
except Exception as e:
    print("  FAIL", handle, repr(e))
EOF
}

W="${1:-all}"

if [ "$W" = synth ] || [ "$W" = all ]; then
  rm -rf "$STAGE/synth"; mkdir -p "$STAGE/synth"
  echo "zipping synth bubbles (14k)..."; ( cd "$SP/synth" && zip -q -r -1 "$STAGE/synth/bubbles.zip" img labels.jsonl )
  echo "zipping rec_synth lines (40k)..."; ( cd "$SP/rec_synth" && zip -q -r -1 "$STAGE/synth/lines.zip" img train_list.txt val_list.txt )
  up manhwa-ocr-synth "$STAGE/synth" "14k bubble crops + 40k line crops (perfect labels, degraded render)"
fi

if [ "$W" = scrapes ] || [ "$W" = all ]; then
  up manhwa-ocr-raw-scrapes /home/azureuser/manhwa-recap-studio/data/ocr_train_raw \
     "9 hard manhwa x 10 chapters, raw scanlation pages (asura + mgeko)"
fi

if [ "$W" = byt5 ] || [ "$W" = all ]; then
  up nm-ocr-byt5-data "$SP/byt5_data" "ByT5 post-OCR corrector train/val/eval_real"
fi

if [ "$W" = pairs ] || [ "$W" = all ]; then
  rm -rf "$STAGE/pairs"; mkdir -p "$STAGE/pairs"
  cp "$SP/corrected_pairs.jsonl" "$SP/panel_vision.json" "$STAGE/pairs/" 2>/dev/null
  [ -f "$SP/xseries_corrector_pairs.jsonl" ] && cp "$SP/xseries_corrector_pairs.jsonl" "$STAGE/pairs/"
  up manhwa-ocr-corrector-pairs "$STAGE/pairs" "1586 nano-machine raw->clean pairs + panel-vision + xseries"
fi

if [ "$W" = frames ] || { [ "$W" = all ] && grep -q "ALL SLICING DONE" /tmp/slice_all.log 2>/dev/null; }; then
  rm -rf "$STAGE/frames"; mkdir -p "$STAGE/frames"
  ( cd "$SP/slice_work" && for t in */; do
      [ -d "${t}temp_slices" ] && echo "zip ${t%/}..." && zip -q -r -1 "$STAGE/frames/${t%/}.zip" "${t}temp_slices"
    done )
  up manhwa-ocr-slice-frames "$STAGE/frames" "1920x1080 canonical panel frames + manifests (RapidOCR baseline in manifest.ocr_text)"
fi

if [ "$W" = qwen ]; then
  [ -d "$SP/qwen_out" ] && up manhwa-ocr-qwen-batches "$SP/qwen_out" "curated Qwen transcription batches + INDEX.json"
fi
echo "kaggle_push done ($W)"
