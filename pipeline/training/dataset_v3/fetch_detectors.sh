#!/bin/bash
# Free detector models for webtoon_cv2.py's §8.5 protected-row cascade.
# All public on HuggingFace. ~200 MB total. models/ is gitignored.
set -e
M="$(dirname "$0")/../../models"
HF="${HF_TOKEN:-}"
dl(){ curl -sL ${HF:+-H "Authorization: Bearer $HF"} "$1" -o "$2" && echo "  $2"; }

mkdir -p "$M/comic-text-and-bubble-detector" "$M/manga109-yolo"
# ogkalu comic-text-and-bubble-detector (RT-DETR-v2 INT8) — bubble / text / caption
[ -f "$M/comic-text-and-bubble-detector/detector-v4-s_int8.onnx" ] || \
  dl "https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector-v4-s_int8.onnx" \
     "$M/comic-text-and-bubble-detector/detector-v4-s_int8.onnx"
# deepghs/manga109_yolo (YOLOv11-m) — body / face / frame / text
dl "https://huggingface.co/deepghs/manga109_yolo/resolve/main/v2023.12.07_m_yv11/model.onnx" \
   "$M/manga109-yolo/model.onnx"
dl "https://huggingface.co/deepghs/manga109_yolo/resolve/main/v2023.12.07_m_yv11/labels.json" \
   "$M/manga109-yolo/labels.json"
echo "done."
