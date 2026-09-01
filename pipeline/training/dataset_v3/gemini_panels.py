#!/usr/bin/env python3
"""Ask Gemini 2.5 Flash for the panel bounding boxes on one comic page.

Gemini returns boxes as [ymin, xmin, ymax, xmax] normalised to 0-1000.
We convert to pixel xyxy. Handles tall webtoon strips by tiling.
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request

KEY_FILE = os.path.join(os.path.dirname(__file__), ".gemini_key")
API_KEY = (open(KEY_FILE).read().strip() if os.path.exists(KEY_FILE)
           else os.environ.get("GEMINI_API_KEY", ""))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = (
    "This is one page from a full-colour Korean webtoon / manhwa. Panels are "
    "often BORDERLESS (no drawn frame) and separated only by white/blank gutter "
    "space or a change of scene. "
    "Detect every distinct STORY PANEL — one bounding box per panel a reader "
    "would perceive as a single beat. Rules:\n"
    "- Do NOT include the blank gutter space between panels.\n"
    "- A speech bubble belongs to the panel it overlaps; do not box bubbles "
    "separately.\n"
    "- If two drawings share one continuous background with no gutter, that is "
    "ONE panel.\n"
    "- Ignore chapter-title cards, credits, author notes and ads.\n"
    "- Order boxes top-to-bottom.\n"
    'Return ONLY a JSON array, each item {"box_2d":[ymin,xmin,ymax,xmax],'
    '"kind":"panel"} with coordinates normalised 0-1000. No prose.'
)


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _mime(path):
    p = path.lower()
    return ("image/png" if p.endswith(".png") else
            "image/webp" if p.endswith(".webp") else "image/jpeg")


def call_gemini(img_path, retries=4):
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": _mime(img_path), "data": _b64(img_path)}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192,
                             "responseMimeType": "application/json"},
    }
    data = json.dumps(body).encode()
    last = None
    for att in range(retries):
        try:
            req = urllib.request.Request(
                f"{URL}?key={API_KEY}", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.load(r)
            txt = j["candidates"][0]["content"]["parts"][0]["text"]
            m = re.search(r"\[.*\]", txt, re.S)
            arr = json.loads(m.group(0) if m else txt)
            out = []
            for it in arr:
                b = it.get("box_2d") or it.get("box") or it.get("bbox")
                if not b or len(b) != 4:
                    continue
                out.append({"box_2d": [float(x) for x in b],
                            "kind": it.get("kind", "panel")})
            return out, j.get("usageMetadata", {})
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]}"
            if e.code == 429:
                time.sleep(30 * (att + 1))
            elif e.code >= 500:
                time.sleep(5 * (att + 1))
            else:
                break
        except Exception as e:
            last = str(e)
            time.sleep(3 * (att + 1))
    raise RuntimeError(f"gemini failed: {last}")


if __name__ == "__main__":
    import cv2
    path = sys.argv[1]
    im = cv2.imread(path)
    H, W = im.shape[:2]
    boxes, usage = call_gemini(path)
    print(f"{path}  {W}x{H}  -> {len(boxes)} panels   tokens={usage}")
    for b in boxes:
        y1, x1, y2, x2 = b["box_2d"]
        px = [int(x1 / 1000 * W), int(y1 / 1000 * H),
              int(x2 / 1000 * W), int(y2 / 1000 * H)]
        print(f"  {b['kind']:6s} xyxy={px}")
        cv2.rectangle(im, (px[0], px[1]), (px[2], px[3]), (0, 0, 255), 3)
    out = "/tmp/gemini_panels_overlay.jpg"
    cv2.imwrite(out, im)
    print(f"overlay -> {out}")
