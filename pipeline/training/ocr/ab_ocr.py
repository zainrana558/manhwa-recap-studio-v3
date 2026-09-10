#!/usr/bin/env python3
"""A/B: Baberu OCR vs our RapidOCR (PP-OCRv5-EN) on the same crops.
Benchmark = Grok panel-vision transcripts (hard) + clean narration lines (easy).
Metrics: CER, WER, exact-match. Content-crops the composed frame first."""
import os, re, sys, json, glob, base64, subprocess, io, urllib.request
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/azureuser/manhwa-recap-studio"
JOB = "cmtug9seh000axiznhpek3nax"
OCR = "http://localhost:3002"


def norm(s):
    s = re.sub(r"[^A-Za-z0-9' ]", " ", (s or "").upper())
    return re.sub(r"\s+", " ", s).strip()

def cer(ref, hyp):
    r, h = ref, hyp
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (r[i-1] != h[j-1]))
            prev = cur
    return dp[len(h)] / max(1, len(r))

def wer(ref, hyp):
    return cer(ref.split(), hyp.split()) if ref else (0.0 if not hyp else 1.0)


def content_crop(im):
    a = np.asarray(im.convert("L"), np.int16)
    cv, rv = a.std(0), a.std(1)
    cx = np.where(cv > cv.max() * 0.45)[0]
    rx = np.where(rv > rv.max() * 0.45)[0]
    if len(cx) < 10 or len(rx) < 10:
        return im
    return im.crop((max(0, cx[0]-8), max(0, rx[0]-8),
                    min(im.width, cx[-1]+8), min(im.height, rx[-1]+8)))


def rapid_ocr(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    try:
        r = urllib.request.Request(f"{OCR}/ocr/base64",
                data=json.dumps({"image": b}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
        d = json.loads(urllib.request.urlopen(r, timeout=90).read())
        return d.get("text", "")
    except Exception as e:
        return f"[ERR {e}]"


_BAB = None
def baberu_ocr(path):
    global _BAB
    if _BAB is None:
        sys.path.insert(0, f"{ROOT}/models/baberu")
        from onnx_infer import BaberuOnnxOCR
        _BAB = BaberuOnnxOCR(f"{ROOT}/models/baberu/onnx",
                             f"{ROOT}/models/baberu/tokenizer", vision="vision_fp16.onnx")
    try:
        return _BAB(Image.open(path))
    except Exception as e:
        return f"[ERR {e}]"


# ---- build benchmark ----------------------------------------------------
bench = []   # (id, gt, crop_path)
pv = json.load(open(f"{HERE}/panel_vision.json"))
pvman = {r["id"]: r for r in json.load(open(f"{HERE}/panel_export/manifest_panels.json"))["rows"]}
os.makedirs(f"{HERE}/ab_crops", exist_ok=True)
for k, v in pv.items():
    if v.strip().upper() == "DROP" or k not in pvman:
        continue
    ch = int(k.split("-")[1]); idx = int(k.split("-")[2])
    src = f"{ROOT}/data/jobs/{JOB}/work/temp_slices/chap_{ch:03d}/frame_{idx:05d}.jpg"
    if not os.path.exists(src):
        continue
    cp = f"{HERE}/ab_crops/{k}.jpg"
    if not os.path.exists(cp):
        content_crop(Image.open(src)).convert("RGB").save(cp, quality=90)
    bench.append((k, norm(v), cp, "hard"))

# easy: clean narration lines (short, all-known-words) that never needed a fix
import random
random.seed(2)
clean_ids = []
sus = {r["id"] for r in json.load(open(f"{HERE}/round3_residual.json"))}
for nf in sorted(glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json")):
    ch = int(re.search(r"(\d+)", nf.split("/")[-2]).group(1))
    d = json.load(open(nf))
    for i, it in enumerate(d):
        t = (it.get("text") or "").strip()
        mid = "NM-%03d-%03d" % (ch, i)
        if not t or mid in sus:
            continue
        ws = t.split()
        if 4 <= len(ws) <= 14 and "-" not in t and t == t.upper():
            src = f"{ROOT}/data/jobs/{JOB}/work/temp_slices/chap_{ch:03d}/frame_{i:05d}.jpg"
            if os.path.exists(src):
                clean_ids.append((mid, norm(t), src))
random.shuffle(clean_ids)
for mid, gt, src in clean_ids[:70]:
    cp = f"{HERE}/ab_crops/{mid}.jpg"
    if not os.path.exists(cp):
        content_crop(Image.open(src)).convert("RGB").save(cp, quality=90)
    bench.append((mid, gt, cp, "easy"))

print(f"benchmark: {len(bench)} ({sum(1 for b in bench if b[3]=='hard')} hard, {sum(1 for b in bench if b[3]=='easy')} easy)")

# ---- run ---------------------------------------------------------------
res = []
for n, (mid, gt, cp, kind) in enumerate(bench):
    r = norm(rapid_ocr(cp))
    b = norm(baberu_ocr(cp))
    res.append({"id": mid, "kind": kind, "gt": gt, "rapid": r, "baberu": b,
                "cer_rapid": cer(gt, r), "cer_baberu": cer(gt, b),
                "wer_rapid": wer(gt, r), "wer_baberu": wer(gt, b),
                "exact_rapid": r == gt, "exact_baberu": b == gt})
    if (n + 1) % 20 == 0:
        print(f"  {n+1}/{len(bench)}")
json.dump(res, open(f"{HERE}/ab_results.json", "w"), indent=1)

def agg(rows, key):
    return sum(r[key] for r in rows) / max(1, len(rows))

for kind in ("hard", "easy", "all"):
    rows = res if kind == "all" else [r for r in res if r["kind"] == kind]
    if not rows:
        continue
    print(f"\n=== {kind}  (n={len(rows)}) ===")
    print(f"  CER    rapid {agg(rows,'cer_rapid'):.3f}   baberu {agg(rows,'cer_baberu'):.3f}")
    print(f"  WER    rapid {agg(rows,'wer_rapid'):.3f}   baberu {agg(rows,'wer_baberu'):.3f}")
    print(f"  exact% rapid {agg(rows,'exact_rapid')*100:.1f}   baberu {agg(rows,'exact_baberu')*100:.1f}")
