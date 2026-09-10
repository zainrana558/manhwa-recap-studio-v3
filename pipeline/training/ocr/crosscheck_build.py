#!/usr/bin/env python3
"""Cross-check Qwen panel transcriptions against RapidOCR and build two corpora:

  1. xseries_corrector_pairs.jsonl  — {src: rapidocr_frame_text, tgt: qwen_clean, ...}
     cross-manhwa post-OCR correction pairs (feeds the ByT5 corrector, generalises
     the nano-machine-only pairs).

  2. rec_real/  — verified single-LINE crops for PP-OCRv5 rec fine-tuning:
       img/*.jpg  +  train_list.txt / val_list.txt  ("<relpath>\t<label>")
     Every label is a RapidOCR line whose text is CONFIRMED by appearing in the
     Qwen transcription (fuzzy) — so labels are trustworthy with no fragile
     word-level alignment. Unconfirmed lines -> rec_real/unmatched.tsv for an
     optional targeted second Qwen pass.

Inputs:
  qwen_out/INDEX.json                (from curate_qwen.py)
  qwen_returns/*.txt                 (paste Qwen output here: "<img> | <text>" lines)

  .venv/bin/python crosscheck_build.py [--val-frac 0.06]
"""
import os, re, sys, json, glob, csv, argparse, difflib, hashlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import garble_helpers as G

QOUT = f"{HERE}/qwen_out"
QRET = f"{HERE}/qwen_returns"
REC = f"{HERE}/rec_real"

ap = argparse.ArgumentParser()
ap.add_argument("--val-frac", type=float, default=0.06)
ap.add_argument("--no-rec", action="store_true", help="corrector pairs only, skip line-crop rebuild")
args = ap.parse_args()


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())

def clean_qwen(v):
    """Qwen segment string -> (joined_text, [segments]).  Drops SFX/markers."""
    v = v.strip()
    if not v or v.upper() in ("[NONE]", "NONE", "[BLANK]"):
        return "", []
    segs = []
    for seg in re.split(r"\s*/{3}\s*|\s*\|{2,}\s*", v):
        seg = seg.strip()
        if not seg:
            continue
        if re.fullmatch(r"\[?\s*(SFX\s*:[^\]]*|UNREADABLE|ILLEGIBLE|CREDIT|NONE)\s*\]?", seg, re.I):
            continue
        seg = re.sub(r"\[SFX\s*:\s*[^\]]*\]", "", seg, flags=re.I)
        seg = re.sub(r"\[(UNREADABLE|ILLEGIBLE|CREDIT)\]", "", seg, flags=re.I)
        seg = norm(seg.strip(' "'))
        if seg:
            segs.append(seg)
    return norm(" ".join(segs)), segs


def toks(s):
    return re.findall(r"[A-Za-z0-9']+", s.lower())


def fuzzy_contains(needle, hay):
    """best ratio of `needle` against any same-length window of `hay` token stream."""
    n, h = toks(needle), toks(hay)
    if not n:
        return 0.0, ""
    if len(n) > len(h):
        return difflib.SequenceMatcher(None, " ".join(n), " ".join(h)).ratio(), " ".join(h)
    best, bj = 0.0, 0
    for j in range(0, len(h) - len(n) + 1):
        r = difflib.SequenceMatcher(None, n, h[j:j + len(n)]).ratio()
        if r > best:
            best, bj = r, j
    return best, " ".join(h[bj:bj + len(n)])


# ---- load INDEX + Qwen returns ----------------------------------------
idx = {r["img"]: r for r in json.load(open(f"{QOUT}/INDEX.json"))}
qwen = {}
dupe = 0
for fp in sorted(glob.glob(f"{QRET}/*.txt")) + sorted(glob.glob(f"{QRET}/*.md")):
    for ln in open(fp, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*([A-Za-z0-9][\w.\-]+\.jpg)\s*[|\t]\s*(.*)$", ln.rstrip())
        if not m:
            continue
        img, val = m.group(1), m.group(2)
        if img in qwen and qwen[img] != val:
            dupe += 1
        qwen[img] = val

matched = [(img, idx[img], qwen[img]) for img in qwen if img in idx]
print(f"qwen lines: {len(qwen)}  matched to INDEX: {len(matched)}  (dupes {dupe})")
missing = sorted(set(idx) - set(qwen))
if missing:
    print(f"  still awaiting {len(missing)} frames (first: {missing[:3]})")

# ---- 1. corrector pairs ---------------------------------------------
pairs, stats = [], {"identical": 0, "rapid_empty": 0, "qwen_empty": 0, "pair": 0}
cer_r = []
for img, rec, qraw in matched:
    rtext = norm(rec["rapidocr_text"])
    qtext, qsegs = clean_qwen(qraw)
    if not qtext:
        stats["qwen_empty"] += 1
        # a real OCR hallucination on a textless panel — teach "-> empty"
        if rtext and len(toks(rtext)) >= 2:
            pairs.append({"src": rtext, "tgt": "", "kind": "halluc",
                          "title": rec["title"], "frame_id": rec["frame_id"]})
        continue
    if not rtext:
        stats["rapid_empty"] += 1
        continue
    a, b = rtext.upper(), qtext.upper()
    if re.sub(r"[^A-Z0-9]", "", a) == re.sub(r"[^A-Z0-9]", "", b):
        stats["identical"] += 1
        continue
    # character error rate rapid-vs-qwen (how wrong RapidOCR was)
    d = difflib.SequenceMatcher(None, a, b)
    cer = 1 - d.ratio()
    cer_r.append(cer)
    pairs.append({"src": rtext, "tgt": qtext, "kind": "xseries", "cer": round(cer, 3),
                  "title": rec["title"], "frame_id": rec["frame_id"],
                  "bucket": rec.get("bucket")})
    stats["pair"] += 1

with open(f"{HERE}/xseries_corrector_pairs.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"\ncorrector pairs: {len(pairs)}  {stats}")
if cer_r:
    print(f"  RapidOCR CER vs Qwen (changed lines): mean {np.mean(cer_r):.3f}  median {np.median(cer_r):.3f}")
    print(f"  by title:")
    bt = {}
    for p in pairs:
        if p["kind"] == "xseries":
            bt.setdefault(p["title"], []).append(p["cer"])
    for t, cs in sorted(bt.items(), key=lambda x: -np.mean(x[1])):
        print(f"    {t:14} n={len(cs):4}  mean CER {np.mean(cs):.3f}")

if args.no_rec:
    sys.exit(0)

# ---- 2. verified line crops for rec fine-tune ----------------------
os.makedirs(f"{REC}/img", exist_ok=True)
try:
    from rapidocr import RapidOCR
    _eng = RapidOCR()
except Exception as e:
    print("rapidocr unavailable, skipping rec build:", e); sys.exit(0)

rows, unmatched = [], []
seenhash = set()
for n, (img, rec, qraw) in enumerate(matched):
    qtext, qsegs = clean_qwen(qraw)
    if not qtext:
        continue
    fpath = rec.get("path") or ""
    if not os.path.exists(fpath):
        cand = glob.glob(f"{HERE}/slice_work/{rec['title']}/temp_slices/*/{rec['frame_id'].split('_')[-1]}.jpg")
        fpath = cand[0] if cand else ""
    if not os.path.exists(fpath):
        continue
    im = Image.open(fpath).convert("RGB")
    try:
        res = _eng(np.array(im))
    except Exception:
        continue
    boxes = getattr(res, "boxes", None)
    txts = getattr(res, "txts", None) or ()
    if boxes is None or not len(boxes):
        continue
    for box, lt in zip(boxes, txts):
        lt = norm(lt)
        if len(re.sub(r"[^A-Za-z0-9]", "", lt)) < 2:
            continue
        ratio, span = fuzzy_contains(lt, qtext)
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        w, h = x2 - x1, y2 - y1
        if w < 12 or h < 8:
            continue
        pad = max(3, int(h * 0.15))
        crop = im.crop((max(0, x1 - pad), max(0, y1 - pad),
                        min(im.width, x2 + pad), min(im.height, y2 + pad)))
        # normalise to 48px tall like PP-OCR rec input
        r = 48 / crop.height
        crop = crop.resize((max(8, int(crop.width * r)), 48), Image.BILINEAR)
        hh = hashlib.md5(crop.tobytes()).hexdigest()[:16]
        if hh in seenhash:
            continue
        seenhash.add(hh)
        if ratio >= 0.72 and span:
            # manhwa scanlation text is ~always uppercase; normalise the label
            label = span.upper()
            fn = f"{rec['title']}_{rec['frame_id']}_{len(rows):05d}.jpg"
            crop.save(f"{REC}/img/{fn}", quality=92)
            rows.append((f"img/{fn}", label))
        else:
            unmatched.append((img, rec["title"], lt, round(ratio, 2)))
    if (n + 1) % 100 == 0:
        print(f"  rec build {n+1}/{len(matched)}  crops={len(rows)}  unmatched={len(unmatched)}")

import random
random.seed(5); random.shuffle(rows)
nv = max(1, int(len(rows) * args.val_frac))
open(f"{REC}/val_list.txt", "w").write("\n".join(f"{p}\t{l}" for p, l in rows[:nv]))
open(f"{REC}/train_list.txt", "w").write("\n".join(f"{p}\t{l}" for p, l in rows[nv:]))
with open(f"{REC}/unmatched.tsv", "w") as f:
    f.write("img\ttitle\trapidocr_line\tbest_ratio\n")
    for u in unmatched:
        f.write("\t".join(str(x) for x in u) + "\n")
print(f"\nrec_real: {len(rows)} verified line crops ({len(rows)-nv} train / {nv} val)")
print(f"  unmatched rapidocr lines (candidate 2nd Qwen pass): {len(unmatched)} -> rec_real/unmatched.tsv")
