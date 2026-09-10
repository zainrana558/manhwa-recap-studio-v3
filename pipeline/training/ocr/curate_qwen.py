#!/usr/bin/env python3
"""Curate sliced frames into Qwen-transcription batches for OCR fine-tuning.

Reads every slice_work/<title>/temp_slices/chap_*/manifest.json, scores each
frame's RapidOCR hypothesis for garble, samples a title-balanced set weighted
toward hard/failed cases (the ones a fine-tune must fix) plus a clean control
tail, and writes:

  qwen_out/batch_XX/            <- ~180 frames each, renamed <title>_<frameid>.jpg
  qwen_out/batch_XX/MANIFEST.tsv   img \t title \t source_page \t rapidocr_text
  qwen_out/batch_XX.zip
  qwen_out/INDEX.json           <- full record for the cross-check step later

  .venv/bin/python curate_qwen.py [--per-title 220] [--batch 180] [--hard-frac 0.62]
"""
import os, re, sys, json, glob, shutil, zipfile, random, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import garble_helpers as G

SLICE = f"{HERE}/slice_work"
OUT = f"{HERE}/qwen_out"

ap = argparse.ArgumentParser()
ap.add_argument("--per-title", type=int, default=220)
ap.add_argument("--batch", type=int, default=180)
ap.add_argument("--mix", default="0.15,0.50,0.35", help="fail,garbled,clean target fractions")
ap.add_argument("--min-alpha", type=int, default=2, help="min alpha tokens in OCR text to count as 'has text'")
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()

random.seed(20)


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def alpha_toks(s):
    return [w for w in re.findall(r"[A-Za-z][A-Za-z']+", s or "")]


def score_text(txt):
    """Return (has_text: bool, hardness: float, mangled: int).
    hardness ~0 clean, higher = more garbled."""
    t = norm(txt)
    toks = alpha_toks(t)
    has_text = not (len(toks) < args.min_alpha and len(re.sub(r"[^A-Za-z]", "", t)) < 4)
    score = 0.0
    bad = mangled = 0
    for w in toks:
        wl = re.sub(r"[^a-z]", "", w.lower())
        if len(wl) < 2:
            continue
        if G.is_known_wordform(w) or wl in NAMES:
            continue
        if G.is_probable_sfx(w) or G.is_scream(w):
            score += 0.25
            continue
        if G.token_looks_mangled(w):
            score += 1.5
            mangled += 1
        else:
            score += 0.6
        bad += 1
    if toks:
        score += 1.6 * bad / len(toks)
    if G._CREDIT_LINE_RE.search(t):
        score += 0.4
    return has_text, score, mangled


# names vocab (reuse the nano-machine dict + any global)
NAMES = set()
for jp in glob.glob("/home/azureuser/manhwa-recap-studio/pipeline/ocr-corrections/*.json"):
    try:
        d = json.load(open(jp))
        for n in d.get("names", []):
            NAMES.add(re.sub(r"[^a-z]", "", str(n).lower()))
    except Exception:
        pass

records = []
for man in sorted(glob.glob(f"{SLICE}/*/temp_slices/chap_*/manifest.json")):
    title = man.split("/slice_work/")[1].split("/")[0]
    try:
        d = json.load(open(man))
    except Exception as e:
        print("skip", man, e); continue
    chap = re.search(r"chap_(\d+)", man).group(1)
    for f in d.get("frames", []):
        p = f.get("path")
        if not p or not os.path.exists(p):
            p = os.path.join(os.path.dirname(man), f.get("filename", ""))
        if not os.path.exists(p):
            continue
        txt = norm(f.get("ocr_text"))
        stt = f.get("ocr_status")
        has_text, hard, mangled = score_text(txt)
        if not has_text and stt in ("NO_TEXT",) and not txt:
            bucket = "fail"          # detector emitted a frame, OCR found nothing
        elif not has_text:
            bucket = "fail"
        elif mangled >= 1 or hard >= 0.55 or stt == "UNCERTAIN":
            bucket = "garbled"
        else:
            bucket = "clean"
        records.append({
            "title": title, "chap": chap,
            "frame_id": f.get("frame_id") or f.get("filename", "").split(".")[0],
            "path": p, "source_page": f.get("source_page"),
            "ocr_status": stt, "rapidocr_text": txt,
            "hardness": round(hard, 2), "mangled": mangled, "bucket": bucket,
        })

print(f"total frames: {len(records)}")

# ---- fail-bucket refinement: keep only frames that DO contain a text region
# (per the ogkalu detector) — those are real OCR misses, the gold signal.
# Pure-art frames with no detected text box are dropped from 'fail'.
_det = None
def has_text_box(path):
    global _det
    if _det is None:
        import onnxruntime as ort
        mp = "/home/azureuser/manhwa-recap-studio/pipeline/models/comic-text-and-bubble-detector/detector-v4-s_int8.onnx"
        so = ort.SessionOptions(); so.intra_op_num_threads = 3
        _det = ort.InferenceSession(mp, sess_options=so, providers=["CPUExecutionProvider"])
    import numpy as _np, cv2 as _cv2
    from PIL import Image as _Im
    im = _Im.open(path).convert("RGB"); rgb = _np.asarray(im); h, w = rgb.shape[:2]
    x = _cv2.resize(rgb, (640, 640)).transpose(2, 0, 1)[None].astype(_np.float32) / 255.0
    lab, box, sc = _det.run(None, {"images": x, "orig_target_sizes": _np.array([[h, w]], _np.int64)})
    for l, b, c in zip(lab[0], box[0], sc[0]):
        if float(c) >= 0.42 and int(l) in (1, 2):
            x1, y1, x2, y2 = b
            if (x2 - x1) >= 30 and (y2 - y1) >= 16:
                return True
    return False

if not args.dry:
    kept, drop = [], 0
    for r in records:
        if r["bucket"] == "fail":
            try:
                if not has_text_box(r["path"]):
                    drop += 1
                    continue
            except Exception:
                pass
        kept.append(r)
    print(f"fail-bucket refine: dropped {drop} textless frames")
    records = kept

by_title = collections.defaultdict(list)
for r in records:
    by_title[r["title"]].append(r)

# de-dup near-identical OCR text within a title (keeps first)
def keydup(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())[:60]

FAIL_F, GARB_F, CLEAN_F = (float(x) for x in args.mix.split(","))
picked = []
for title, rs in sorted(by_title.items()):
    seen = set()
    buckets = {"fail": [], "garbled": [], "clean": []}
    for r in rs:
        k = keydup(r["rapidocr_text"])
        if k and k in seen and r["bucket"] != "fail":
            continue
        if k:
            seen.add(k)
        buckets[r["bucket"]].append(r)
    for b in buckets:
        buckets[b].sort(key=lambda r: -r["hardness"] if b == "garbled" else random.random())
    want = {"fail": int(args.per_title * FAIL_F),
            "garbled": int(args.per_title * GARB_F),
            "clean": args.per_title - int(args.per_title * FAIL_F) - int(args.per_title * GARB_F)}
    sel, short = [], 0
    for b in ("fail", "garbled", "clean"):
        take = min(len(buckets[b]), want[b] + short)
        sel += buckets[b][:take]
        short += want[b] - take
    # if still short, backfill from whatever bucket has extra
    if short > 0:
        pool = [r for b in buckets for r in buckets[b][want[b]:] if r not in sel]
        random.shuffle(pool)
        sel += pool[:short]
    random.shuffle(sel)
    picked += sel
    bc = collections.Counter(r["bucket"] for r in sel)
    print(f"  {title:14} frames={len(rs):5} (f{len(buckets['fail'])}/g{len(buckets['garbled'])}/c{len(buckets['clean'])}) "
          f"-> pick {len(sel)}  fail={bc['fail']} garbled={bc['garbled']} clean={bc['clean']}")

random.shuffle(picked)
print(f"\nTOTAL picked: {len(picked)}")

if args.dry:
    sys.exit(0)

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
index = []
nb = (len(picked) + args.batch - 1) // args.batch
for bi in range(nb):
    chunk = picked[bi * args.batch:(bi + 1) * args.batch]
    bdir = f"{OUT}/batch_{bi:02d}"
    os.makedirs(bdir, exist_ok=True)
    rows = []
    for r in chunk:
        img = f"{r['title']}_{r['frame_id']}.jpg"
        shutil.copy2(r["path"], f"{bdir}/{img}")
        rows.append((img, r["title"], r.get("source_page") or "", r["rapidocr_text"]))
        index.append({**{k: r[k] for k in ("title", "chap", "frame_id", "source_page", "path",
                                           "ocr_status", "rapidocr_text", "hardness", "mangled", "bucket")},
                      "batch": bi, "img": img})
    with open(f"{bdir}/MANIFEST.tsv", "w") as fh:
        fh.write("img\ttitle\tsource_page\trapidocr_text\n")
        for x in rows:
            fh.write("\t".join(str(v).replace("\t", " ") for v in x) + "\n")
    with zipfile.ZipFile(f"{OUT}/batch_{bi:02d}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(bdir)):
            z.write(f"{bdir}/{fn}", fn)
    print(f"  batch_{bi:02d}: {len(chunk)} frames -> batch_{bi:02d}.zip")

json.dump(index, open(f"{OUT}/INDEX.json", "w"), indent=1)
print(f"\nwrote {nb} batches + INDEX.json to {OUT}/")
