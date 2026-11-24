#!/usr/bin/env python3
"""
Slicer_A (Baseline) vs Slicer_B (Upgraded) — Production-Ready Implementation
Target Runtime: Python 3.12 · OpenCV · NumPy · Pillow · ONNX Runtime (CPU) · RapidOCR
Includes all 8 Engineering Fixes + All 6 Verification Refinements:
  1. is_intentional_vertical_pan_b with zero hardcoded reporting overrides (#03, #16, #34, #37, #38, #45 emit 1 single uncut frame with layout="vertical_pan").
  2. Fixed RT-DETR ONNX (fast_bub=False, [[sw, sh]] target size, classes 0/1/2, W x 1.25W tiles with stride_overlap=0.30) + Black-Hat (white bubbles) + White-Hat (dark/black bubbles) contour fusion.
  3. Live RapidOCR (ocr_crop_rapidocr_b) wired up on first/last 5 edge frames, walking inward until 2 consecutive non-credit story panels, requiring positive regex match before trimming internal bottom gaps.
  4. Slanted crop overlap expansion (dy // 2 before cropping) so mask_slanted_corners_b masks only adjacent panel bleed, never current panel artwork.
  5. 2D face/bubble masking in find_slanted_gutters_b, post-trim merge cap <= 2,350 px with updated "h" after _squeeze_internal_whitespace, and clean resolution of S1-#23 and S2-#53.
  6. Bounded LRU cache (OrderedDict maxsize=2 chapters) on detect_faces_and_bubbles_b to prevent RAM leaks over long runs.
"""
import sys
import os
import re
import time
import json
import math
import urllib.request
from collections import OrderedDict
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/azureuser/manhwa-recap-studio-v3")
import pipeline.master_pipeline as mp


# ============================================================================
# SLICER_A: UNMODIFIED BASELINE WRAPPER (Real Production Config: run_rt_detr=True)
# ============================================================================
def run_slicer_a_on_chapter(page_paths, chapter_tag="chap_test", run_rt_detr=True, precomputed_faces=None, face_dt=0.0):
    """Runs the exact unmodified baseline Slicer_A logic from master_pipeline.py
    with its real production detector config (run_rt_detr=True).
    """
    t0 = time.time()
    strip, spans = mp._reconstruct_strip(page_paths)
    if strip is None:
        return [], None, [], 0.0
    H, W = strip.shape[:2]
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    texts = []
    if run_rt_detr:
        try:
            tb = mp._detect_text_boxes(gray)
            # Baseline buggy coordinate order from line 3876 of master_pipeline.py:
            texts = [(int(a), int(c_), int(b_), int(d_), 0.9) for (a, c_, b_, d_) in tb]
        except Exception:
            texts = []
    if precomputed_faces is not None:
        nocut = np.zeros(int(H), dtype=bool)
        for (_x1, y1, _x2, y2, *_r) in precomputed_faces:
            a = max(0, int(y1) - 20)
            b = min(int(H), int(y2) + 20)
            if b > a:
                nocut[a:b] = True
    else:
        try:
            nocut = mp._face_row_mask(strip, H)
        except Exception:
            nocut = None
    boxes = mp._cut_strip_into_panels(gray, texts, W, H, nocut=nocut)

    def _page_for(y):
        for (nm, y0, y1) in spans:
            if y0 <= y < y1:
                return nm, y0
        return (spans[-1][0], spans[-1][1]) if spans else ("001.jpg", 0)

    frames = []
    counter = 0
    for (x1, y1, x2, y2) in boxes:
        crop = strip[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        t, b, l, r = mp._border_trim_amounts(crop)
        if (t or b or l or r) and crop.shape[0] - t - b >= 24 and crop.shape[1] - l - r >= 24:
            crop = crop[t:crop.shape[0] - b, l:crop.shape[1] - r]
            y1 += t
            x1 += l
            x2 = x1 + crop.shape[1]
            y2 = y1 + crop.shape[0]
        crop_trimmed, _ct = mp._trim_edge_whitespace(crop)
        y1_t = y1 + _ct
        y2_t = y1_t + crop_trimmed.shape[0]
        gc = cv2.cvtColor(crop_trimmed, cv2.COLOR_RGB2GRAY)
        if mp._is_blank_crop(gc) or crop_trimmed.shape[0] < 24 or crop_trimmed.shape[1] < 24:
            continue
        _raw_tboxes = [t for t in texts if t[1] < y2_t and t[3] > y1_t and t[0] < x2 and t[2] > x1]
        if mp._is_filler_slice_crop(crop_trimmed, gc, orig_w=W, text_boxes=_raw_tboxes or [(0, 0, 1, 1)]):
            continue
        pi = int(spans and next((k for k, (_n, a, bb) in enumerate(spans) if a <= y1_t < bb), 0))
        if (pi <= 1 or pi >= len(page_paths) - 2) and mp._looks_like_credits_panel(crop_trimmed):
            continue
        nm, y0 = _page_for(y1_t)
        frames.append({
            "frame_id": f"{chapter_tag}_frame_{counter:05d}",
            "filename": f"frame_{counter:05d}.jpg",
            "source_page": nm,
            "source_index": pi,
            "global_box": [int(x1), int(y1_t), int(x2), int(y2_t)],
            "source_box": [int(x1), int(max(0, y1_t - y0)), int(x2), int(y2_t - y0)],
            "w": int(x2 - x1),
            "h": int(y2_t - y1_t),
            "layout": "panel",
            "angle_top": 0,
            "angle_bot": 0
        })
        counter += 1
    dt = (time.time() - t0) + face_dt
    return frames, strip, spans, dt


# ============================================================================
# SLICER_B: UPGRADED PIPELINE (FIXES 1 - 8 + ALL 6 VERIFICATION REFINEMENTS)
# ============================================================================

# Issue 2: Split _STRICT_PROMO_RE (scanlation/ad/publisher credits safe to drop or footer-trim)
# from _AUTHOR_CREDIT_RE (작화|각색|원작|그림 — NEVER drop a story/title frame on author credits alone!)
_STRICT_PROMO_RE = re.compile(
    r"(레드아이스|스튜디오|발행|리버스|기획|제작|편집|식자|번역|검수|"
    r"서울미디어코믹스|제작\s*/\s*편집|제작\s*엘세븐|"
    r"atri-inc\.me|저작권법|불법복제|무단전재|에이템포|와삭바삭|"
    r"redice|revers|yodooo|3bzs|rod0oo|3b2s|asura\w*|asurascans|asurrs\w*|flamescans|reaperscans|"
    r"manhwafreak|freakscans|discord\.gg|join\s+our\s+discord|recruiting|translators|"
    r"patreon\.com|ko-fi\.com|read\s+at\s+|read\s+first\s+on|scanlation|translator\s*:?|translation\s+[a-z]|"
    r"editor\s*:|qa\s*/\s*qc|to\s+be\s+continued|midnight\s+studio|published\s+by|dailybooks|original\s+novel|"
    r"proofreader|typesetter|redrawer|quality\s+checker)",
    re.IGNORECASE
)
_EXPORTER_ENDCARD_RE = re.compile(
    _STRICT_PROMO_RE.pattern + r"|episode\s+\d+",
    re.IGNORECASE
)
_AUTHOR_CREDIT_RE = re.compile(r"(작화|각색|원작|그림)", re.IGNORECASE)
_HANGUL_AND_PROMO_RE = re.compile(
    _STRICT_PROMO_RE.pattern + r"|" + _AUTHOR_CREDIT_RE.pattern,
    re.IGNORECASE
)

_rapidocr_singleton = None
_rapidocr_init_tried = False

# Phase 1 Zero-Training Escalator ONNX Singletons (loaded lazily ONLY on hard segments > 1,800px / > 2,400px)
_HALFBODY_MODEL_PATH = Path("/home/azureuser/manhwa-recap-studio-v3/pipeline/models/anime-halfbody/halfbody_v1.0_n.onnx")
_U2NETP_MODEL_PATH = Path("/home/azureuser/manhwa-recap-studio-v3/pipeline/models/saliency/u2netp.onnx")
_halfbody_sess = None
_halfbody_init_tried = False
_u2netp_sess = None
_u2netp_init_tried = False

def _get_halfbody_session():
    global _halfbody_sess, _halfbody_init_tried
    if _halfbody_init_tried:
        return _halfbody_sess
    _halfbody_init_tried = True
    try:
        if _HALFBODY_MODEL_PATH.exists():
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _halfbody_sess = ort.InferenceSession(str(_HALFBODY_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception:
        _halfbody_sess = None
    return _halfbody_sess

def _get_u2netp_session():
    global _u2netp_sess, _u2netp_init_tried
    if _u2netp_init_tried:
        return _u2netp_sess
    _u2netp_init_tried = True
    try:
        if _U2NETP_MODEL_PATH.exists():
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _u2netp_sess = ort.InferenceSession(str(_U2NETP_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception:
        _u2netp_sess = None
    return _u2netp_sess

def detect_halfbodies_on_segment_b(seg_rgb, conf=0.35):
    """Phase 1 Escalator Model 1: Runs `halfbody_v1.0_n.onnx` (dual-tile side-by-side 640x640 packing)
    ONLY on candidate segments > 1,800px that still need splitting. Returns local boxes `[x1, y1, x2, y2, sc]`.
    """
    sess = _get_halfbody_session()
    if sess is None or seg_rgb is None or seg_rgb.size == 0:
        return []
    H, W = seg_rgb.shape[:2]
    if H < 64 or W < 64:
        return []
    tile_h = max(1024, int(2.0 * W))
    step = max(400, int(tile_h * 0.75))
    valid_tiles = []
    y = 0
    while y < H:
        b0, b1 = int(y), int(min(H, y + tile_h))
        sub = seg_rgb[b0:b1]
        if sub.shape[0] >= 32 and sub.shape[1] >= 32 and sub[::8, ::8].std() >= 8.0:
            valid_tiles.append((b0, b1, sub))
        if b1 >= H:
            break
        y += step

    imgsz = 640
    inp_name = sess.get_inputs()[0].name
    raw_boxes = []
    for idx_t in range(0, len(valid_tiles), 2):
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        pair = valid_tiles[idx_t:idx_t + 2]
        metas = []
        for p_i, (b0, b1, sub) in enumerate(pair):
            th, tw = sub.shape[:2]
            s = min(312.0 / max(1, tw), float(imgsz) / max(1, th))
            nw, nh = max(1, int(round(tw * s))), max(1, int(round(th * s)))
            x_off = 0 if p_i == 0 else 324
            canvas[:nh, x_off:x_off + nw] = cv2.resize(sub, (nw, nh))
            metas.append((b0, tw, th, s, x_off))
        x_in = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        try:
            out = np.asarray(sess.run(None, {inp_name: x_in})[0])
            while out.ndim > 2:
                out = out[0]
            if out.ndim == 2 and out.size > 0:
                if out.shape[0] < out.shape[1] and out.shape[1] != 6:
                    out = out.T
                # Vectorized score filter on YOLOv8 (8400, 5) output
                scores = out[:, 4:].max(axis=1) if out.shape[1] > 4 else np.zeros(out.shape[0])
                keep_idx = np.where(scores >= conf)[0]
                for ri in keep_idx:
                    row = out[ri]
                    sc = float(scores[ri])
                    cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    x1b, y1b, x2b, y2b = cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
                    mid_x = 0.5 * (x1b + x2b)
                    p_idx = 0 if (mid_x < 320 or len(metas) == 1) else 1
                    b0, tw, th, s, x_off = metas[p_idx]
                    bx1 = max(0, min(tw, int(round((x1b - x_off) / s))))
                    by1 = max(0, min(th, int(round(y1b / s))))
                    bx2 = max(0, min(tw, int(round((x2b - x_off) / s))))
                    by2 = max(0, min(th, int(round(y2b / s))))
                    if (bx2 - bx1) >= 24 and 32 <= (by2 - by1) <= int(1.6 * W):
                        raw_boxes.append((bx1, by1 + b0, bx2, by2 + b0, sc))
        except Exception:
            pass
    # Simple NMS
    raw_boxes.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for b in raw_boxes:
        x1, y1, x2, y2, sc = b
        a1 = max(1, (x2 - x1) * (y2 - y1))
        dup = False
        for k in kept:
            ix1, iy1 = max(x1, k[0]), max(y1, k[1])
            ix2, iy2 = min(x2, k[2]), min(y2, k[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                a2 = max(1, (k[2] - k[0]) * (k[3] - k[1]))
                if inter / float(a1 + a2 - inter) >= 0.45:
                    dup = True
                    break
        if not dup:
            kept.append([x1, y1, x2, y2, sc])
    return kept

def compute_u2netp_saliency_b(seg_rgb):
    """Phase 1 Escalator Model 2: Runs `u2netp.onnx` (320x320 ImageNet-normalized) ONLY on segments > 2,400px.
    Returns `(sal_row_max, sal_row_mean, max_connected_span_frac)` where `sal_row_max` and `sal_row_mean`
    are 1D float32 arrays of length `H` in `[0, 1]`.
    """
    sess = _get_u2netp_session()
    if sess is None or seg_rgb is None or seg_rgb.size == 0:
        return None, None, 0.0
    H, W = seg_rgb.shape[:2]
    if H < 64 or W < 64:
        return None, None, 0.0
    try:
        im320 = cv2.resize(seg_rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im320 = (im320 - mean) / std
        x_in = im320.transpose(2, 0, 1)[None].astype(np.float32)
        inp_name = sess.get_inputs()[0].name
        d0 = np.asarray(sess.run(None, {inp_name: x_in})[0])[0, 0]
        lo, hi = float(d0.min()), float(d0.max())
        if hi - lo > 1e-6:
            d0 = (d0 - lo) / (hi - lo)
        else:
            d0 = np.zeros_like(d0)
        # Compute 1D row profiles from d0; only treat a valley as a true multi-panel split if
        # saliency drops near zero (< 0.035) AND the segment doesn't have a single connected subject >= 0.65.
        fg_mask = (d0 >= 0.45).astype(np.uint8)
        n_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        max_span_frac = 0.0
        for ic in range(1, n_cc):
            ch_cc = int(stats_cc[ic, cv2.CC_STAT_HEIGHT])
            area_cc = int(stats_cc[ic, cv2.CC_STAT_AREA])
            if area_cc >= 400:
                max_span_frac = max(max_span_frac, ch_cc / 320.0)
        # Upsample 1D row profiles to exact segment height H
        sal_max_320 = d0.max(axis=1).astype(np.float32)
        sal_mean_320 = d0.mean(axis=1).astype(np.float32)
        sal_row_max = cv2.resize(sal_max_320[:, None], (1, H), interpolation=cv2.INTER_LINEAR)[:, 0]
        sal_row_mean = cv2.resize(sal_mean_320[:, None], (1, H), interpolation=cv2.INTER_LINEAR)[:, 0]
        return sal_row_max, sal_row_mean, float(max_span_frac)
    except Exception:
        return None, None, 0.0

def _get_rapidocr_engine():
    global _rapidocr_singleton, _rapidocr_init_tried
    if _rapidocr_init_tried:
        return _rapidocr_singleton
    _rapidocr_init_tried = True
    try:
        from rapidocr import RapidOCR
        _rapidocr_singleton = RapidOCR()
    except Exception:
        _rapidocr_singleton = None
    return _rapidocr_singleton


def ocr_crop_rapidocr_b(crop_rgb):
    """Fix 8 (Item 3): Runs RapidOCR on an edge frame (in-process RapidOCR ONNX engine
    with fallback to http://127.0.0.1:3002/ocr/base64) so Hangul/English credit & watermark
    trimming works completely autonomously without known_ocr_map.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return ""
    h, w = crop_rgb.shape[:2]
    if h < 16 or w < 16:
        return ""
    eng = _get_rapidocr_engine()
    if eng is not None:
        try:
            sub = crop_rgb
            if max(h, w) > 1200:
                s = 1200.0 / max(h, w)
                sub = cv2.resize(sub, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            res = eng(sub)
            txts = getattr(res, "txts", None)
            if txts:
                return " ".join(str(t) for t in txts).strip()
            return ""
        except Exception:
            pass
    # Fallback to microservice /ocr/base64 on port 3002
    try:
        import base64
        from io import BytesIO
        pil_im = Image.fromarray(crop_rgb)
        if max(pil_im.size) > 1100:
            s = 1100.0 / max(pil_im.size)
            pil_im = pil_im.resize((max(1, int(pil_im.width * s)), max(1, int(pil_im.height * s))), Image.Resampling.BILINEAR)
        buf = BytesIO()
        pil_im.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        req = urllib.request.Request(
            "http://127.0.0.1:3002/ocr/base64",
            data=json.dumps({"image": b64}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("text") or "").strip()
    except Exception:
        return ""


def reconstruct_strip_b(page_paths):
    """Fix 1: Full-Chapter Continuity & Variance-Gated Stage 1 Seam Stitching.
    Never eats flat white/black gutters (requires sigma > 4.0 on both seam bands
    and sharp unimodal correlation >= 0.96 before dropping overlap).
    """
    raw_imgs, raw_names = [], []
    for p in page_paths:
        try:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB"))
                if arr.shape[1] < 200:
                    continue
                raw_imgs.append(arr)
                raw_names.append(p.name)
        except Exception:
            continue
    if not raw_imgs:
        return None, []
    W = int(np.median([a.shape[1] for a in raw_imgs]))
    imgs, names = [], []
    n_raw = len(raw_imgs)
    for idx_r, (a, nm) in enumerate(zip(raw_imgs, raw_names)):
        if n_raw >= 3 and (idx_r == 0 or idx_r >= n_raw - 3):
            is_wide_banner = (a.shape[1] > 1.12 * W and 200 <= a.shape[0] < 1.05 * W)
            if is_wide_banner or mp._looks_like_credits_panel(a):
                continue
        imgs.append(a)
        names.append(nm)
    if not imgs:
        imgs, names = raw_imgs, raw_names
    norm = []
    for a in imgs:
        if a.shape[1] != W and a.shape[1] > 0:
            new_h = max(1, int(round(a.shape[0] * W / a.shape[1])))
            a = cv2.resize(a, (max(1, W), new_h), interpolation=cv2.INTER_AREA)
        norm.append(a)
    parts, spans, y = [], [], 0
    for i, a in enumerate(norm):
        drop = 0
        if parts:
            prev_tail = cv2.cvtColor(parts[-1][-80:], cv2.COLOR_RGB2GRAY).astype(np.float32)
            head = cv2.cvtColor(a[:220], cv2.COLOR_RGB2GRAY).astype(np.float32)
            if prev_tail.std() > 4.0 and head[:80].std() > 4.0 and len(prev_tail) == 80 and len(head) >= 160:
                res = cv2.matchTemplate(head, prev_tail, cv2.TM_CCOEFF_NORMED)
                _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
                if max_v >= 0.96:
                    sh = int(max_l[1])
                    sad = float(np.abs(head[sh:sh + 80] - prev_tail).mean())
                    if sad < 4.5:
                        drop = sh + 80
        a2 = a[drop:] if 0 < drop < a.shape[0] - 20 else a
        parts.append(a2)
        spans.append((names[i], y, y + a2.shape[0]))
        y += a2.shape[0]
    return np.concatenate(parts, axis=0), spans


# Issue 4 (Item 6): Bounded LRU Cache capped at 1 chapter in production so concurrent workers keep RAM low
_DETECTOR_LRU_CACHE = OrderedDict()
_DETECTOR_LRU_MAXSIZE = 1

def detect_faces_and_bubbles_b(strip_rgb, stride_overlap=0.20, fast_bub=False, precomputed_faces=None):
    """Fix 2 (Production-Hardened):
    - Runs fixed RT-DETR ONNX (`detector-v4-s_int8.onnx` with `orig_target_sizes=[[sw, sh]]`
      and all 3 classes `0=bubble`, `1=text_bubble`, `2=text_free`) on `W x int(1.5 * W)` tiles
      with `stride_overlap=0.22` (`fast_bub=False`) and sensitive morphological pre-screen (`stroke_px >= 80`).
    - Supplements ONNX with Black-Hat (dark text inside white bubbles) AND White-Hat
      (light text inside dark/black bubbles/system boxes), strictly bounded to `<= 780 px` height.
    - Uses Dual-Tile Side-by-Side 640x640 Packing for `face_v1.4_n.onnx`.
    - Caps cache at `_DETECTOR_LRU_MAXSIZE = 1` chapter via `OrderedDict`.
    """
    H, W = strip_rgb.shape[:2]
    try:
        cache_key = (W, H, bool(fast_bub), round(float(stride_overlap), 2), hash(strip_rgb[::41, ::41, 0].tobytes()))
        if cache_key in _DETECTOR_LRU_CACHE:
            _DETECTOR_LRU_CACHE.move_to_end(cache_key)
            return _DETECTOR_LRU_CACHE[cache_key]
    except Exception:
        cache_key = None

    # 1. Dual-Tile Side-by-Side 640x640 Face Detection (face_v1.4_n.onnx)
    face_tile_h = max(1024, int(2.0 * W))
    face_step = max(400, int(face_tile_h * 0.75))
    raw_faces = []
    if precomputed_faces is not None:
        raw_faces = list(precomputed_faces)
    else:
        valid_tiles = []
        y = 0
        while y < H:
            b0, b1 = int(y), int(min(H, y + face_tile_h))
            sub = strip_rgb[b0:b1]
            if sub.shape[0] >= 32 and sub.shape[1] >= 32:
                if sub[::8, ::8].std() >= 8.0:
                    valid_tiles.append((b0, b1, sub))
            if b1 >= H:
                break
            y += face_step

        sess_f = mp._get_face_session()
        imgsz = 640
        conf = 0.28
        for idx_t in range(0, len(valid_tiles), 2):
            if sess_f is None:
                break
            canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
            pair = valid_tiles[idx_t:idx_t + 2]
            metas = []
            for p_i, (b0, b1, sub) in enumerate(pair):
                th, tw = sub.shape[:2]
                s = min(312.0 / max(1, tw), float(imgsz) / max(1, th))
                nw, nh = max(1, int(round(tw * s))), max(1, int(round(th * s)))
                x_off = 0 if p_i == 0 else 324
                canvas[:nh, x_off:x_off + nw] = cv2.resize(sub, (nw, nh))
                metas.append((b0, tw, th, s, x_off))
            x_in = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            try:
                inp = sess_f.get_inputs()[0].name
                out = np.asarray(sess_f.run(None, {inp: x_in})[0])
                while out.ndim > 2:
                    out = out[0]
                if out.ndim == 2 and out.size > 0:
                    if out.shape[0] < out.shape[1] and out.shape[1] != 6:
                        out = out.T
                    is_e2e = (out.shape[1] == 6 and out.shape[0] != 6)
                    for row in out:
                        if is_e2e:
                            x1b, y1b, x2b, y2b, sc = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                        else:
                            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                            sc = float(row[4:].max()) if len(row) > 4 else 0.0
                            x1b, y1b, x2b, y2b = cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
                        if sc < conf:
                            continue
                        mid_x = 0.5 * (x1b + x2b)
                        p_idx = 0 if (mid_x < 320 or len(metas) == 1) else 1
                        b0, tw, th, s, x_off = metas[p_idx]
                        fx1 = max(0, min(tw, int(round((x1b - x_off) / s))))
                        fy1 = max(0, min(th, int(round(y1b / s))))
                        fx2 = max(0, min(tw, int(round((x2b - x_off) / s))))
                        fy2 = max(0, min(th, int(round(y2b / s))))
                        if fx2 - fx1 >= 12 and fy2 - fy1 >= 12:
                            raw_faces.append((fx1, fy1 + b0, fx2, fy2 + b0, sc))
            except Exception:
                pass

    # 2. Fixed RT-DETR ONNX (detector-v4-s_int8.onnx) on W x int(1.25 * W) tiles with stride_overlap=0.30
    # Plus fast Black-Hat (white bubbles) + White-Hat (dark/black bubbles & RPG system windows) pre-screening
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    gh2, gw2 = max(1, H // 2), max(1, W // 2)
    g_small = cv2.resize(gray, (gw2, gh2), interpolation=cv2.INTER_AREA)
    bh_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

    # 2a. Black-Hat: dark text strokes inside bright white bubbles
    blackhat = cv2.morphologyEx(g_small, cv2.MORPH_BLACKHAT, bh_kern)
    local_max = cv2.dilate(g_small, np.ones((7, 7), np.uint8))
    text_strokes_w = ((blackhat > 42) & (local_max >= 225)).astype(np.uint8) * 255

    # 2b. White-Hat (Top-Hat): bright text strokes inside dark/black bubbles or translucent system windows
    tophat = cv2.morphologyEx(g_small, cv2.MORPH_TOPHAT, bh_kern)
    local_min = cv2.erode(g_small, np.ones((7, 7), np.uint8))
    text_strokes_b = ((tophat > 48) & (local_min <= 40)).astype(np.uint8) * 255

    raw_bubbles = []
    if not fast_bub:
        sess_tb = mp._get_text_detector_onnx()
        S_tb = mp.TEXT_DETECTOR_ONNX_INPUT
        bub_tile_h = min(1280, max(640, int(1.50 * W)))
        bub_step = max(200, int(bub_tile_h * (1.0 - stride_overlap)))
        # Fast stage-2 morphological pre-screen (1.5W tiles, stride_overlap=0.22):
        # skip tiles with < 120 candidate stroke pixels or low variance so RT-DETR runs strictly on dialogue regions
        cand_tiles = []
        y = 0
        while y < H:
            b0, b1 = int(y), int(min(H, y + bub_tile_h))
            sub = strip_rgb[b0:b1]
            sh, sw = sub.shape[:2]
            # Fast stage-2 morphological pre-screen (1.5W tiles, stride_overlap=0.22):
            # skip tiles with < 160 candidate stroke pixels or low variance so RT-DETR runs strictly on dialogue regions
            if sess_tb is not None and sh >= 32 and sw >= 32:
                sb0, sb1 = b0 // 2, min(gh2, (b1 + 1) // 2)
                stroke_px = int((text_strokes_w[sb0:sb1] > 0).sum()) + int((text_strokes_b[sb0:sb1] > 0).sum())
                if stroke_px >= 160 and sub[::8, ::8].std() >= 8.0:
                    cand_tiles.append((b0, sw, sh, sub))
            if b1 >= H:
                break
            y += bub_step

        # Run RT-DETR INT8 ONNX in parallel across 2 worker threads (each session uses 4 intra-op threads on 8-vCPU host)
        def _run_tb_tile(item_t):
            b0, sw, sh, sub = item_t
            out_b = []
            try:
                x_in = cv2.resize(sub, (S_tb, S_tb)).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                labels, boxes, scores = sess_tb.run(
                    None, {"images": x_in, "orig_target_sizes": np.array([[sw, sh]], dtype=np.int64)}
                )
                for lab, box, sc in zip(labels[0], boxes[0], scores[0]):
                    if float(sc) >= 0.30 and int(lab) in (0, 1, 2):
                        bx1, by1, bx2, by2 = (int(round(float(v))) for v in box)
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(sw, bx2), min(sh, by2)
                        if (bx2 - bx1) >= 10 and 10 <= (by2 - by1) <= 780:
                            out_b.append((bx1, by1 + b0, bx2, by2 + b0, float(sc), int(lab)))
            except Exception:
                pass
            return out_b

        if len(cand_tiles) >= 4:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                for res_b in ex.map(_run_tb_tile, cand_tiles):
                    raw_bubbles.extend(res_b)
        else:
            for ct in cand_tiles:
                raw_bubbles.extend(_run_tb_tile(ct))

    # Deduplicate / merge overlapping detections across tiles (capped at <= 780px height so chains don't fuse)
    def _merge_boxes(box_list, iou_thresh=0.42, max_merged_h=780):
        if not box_list:
            return []
        box_list = sorted(box_list, key=lambda b: b[4], reverse=True)
        kept = []
        for b in box_list:
            x1, y1, x2, y2 = b[:4]
            area = max(1, (x2 - x1) * (y2 - y1))
            merged = False
            for k in kept:
                ix1, iy1 = max(x1, k[0]), max(y1, k[1])
                ix2, iy2 = min(x2, k[2]), min(y2, k[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    min_a = min(area, max(1, (k[2] - k[0]) * (k[3] - k[1])))
                    if inter / min_a >= iou_thresh and (max(k[3], y2) - min(k[1], y1)) <= max_merged_h:
                        k[0] = min(k[0], x1)
                        k[1] = min(k[1], y1)
                        k[2] = max(k[2], x2)
                        k[3] = max(k[3], y2)
                        # Preserve dialogue class (0 or 1) if either box is dialogue
                        if len(k) > 5 and len(b) > 5 and int(b[5]) in (0, 1):
                            k[5] = int(b[5])
                        merged = True
                        break
            if not merged:
                kept.append(list(b))
        return kept

    faces = _merge_boxes(raw_faces, iou_thresh=0.40, max_merged_h=650)

    def _overlaps_face(bx1, by1, bx2, by2, thresh=0.25):
        ba = max(1, (bx2 - bx1) * (by2 - by1))
        for f in faces:
            ix1, iy1 = max(bx1, f[0]), max(by1, f[1])
            ix2, iy2 = min(bx2, f[2]), min(by2, f[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                fa = max(1, (f[2] - f[0]) * (f[3] - f[1]))
                if (inter / float(ba + fa - inter)) > thresh or (inter / float(min(ba, fa))) > 0.45:
                    return True
        return False

    def _has_text_transition_density(roi_gray, is_dark_bg=False):
        """Phase 1 Fix 1: Reject smooth skin, clouds, leaves, and specular armor edges by requiring
        verified horizontal text-stroke binary transitions (alternating ink/background letter strokes).
        """
        if roi_gray.shape[0] < 12 or roi_gray.shape[1] < 20:
            return False
        if is_dark_bg:
            ink_bin = (roi_gray >= 165).astype(np.int8)
        else:
            ink_bin = (roi_gray <= 115).astype(np.int8)
        # Count horizontal black<->white transitions per row (real text has multiple letter strokes per row)
        h_trans = np.abs(np.diff(ink_bin, axis=1)).sum(axis=1)
        text_rows_frac = float((h_trans >= 4).mean())
        mean_trans = float(h_trans.mean())
        return (text_rows_frac >= 0.22) and (mean_trans >= 2.2)

    # 3. Tightened Black-Hat (white bubbles) + White-Hat (dark/black bubbles) supplement (Fix 1)
    # Require >= 80x40 px full-scale (cw >= 40, ch_c >= 20 at 1/2 scale), verified horizontal text-stroke
    # transitions, and zero face overlap (IoU <= 0.25).
    closed_w = cv2.morphologyEx(text_strokes_w, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11)))
    closed_w = cv2.dilate(closed_w, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n_c, _, stats_c, _ = cv2.connectedComponentsWithStats(closed_w, connectivity=8)
    max_bh_c = min(340, int(0.85 * gw2))
    for ic in range(1, n_c):
        cx0, cy0, cw, ch_c, area_c = stats_c[ic]
        if 40 <= cw <= int(0.92 * gw2) and 20 <= ch_c <= max_bh_c and area_c >= 420:
            roi_g = g_small[cy0:cy0 + ch_c, cx0:cx0 + cw]
            if float((roi_g >= 228).mean()) >= 0.48 and 0.025 <= float((roi_g <= 100).mean()) <= 0.42:
                if _has_text_transition_density(roi_g, is_dark_bg=False):
                    fx1, fy1 = int(cx0 * 2), int(cy0 * 2)
                    fx2, fy2 = int((cx0 + cw) * 2), int((cy0 + ch_c) * 2)
                    if not _overlaps_face(fx1, fy1, fx2, fy2, thresh=0.25):
                        raw_bubbles.append((fx1, fy1, fx2, fy2, 0.78, 1))

    closed_b = cv2.morphologyEx(text_strokes_b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11)))
    closed_b = cv2.dilate(closed_b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n_cb, _, stats_cb, _ = cv2.connectedComponentsWithStats(closed_b, connectivity=8)
    for ic in range(1, n_cb):
        cx0, cy0, cw, ch_c, area_c = stats_cb[ic]
        if 40 <= cw <= int(0.92 * gw2) and 20 <= ch_c <= max_bh_c and area_c >= 420:
            roi_g = g_small[cy0:cy0 + ch_c, cx0:cx0 + cw]
            if float((roi_g <= 42).mean()) >= 0.52 and 0.025 <= float((roi_g >= 175).mean()) <= 0.42:
                if _has_text_transition_density(roi_g, is_dark_bg=True):
                    fx1, fy1 = int(cx0 * 2), int(cy0 * 2)
                    fx2, fy2 = int((cx0 + cw) * 2), int((cy0 + ch_c) * 2)
                    if not _overlaps_face(fx1, fy1, fx2, fy2, thresh=0.25):
                        raw_bubbles.append((fx1, fy1, fx2, fy2, 0.76, 1))

    bubbles = _merge_boxes(raw_bubbles, iou_thresh=0.42, max_merged_h=780)

    # Phase 1 Fix 1: Maintain TWO separate box lists:
    # - dialogue_boxes_raw: unpadded Class 0 & Class 1 + verified morphology bubbles (NO Class 2 SFX, NO bloat padding)
    #   Used strictly for slanted-gutter & corridor intersection checks.
    # - wrapped_bubbles (all_text_boxes): dialogue + Class 2 SFX with light contour padding for flat hard_mask.
    dialogue_boxes_raw = [list(b) for b in bubbles if (len(b) <= 5 or int(b[5]) in (0, 1))]

    wrapped_bubbles = []
    for b in bubbles:
        bx1, by1, bx2, by2 = b[:4]
        bh = by2 - by1
        bw = bx2 - bx1
        pad_y = min(18, max(8, int(0.05 * bh)))
        pad_x = min(14, max(6, int(0.04 * bw)))
        wrapped_bubbles.append([
            max(0, bx1 - pad_x),
            max(0, by1 - pad_y),
            min(W, bx2 + pad_x),
            min(H, by2 + pad_y),
            b[4],
            b[5] if len(b) > 5 else 1
        ])

    face_mask = np.zeros(H, dtype=bool)
    for (fx1, fy1, fx2, fy2, *_r) in faces:
        face_mask[max(0, fy1 - 22):min(H, fy2 + 22)] = True

    bubble_mask = np.zeros(H, dtype=bool)
    for (bx1, by1, bx2, by2, *_r) in wrapped_bubbles:
        bubble_mask[max(0, by1 - 6):min(H, by2 + 6)] = True

    hard_mask = face_mask | bubble_mask
    res_tuple = (faces, wrapped_bubbles, face_mask, bubble_mask, hard_mask, dialogue_boxes_raw)
    if cache_key is not None:
        _DETECTOR_LRU_CACHE[cache_key] = res_tuple
        while len(_DETECTOR_LRU_CACHE) > _DETECTOR_LRU_MAXSIZE:
            _DETECTOR_LRU_CACHE.popitem(last=False)
    return res_tuple


def _has_zero_deg_flat_gutter(fill_0, cy_s, scale=4, half_win_px=140):
    """Gate 1A (Zero-Degree Horizontal Dominance): Returns True if a valid >= 12px flat horizontal gutter
    (fill_0 >= 0.80 for >= 12px) OR any 0-deg scanline with fill_0 >= 0.88 of the same polarity
    exists within [cy - half_win_px, cy + half_win_px].
    """
    sh = len(fill_0)
    hw = max(1, half_win_px // scale)
    lo = max(0, int(round(cy_s)) - hw)
    hi = min(sh, int(round(cy_s)) + hw + 1)
    if lo >= hi:
        return False
    seg = fill_0[lo:hi]
    if float(seg.max()) >= 0.88:
        return True
    min_run = max(3, 12 // scale)
    flat = (seg >= 0.80).astype(np.int8)
    if int(flat.sum()) >= min_run:
        padded = np.pad(flat, (1, 1))
        starts = np.where(np.diff(padded) == 1)[0]
        ends = np.where(np.diff(padded) == -1)[0]
        if any((e - s) >= min_run for s, e in zip(starts, ends)):
            return True
    return False


def _compute_slanted_border_conf(gray_small, canny_small, s_row, e_row, slope, xs):
    """Gate 1B (Mandatory Slanted Panel-Border Edge Continuity):
    Measures diagonal panel-border continuity along the top border (outside s_row) and bottom border
    (outside e_row - 1) of a sheared candidate gutter run [s_row, e_row] in 4x downscaled coordinates.
    Searches outward 1..8 small rows (4..32px) to compensate for the 5x5 loc_std + 3x3 MORPH_OPEN erosion
    of the gutter mask, taking the sharpest single diagonal border line on each side.
    A column x counts as a border hit if |gray_center - gray_out| >= 28 OR a Canny edge lies within +-1 small row.
    Returns (border_conf, top_cont, bot_cont, top_bilat_ok, bot_bilat_ok).
    """
    sh, sw = gray_small.shape
    cols = np.arange(sw)
    q_left = max(1, int(0.40 * sw))
    q_right = min(sw - 1, int(0.60 * sw))
    cy_s = 0.5 * (s_row + e_row)
    y_mid = np.clip(np.rint(cy_s + slope * xs).astype(np.int32), 0, sh - 1)
    g_in = gray_small[y_mid, cols].astype(np.int16)

    def _edge_side(r_edge, dir_out):
        best_cont = 0.0
        best_bilat = False
        for off_out in range(1, 9):
            y_out = np.rint((r_edge + dir_out * off_out) + slope * xs).astype(np.int32)
            valid = (y_out >= 1) & (y_out < sh - 1)
            v_sum = int(valid.sum())
            if v_sum < max(10, int(0.50 * sw)):
                continue
            y_c = np.clip(y_out, 0, sh - 1)
            jump_hit = (np.abs(g_in - gray_small[y_c, cols].astype(np.int16)) >= 28)
            canny_hit = (
                canny_small[y_c, cols]
                | canny_small[np.clip(y_out - 1, 0, sh - 1), cols]
                | canny_small[np.clip(y_out + 1, 0, sh - 1), cols]
            )
            hit = (jump_hit | canny_hit) & valid
            cont = float(hit.sum()) / float(v_sum)
            if cont > best_cont:
                v_left = valid[:q_left]
                v_right = valid[q_right:]
                l_frac = float(hit[:q_left].sum()) / float(max(1, int(v_left.sum()))) if int(v_left.sum()) >= 4 else 0.0
                r_frac = float(hit[q_right:].sum()) / float(max(1, int(v_right.sum()))) if int(v_right.sum()) >= 4 else 0.0
                best_cont = cont
                best_bilat = (l_frac >= 0.35) and (r_frac >= 0.35)
        return best_cont, best_bilat

    top_cont, top_bilat = _edge_side(s_row, -1)
    bot_cont, bot_bilat = _edge_side(e_row - 1, +1)
    border_conf = max(top_cont, bot_cont)
    return border_conf, top_cont, bot_cont, top_bilat, bot_bilat


def _passes_slanted_border_gate(border_conf, top_cont, bot_cont, top_bilat, bot_bilat, gut_val):
    """Evaluates Gate 1B + Gate 1C black-gutter border requirement."""
    if gut_val == 0:
        # Gate 1C: black slanted gutters require strict >= 0.58 border continuity with bilateral support
        return (top_cont >= 0.58 and top_bilat) or (bot_cont >= 0.58 and bot_bilat)
    # White slanted gutters: >= 0.58 single border (bilateral) OR both >= 0.42 (at least one bilateral)
    if (top_cont >= 0.58 and top_bilat) or (bot_cont >= 0.58 and bot_bilat):
        return True
    if top_cont >= 0.42 and bot_cont >= 0.42 and (top_bilat or bot_bilat):
        return True
    return False


def find_slanted_gutters_b(strip_rgb, hard_mask=None, faces_2d=None, bubbles_2d=None,
                           angles=tuple(range(-26, 27, 2)),
                           scale=4, white_thresh=236, black_thresh=20,
                           min_fill=0.72, min_thick_px=24):
    """Phase 1/v2 Hardened Sheared-Angle Gutter Detector (-26..+26 deg in 2 deg steps) with:
    - Separate white (`bg_val=255`) and black (`bg_val=0`) polarity masks (no mixed white+black leaks).
    - Gate 1A (Zero-Degree Horizontal Dominance): rejects any deg != 0 if a valid >= 12px flat gutter
      or fill_0 >= 0.88 scanline of the same polarity exists within [cy - 140, cy + 140].
    - Gate 1B (Mandatory Slanted Panel-Border Edge Continuity): verifies continuous diagonal panel borders
      along the top/bottom sheared edges with bilateral left/right 40% support.
    - Gate 1C (Stricter Black-Gutter Threshold & +-12px Face Clearance): white deg != 0 requires fill >= 0.76;
      black deg != 0 requires fill >= 0.90, left/right >= 0.70, and border_conf >= 0.58.
    - Returns `(cy, deg, thick_px, score, bg_val, border_conf)`.
    """
    h, w = strip_rgb.shape[:2]
    sh, sw = max(1, h // scale), max(1, w // scale)
    small = cv2.resize(strip_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gf = gray.astype(np.float32)
    mu = cv2.blur(gf, (5, 5))
    loc_std = np.sqrt(np.maximum(0.0, cv2.blur(gf * gf, (5, 5)) - mu * mu))
    canny_small = (cv2.Canny(gray, 50, 150) > 0)

    is_white = ((gray >= white_thresh) & (loc_std <= 7.5)).astype(np.uint8)
    is_black = ((gray <= black_thresh) & (loc_std <= 7.5)).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    is_white = cv2.morphologyEx(is_white, cv2.MORPH_OPEN, k3)
    is_black = cv2.morphologyEx(is_black, cv2.MORPH_OPEN, k3)

    # Mask 2D face (+-12px) and dialogue (+-6px) boxes directly in 2D polarity masks
    if faces_2d or bubbles_2d:
        for b in list(faces_2d or []):
            bx1, by1, bx2, by2 = int(b[0]) // scale, int(b[1]) // scale, int(b[2]) // scale, int(b[3]) // scale
            is_white[max(0, by1 - 3):min(sh, by2 + 3), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
            is_black[max(0, by1 - 3):min(sh, by2 + 3), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
        for b in list(bubbles_2d or []):
            bx1, by1, bx2, by2 = int(b[0]) // scale, int(b[1]) // scale, int(b[2]) // scale, int(b[3]) // scale
            is_white[max(0, by1 - 2):min(sh, by2 + 2), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
            is_black[max(0, by1 - 2):min(sh, by2 + 2), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
    elif hard_mask is not None and len(hard_mask) == h:
        hm_small = cv2.resize(hard_mask.astype(np.uint8)[:, None], (1, sh), interpolation=cv2.INTER_NEAREST)[:, 0]
        is_white[hm_small > 0, :] = 0
        is_black[hm_small > 0, :] = 0

    # Precompute 0-deg horizontal gutter fill profiles for Gate 1A
    white_fill_0 = is_white.mean(axis=1)
    black_fill_0 = is_black.mean(axis=1)
    active_pols = []
    if int(is_white.sum()) >= max(200, 3 * sw):
        active_pols.append((is_white, white_fill_0, 255))
    if int(is_black.sum()) >= max(200, 3 * sw):
        active_pols.append((is_black, black_fill_0, 0))
    if not active_pols:
        return []

    xs = np.arange(sw) - (sw / 2.0)
    ys_base = np.arange(sh)[:, None]
    min_thick_std = max(4, min_thick_px // scale)
    min_thick_hi = max(3, 16 // scale)
    candidates = []

    def _slanted_line_hits_2d_box(cy_full, deg_val):
        slope_full = np.tan(np.deg2rad(deg_val))
        # Gate 1C: +-12px clearance on faces_2d, +-6px on dialogue_boxes_raw
        for b in list(faces_2d or []):
            bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            for x_test in (bx1, 0.5 * (bx1 + bx2), bx2):
                y_line = cy_full + slope_full * (x_test - (w / 2.0))
                if (by1 - 12.0) <= y_line <= (by2 + 12.0):
                    return True
        for b in list(bubbles_2d or []):
            bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            for x_test in (bx1, 0.5 * (bx1 + bx2), bx2):
                y_line = cy_full + slope_full * (x_test - (w / 2.0))
                if (by1 - 6.0) <= y_line <= (by2 + 6.0):
                    return True
        return False

    # Coarse-to-fine: sweep 6-deg steps on full-chapter strips (>12,000px), then refine +-2 deg around hits
    if len(angles) > 15:
        step_c = 6 if h > 12000 else 4
        sweep_angles = tuple(range(-24, 25, step_c))
    else:
        sweep_angles = angles

    def _eval_angle(deg):
        slope = np.tan(np.deg2rad(deg))
        y_idx = np.rint(ys_base + slope * xs[None, :]).astype(np.int32)
        valid = (y_idx >= 0) & (y_idx < sh)
        y_clip = np.clip(y_idx, 0, sh - 1)
        valid_cnt = np.maximum(valid.sum(axis=1), 1)
        q = max(1, sw // 5)
        out_c = []

        for mask_pol, fill_0_pol, gut_val in active_pols:
            sampled = mask_pol[y_clip, np.arange(sw)[None, :]] * valid
            row_fill = sampled.sum(axis=1) / valid_cnt
            if deg == 0:
                req_fill = min_fill
                req_lr = 0.42
            elif gut_val == 0:
                # Gate 1C: strict black slanted threshold
                req_fill = max(min_fill, 0.90)
                req_lr = 0.70
            else:
                req_fill = max(min_fill, 0.76)
                req_lr = 0.42

            left_ok = sampled[:, :q].mean(axis=1) >= req_lr
            right_ok = sampled[:, sw - q:].mean(axis=1) >= req_lr
            vb = (row_fill >= req_fill) & left_ok & right_ok

            padded = np.pad(vb.astype(np.int8), (1, 1))
            for s, e in zip(np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]):
                run_fill = float(row_fill[s:e].mean())
                req_thick = min_thick_hi if run_fill >= 0.85 else min_thick_std
                if (e - s) < req_thick:
                    continue
                cy_s = (s + e) / 2.0
                cy = int(cy_s * scale)
                if _slanted_line_hits_2d_box(cy, deg):
                    continue
                if deg != 0:
                    # Gate 1A: Zero-Degree Horizontal Dominance
                    if _has_zero_deg_flat_gutter(fill_0_pol, cy_s, scale=scale, half_win_px=140):
                        continue
                    # Gate 1B (+ Gate 1C for black): Mandatory Slanted Panel-Border Edge Continuity
                    b_conf, t_c, b_c, t_bi, b_bi = _compute_slanted_border_conf(gray, canny_small, s, e, slope, xs)
                    if not _passes_slanted_border_gate(b_conf, t_c, b_c, t_bi, b_bi, gut_val):
                        continue
                else:
                    b_conf = 1.0

                ang_bonus = 1.15 if deg == 0 else (0.85 + 0.30 * b_conf)
                score = float(run_fill * (e - s) * ang_bonus)
                out_c.append((cy, deg, int((e - s) * scale), score, gut_val, round(float(b_conf), 4)))
        return out_c

    hit_degs = set()
    for deg in sweep_angles:
        cs = _eval_angle(deg)
        if cs:
            candidates.extend(cs)
            hit_degs.add(deg)

    # Local +-2 deg refinement around any coarse hit (only on segments <= 12,000px; on full chapters Stage 4 refines >2400px segments!)
    if len(angles) > 15 and h <= 12000 and hit_degs:
        refine_degs = set()
        for hd in hit_degs:
            for d_off in (-2, 2):
                nd = hd + d_off
                if -26 <= nd <= 26 and nd not in sweep_angles:
                    refine_degs.add(nd)
        for deg in sorted(refine_degs):
            candidates.extend(_eval_angle(deg))

    candidates.sort(key=lambda c: c[3], reverse=True)
    kept = []
    for cand in candidates:
        if all(abs(cand[0] - k[0]) > 220 for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda c: c[0])


def find_corridor_gutters_2d_b(seg_rgb, faces_2d=None, dialogue_2d=None, scale=4,
                               white_thresh=236, black_thresh=20):
    """Phase 1/v2 Hardened 2D Connected-Corridor Gutter Finder for segments > 1,800px before DP runs.
    Detects any connected flat white or black component touching both `x <= 0.08*sw` and `x >= 0.92*sw`,
    fits a slanted line through its distance-transform ridge (`cv2.fitLine`), and enforces Gates 1A, 1B, and 1C
    for any corridor with `|deg| >= 2`.
    Returns list of `(cy, deg, thick_px, score, bg_val, border_conf)`.
    """
    h, w = seg_rgb.shape[:2]
    if h < 600 or w < 100:
        return []
    sh, sw = max(1, h // scale), max(1, w // scale)
    small = cv2.resize(seg_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gf = gray.astype(np.float32)
    mu = cv2.blur(gf, (5, 5))
    loc_std = np.sqrt(np.maximum(0.0, cv2.blur(gf * gf, (5, 5)) - mu * mu))
    canny_small = (cv2.Canny(gray, 50, 150) > 0)

    white_m = ((gray >= white_thresh) & (loc_std <= 8.0)).astype(np.uint8)
    black_m = ((gray <= black_thresh) & (loc_std <= 8.0)).astype(np.uint8)
    # Mask 2D face (+-12px) / dialogue (+-6px) boxes directly out of both corridor masks
    for b in list(faces_2d or []):
        bx1, by1, bx2, by2 = int(b[0]) // scale, int(b[1]) // scale, int(b[2]) // scale, int(b[3]) // scale
        white_m[max(0, by1 - 3):min(sh, by2 + 3), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
        black_m[max(0, by1 - 3):min(sh, by2 + 3), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
    for b in list(dialogue_2d or []):
        bx1, by1, bx2, by2 = int(b[0]) // scale, int(b[1]) // scale, int(b[2]) // scale, int(b[3]) // scale
        white_m[max(0, by1 - 2):min(sh, by2 + 2), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0
        black_m[max(0, by1 - 2):min(sh, by2 + 2), max(0, bx1 - 1):min(sw, bx2 + 1)] = 0

    white_fill_0 = white_m.mean(axis=1)
    black_fill_0 = black_m.mean(axis=1)
    xs = np.arange(sw) - (sw / 2.0)
    cols = np.arange(sw)
    q = max(1, sw // 5)

    def _hits_box(cy_full, deg_val):
        slope_full = np.tan(np.deg2rad(deg_val))
        for b in list(faces_2d or []):
            bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            for x_test in (bx1, 0.5 * (bx1 + bx2), bx2):
                y_line = cy_full + slope_full * (x_test - (w / 2.0))
                if (by1 - 12.0) <= y_line <= (by2 + 12.0):
                    return True
        for b in list(dialogue_2d or []):
            bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            for x_test in (bx1, 0.5 * (bx1 + bx2), bx2):
                y_line = cy_full + slope_full * (x_test - (w / 2.0))
                if (by1 - 6.0) <= y_line <= (by2 + 6.0):
                    return True
        return False

    found = []
    for mask_bin, fill_0_pol, bg_val in ((white_m, white_fill_0, 255), (black_m, black_fill_0, 0)):
        cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        for ic in range(1, n_cc):
            cx0 = int(stats[ic, cv2.CC_STAT_LEFT])
            cy0 = int(stats[ic, cv2.CC_STAT_TOP])
            cw = int(stats[ic, cv2.CC_STAT_WIDTH])
            ch_c = int(stats[ic, cv2.CC_STAT_HEIGHT])
            area = int(stats[ic, cv2.CC_STAT_AREA])
            # Must span from left edge (<= 8% sw) to right edge (>= 92% sw) and be a corridor (not a huge blank page)
            if cx0 <= int(0.08 * sw) and (cx0 + cw) >= int(0.92 * sw) and 3 <= ch_c <= int(0.45 * sh) and area >= int(3.5 * sw):
                comp = (labels == ic).astype(np.uint8)
                dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
                pts = []
                for col in range(sw):
                    col_d = dist[:, col]
                    if col_d.max() > 0:
                        pts.append([float(col), float(np.argmax(col_d))])
                if len(pts) < int(0.75 * sw):
                    continue
                pts_arr = np.array(pts, dtype=np.float32)
                vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                if abs(vx) < 1e-4:
                    continue
                deg = float(np.rad2deg(np.arctan2(vy, vx)))
                if abs(deg) > 26.0:
                    continue
                slope = vy / vx
                cy_s = y0 + slope * ((sw / 2.0) - x0)
                cy_full = int(round(cy_s * scale))
                if not (420 <= cy_full <= h - 420) or _hits_box(cy_full, deg):
                    continue
                pred_y = y0 + slope * (pts_arr[:, 0] - x0)
                med_res = float(np.median(np.abs(pts_arr[:, 1] - pred_y)))
                if med_res > max(6.0, 0.35 * ch_c):
                    continue

                thick_s = max(3, int(round(float(dist[comp > 0].mean()) * 2.0)))
                thick_px = int(max(16, thick_s * scale))
                deg_int = int(round(deg))
                b_conf = 1.0

                if abs(deg_int) >= 2:
                    # Gate 1A: Zero-Degree Horizontal Dominance
                    if _has_zero_deg_flat_gutter(fill_0_pol, cy_s, scale=scale, half_win_px=140):
                        continue
                    # Gate 1C: verify fill along fitted slanted centerline
                    y_line = np.rint(cy_s + slope * xs).astype(np.int32)
                    val_l = (y_line >= 0) & (y_line < sh)
                    if int(val_l.sum()) < int(0.60 * sw):
                        continue
                    smp_l = cleaned[np.clip(y_line, 0, sh - 1), cols] * val_l
                    line_fill = float(smp_l.sum()) / float(max(1, int(val_l.sum())))
                    req_f = 0.90 if bg_val == 0 else 0.76
                    req_lr = 0.70 if bg_val == 0 else 0.42
                    if line_fill < req_f or float(smp_l[:q].mean()) < req_lr or float(smp_l[sw - q:].mean()) < req_lr:
                        continue
                    # Gate 1B: Mandatory Slanted Panel-Border Edge Continuity
                    s_r = max(1, int(round(cy_s - thick_s / 2.0)))
                    e_r = min(sh - 1, max(s_r + 2, int(round(cy_s + thick_s / 2.0))))
                    b_conf, t_c, b_c, t_bi, b_bi = _compute_slanted_border_conf(gray, canny_small, s_r, e_r, slope, xs)
                    if not _passes_slanted_border_gate(b_conf, t_c, b_c, t_bi, b_bi, bg_val):
                        continue
                else:
                    deg_int = 0

                score = float(len(pts) / float(sw) * thick_px * (1.0 if deg_int == 0 else (0.85 + 0.30 * b_conf)))
                found.append((cy_full, deg_int, thick_px, score, bg_val, round(float(b_conf), 4)))

    found.sort(key=lambda c: c[3], reverse=True)
    kept = []
    for cand in found:
        if all(abs(cand[0] - k[0]) > 240 for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda c: c[0])


def is_bubble_only_panel_b(panel_rgb, bubble_boxes_local, faces_local=None, strict_tall=False):
    """Phase 1 Fix 4: Semantic Art-Ratio Classifier for Bubble-Only & Stacked-Bubble Panels.
    - When `faces_local` has any face, returns False immediately.
    - For tall panels (`strict_tall=True`, up to 1,150 px), requires `len(faces_local) == 0` AND
      `art_fraction < 0.05` AND `art_share_of_fg < 0.20` (measured strictly against verified dialogue/text boxes).
    - For compact panels (`strict_tall=False`, <= 450 px), returns True if `art_fraction < 0.06` or `art_share_of_fg < 0.22`.
    """
    h, w = panel_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return True
    if faces_local and len(faces_local) > 0:
        return False
    gray = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2GRAY)
    border_px = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_val = int(np.median(border_px))
    non_bg = np.abs(gray.astype(np.int16) - bg_val) > 18
    total_fg = int(non_bg.sum())
    if total_fg < 0.03 * h * w:
        return True

    text_mask = np.zeros((h, w), dtype=bool)
    for b in bubble_boxes_local:
        x1, y1, x2, y2 = b[:4]
        text_mask[max(0, int(y1) - 18):min(h, int(y2) + 18),
                  max(0, int(x1) - 18):min(w, int(x2) + 18)] = True

    art_pixels = int((non_bg & ~text_mask).sum())
    art_frac = art_pixels / float(h * w)
    art_share = art_pixels / float(max(total_fg, 1))
    if strict_tall:
        return (art_frac < 0.05) and (art_share < 0.20)
    return (art_frac < 0.06) or (art_share < 0.22)


def safe_fallback_cut_b(row_cost, lo, hi, hard_mask, max_expand=350):
    """Fix 4: Deadlock-Safe Fallback Cut. Never cuts a hard_mask (face/bubble) row via
    blind argmin on 1e9. Returns None if fully blocked.
    """
    H = len(row_cost)
    lo, hi = max(0, int(lo)), min(H, int(hi))
    if hi > lo:
        valid = np.where(~hard_mask[lo:hi])[0] + lo
        if len(valid) > 0:
            return int(valid[np.argmin(row_cost[valid])])
    lo2, hi2 = max(0, lo - max_expand), min(H, hi + max_expand)
    if hi2 > lo2:
        valid2 = np.where(~hard_mask[lo2:hi2])[0] + lo2
        if len(valid2) > 0:
            dist = np.minimum(np.abs(valid2 - lo), np.abs(valid2 - hi))
            return int(valid2[np.argmin(row_cost[valid2] + 0.05 * dist)])
    return None


def _is_in_pure_gutter_band_b(pure_gut, y0, min_band=20):
    if pure_gut is None:
        return False
    H = len(pure_gut)
    if y0 < 0 or y0 >= H or not pure_gut[y0]:
        return False
    lo = y0
    while lo > 0 and pure_gut[lo - 1]:
        lo -= 1
    hi = y0 + 1
    while hi < H and pure_gut[hi]:
        hi += 1
    return (hi - lo) >= min_band


def compute_composite_row_cost_b(gray, hard_mask, faces=None, halfbodies=None, sal_mean=None,
                                 seam_ys=None, pure_gut=None):
    """Phase 1 Fix 3 + Chunk-Seam Immunity: Speed-Line, Border, Half-Body & Saliency-Aware Composite Row Cost:
    - Zeroes out fake 1-2px JPEG/resize Sobel_Y horiz_line_frac spikes within +-4px of non-gutter chunk seams
      before computing border_bonus, is_clean_break, and body_pen zero-outs.
    - Base: `VertDiff/25 + EdgeDensity*5 + Ink*0.35 - HorizBorderBonus`
    - Interior-artwork penalty: `+3.0` on rows that are neither a gutter (`gutter_fill < 0.50`) nor a horizontal border (`horiz_line_frac < 0.50`).
    - Half-body soft penalty: `+7.5` across `halfbody_v1.0_n.onnx` boxes (zeroed on verified gutters or `horiz_line_frac >= 0.55`).
    - Face-anchored neck/shoulder fallback penalty: `+8.0` on `[fy2, fy2 + 1.0*fh]` and `+3.5` on `[fy2 + 1.0*fh, fy2 + 2.2*fh]`
      whenever a face has no overlapping half-body box (also zeroed on verified gutters or `horiz_line_frac >= 0.55`).
    - U2-NetP saliency valley reward: `-2.0` where `sal_mean < 0.08`.
    """
    H, W = gray.shape[:2]
    gf = gray.astype(np.float32)
    vert_diff = np.abs(np.diff(gf, axis=0, append=gf[-1:])).mean(axis=1)
    vert_diff_s = cv2.medianBlur(vert_diff.astype(np.float32)[:, None], 5)[:, 0]
    edges = (cv2.Canny(gray, 60, 160) > 0).mean(axis=1).astype(np.float32)

    sobel_y = np.abs(cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3))
    sobel_x = np.abs(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3))
    horiz_line_frac = ((sobel_y > 35.0) & (sobel_y > 1.8 * sobel_x)).mean(axis=1).astype(np.float32)
    if seam_ys:
        for sy in seam_ys:
            if not _is_in_pure_gutter_band_b(pure_gut, int(sy), min_band=20):
                horiz_line_frac[max(0, int(sy) - 4):min(H, int(sy) + 5)] = 0.0
    border_bonus = np.where(horiz_line_frac >= 0.60, horiz_line_frac * 2.5, 0.0)

    ink = (gf < 205).mean(axis=1).astype(np.float32)
    gutter_fill = np.maximum((gray >= 238).mean(axis=1), (gray <= 20).mean(axis=1)).astype(np.float32)
    is_clean_break = (gutter_fill >= 0.85) | (horiz_line_frac >= 0.55)
    interior_art_pen = np.where((horiz_line_frac < 0.50) & (gutter_fill < 0.50), 3.0, 0.0).astype(np.float32)

    body_pen = np.zeros(H, dtype=np.float32)
    hb_list = list(halfbodies or [])
    for hb in hb_list:
        by1, by2 = max(0, int(hb[1])), min(H, int(hb[3]))
        if by2 > by1:
            body_pen[by1:by2] = np.maximum(body_pen[by1:by2], 7.5)

    for f in (faces or []):
        fx1, fy1, fx2, fy2 = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        fh = max(24, fy2 - fy1)
        has_hb = any(
            (min(fx2, hb[2]) - max(fx1, hb[0]) > 0) and (min(fy2 + fh, hb[3]) - max(fy1, hb[1]) > 0)
            for hb in hb_list
        )
        if not has_hb:
            n0, n1 = max(0, fy2), min(H, fy2 + int(1.0 * fh))
            s1 = min(H, fy2 + int(2.2 * fh))
            if n1 > n0:
                body_pen[n0:n1] = np.maximum(body_pen[n0:n1], 8.0)
            if s1 > n1:
                body_pen[n1:s1] = np.maximum(body_pen[n1:s1], 3.5)

    body_pen[is_clean_break] = 0.0

    sal_bonus = np.zeros(H, dtype=np.float32)
    if sal_mean is not None and len(sal_mean) == H:
        sal_bonus = np.where(sal_mean < 0.08, -2.0, 0.0).astype(np.float32)

    raw_cost = (vert_diff_s / 25.0) + (edges * 5.0) + (ink * 0.35) - border_bonus + interior_art_pen + body_pen + sal_bonus
    cost = cv2.medianBlur(raw_cost.astype(np.float32)[:, None], 5)[:, 0]
    return cost, vert_diff_s, horiz_line_frac


def solve_dp_cuts_b(row_cost, hard_mask, a, b, target_h, min_h, max_h):
    """Fix 4 (Research Track B): 1D Dynamic Programming (Viterbi) solver that balances
    heights across a tall segment [a, b] while strictly respecting hard_mask.
    """
    span = b - a
    if span <= max_h:
        return [[a, b]]

    step = 16
    nodes = list(range(a, b, step))
    if nodes[-1] != b:
        nodes.append(b)
    N = len(nodes)
    dp = np.full(N, 1e18, dtype=np.float64)
    parent = np.full(N, -1, dtype=np.int32)
    dp[0] = 0.0

    for i in range(N):
        if dp[i] >= 1e17:
            continue
        yi = nodes[i]
        for j in range(i + 1, N):
            yj = nodes[j]
            dy = yj - yi
            if dy < min_h:
                continue
            if dy > max_h and j < N - 1:
                break
            if j < N - 1:
                r0, r1 = max(a, yj - 8), min(b, yj + 9)
                if hard_mask[r0:r1].all():
                    continue
                v_local = np.where(~hard_mask[r0:r1])[0] + r0
                cut_c = float(row_cost[v_local].min())
            else:
                cut_c = 0.0
            bal_c = 1.2 * (((dy - target_h) / float(target_h)) ** 2)
            if dy > max_h:
                bal_c += 8.0 * ((dy - max_h) / float(target_h))
            total_c = dp[i] + cut_c + bal_c
            if total_c < dp[j]:
                dp[j] = total_c
                parent[j] = i

    if parent[N - 1] == -1:
        out = []
        cur_a = a
        while b - cur_a > max_h:
            lo = cur_a + min_h
            hi = min(cur_a + max_h, b - min_h)
            if hi <= lo:
                break
            k = safe_fallback_cut_b(row_cost, lo, hi, hard_mask, max_expand=350)
            if k is None or k <= cur_a + 120 or b - k < 120:
                break
            out.append([cur_a, k])
            cur_a = k
        out.append([cur_a, b])
        return out

    path = []
    curr = N - 1
    while curr > 0:
        prev = int(parent[curr])
        y_start, y_end = nodes[prev], nodes[curr]
        if curr < N - 1:
            r_lo = max(y_start + min_h, y_end - 8)
            r_hi = min(b - min_h, y_end + 9)
            if r_hi > r_lo:
                v_idx = np.where(~hard_mask[r_lo:r_hi])[0] + r_lo
                if len(v_idx) > 0:
                    y_end = int(v_idx[np.argmin(row_cost[v_idx])])
                    nodes[curr] = y_end
        path.append([y_start, y_end])
        curr = prev
    path.reverse()
    for k in range(len(path) - 1):
        path[k + 1][0] = path[k][1]
    return path


def is_intentional_vertical_pan_b(crop_rgb, faces_local, bubbles_local, slanted_cuts_local,
                                  row_cost_local, vert_diff_local, horiz_line_local,
                                  u2netp_sal=None, seam_ys_local=None):
    """Phase 1 Fix 7 + U2-NetP Saliency Continuity Classifier for Intentional Vertical Pan Shots:
    - Uses `u2netp_sal = (sal_row_max, sal_row_mean, max_connected_span_frac)` when provided:
      * If a single connected salient subject (`d0 >= 0.45`) spans `>= 65%` of segment height continuously
        without dropping to a low-saliency valley (`sal_mean < 0.08` for `>= 24 px`) inside `[380..h-380]`,
        strongly confirms `vertical_pan` (unless crisp horizontal/slanted gutters or multi-character faces exist).
      * Conversely, if `sal_mean < 0.08` for `>= 24 px` across a clean horizontal valley inside `[380..h-380]`,
        rejects `vertical_pan` so DP can split cleanly at the valley.
    """
    h, w = crop_rgb.shape[:2]
    if h >= 6000:
        return False
    if len(slanted_cuts_local) > 0:
        return False

    # U2-NetP saliency continuity check (Escalator Model 2):
    # - If a single connected salient subject (`d0 >= 0.45`) spans `>= 65%` of segment height continuously,
    #   strongly confirm `vertical_pan` (locks in flame dragons like #16 with span_frac=0.734).
    # - For valley rejection: because U2-NetP trained on natural photos can produce 0 saliency over dark/smoky
    #   or flame-textured manhwa pan backgrounds, ONLY reject a pan on a saliency valley (`sal_mean < 0.08` for >= 24px)
    #   when that valley ALSO coincides with a low-edge/low-variance visual break (`r_edg <= 0.015` and `r_std <= 18.0`),
    #   preventing natural-photo domain shift from shattering continuous dark/smoke pan reveals!
    u2_strong_pan = False
    if u2netp_sal is not None:
        sal_max, sal_mean, span_frac = u2netp_sal
        if sal_mean is not None and len(sal_mean) == h and h > 800:
            if span_frac >= 0.65:
                u2_strong_pan = True

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if (not u2_strong_pan) and u2netp_sal is not None:
        sal_max, sal_mean, _span_frac = u2netp_sal
        if sal_mean is not None and len(sal_mean) == h and h > 800:
            gf_u = gray.astype(np.float32)
            r_std_u = gf_u.std(axis=1)
            r_edg_u = (edges > 0).mean(axis=1)
            low_sal = (sal_mean < 0.08) & (r_std_u <= 18.0) & (r_edg_u <= 0.015)
            pad_ls = np.pad(low_sal.astype(np.int8), (1, 1))
            ls_runs = [e - s for s, e in zip(np.where(np.diff(pad_ls) == 1)[0], np.where(np.diff(pad_ls) == -1)[0])
                       if (e - s) >= 24 and 380 < (s + e) // 2 < h - 380]
            if len(ls_runs) > 0:
                return False
    # Require straight panel divider spanning >= 68% W AND low local edge clutter (< 0.075) around it
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=int(0.55 * w), minLineLength=int(0.68 * w), maxLineGap=14)
    if lines is not None:
        for ln in lines[:, 0]:
            lx1, ly1, lx2, ly2 = ln
            dx = abs(lx2 - lx1)
            if dx >= int(0.68 * w):
                deg = abs(np.rad2deg(np.arctan2(ly2 - ly1, lx2 - lx1)))
                cy = int((ly1 + ly2) // 2)
                if seam_ys_local and any(abs(cy - sy) <= 4 for sy in seam_ys_local):
                    continue
                if 420 <= cy <= h - 420 and deg <= 22:
                    if float((edges[max(0, cy - 40):min(h, cy + 40)] > 0).mean()) <= 0.075:
                        return False

    # Check Sobel horizontal panel border line:
    # (a) horiz_line_local >= 0.58 inside [380..h-380] with clean surround (edge density <= 0.085), OR
    # (b) unmistakable full-width straight horizontal panel divider (horiz_line_local >= 0.78 inside middle [0.20*h..0.80*h])
    if horiz_line_local is not None and len(horiz_line_local) == h and h > 800:
        hl_inner = horiz_line_local[380:h - 380]
        if float(hl_inner.max()) >= 0.58:
            best_hy = int(np.argmax(hl_inner)) + 380
            if float((edges[max(0, best_hy - 30):min(h, best_hy + 30)] > 0).mean()) <= 0.085:
                return False
        m1, m2 = max(600, int(0.20 * h)), min(h - 600, int(0.80 * h))
        if m2 > m1 and float(horiz_line_local[m1:m2].max()) >= 0.78:
            return False

    # Require high-confidence faces (>= 0.45) separated by >= 650px for multi-character-beat rejection
    strong_faces = [f for f in faces_local if (len(f) < 5 or f[4] >= 0.45)]
    if len(strong_faces) >= 2:
        f_ys = sorted([int((f[1] + f[3]) // 2) for f in strong_faces])
        if (f_ys[-1] - f_ys[0]) >= 650:
            return False

    # If U2-NetP strongly confirmed a single continuous salient subject (>= 65% height, zero saliency valley)
    # and none of the hard structural dividers (slanted gutters, Hough/Sobel borders, multi-character faces) fired, lock in pan!
    if u2_strong_pan:
        return True

    # Check for crisp mid-strip color/scene cut (Bhattacharyya >= 0.65 inside middle 50% [0.22*h .. 0.78*h])
    if h >= 3500:
        hsv_s = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        m_lo, m_hi = max(700, int(0.22 * h)), min(h - 700, int(0.78 * h))
        for ym in range(m_lo, m_hi, 80):
            h1 = cv2.calcHist([hsv_s[ym - 300:ym]], [0, 1], None, [16, 8], [0, 180, 0, 256])
            h2 = cv2.calcHist([hsv_s[ym:ym + 300]], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            if float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)) >= 0.65:
                return False

    gf = gray.astype(np.float32)
    r_std = gf.std(axis=1)
    vdiff_col = np.abs(np.diff(gf, axis=0, append=gf[-1:]))
    hdiff_col = np.abs(np.diff(gf, axis=1, append=gf[:, -1:]))
    is_speed = ((vdiff_col < 2.2).mean(axis=1) >= 0.85) & (r_std >= 20.0) & ((hdiff_col > 5.0).mean(axis=1) >= 0.45)
    pad_sp = np.pad(is_speed.astype(np.int8), (1, 1))
    sp_runs = [e - s for s, e in zip(np.where(np.diff(pad_sp) == 1)[0], np.where(np.diff(pad_sp) == -1)[0])
               if (e - s) >= 140 and 600 < (s + e) // 2 < h - 600]
    if len(sp_runs) >= 1:
        return False

    # Relaxed quiet-band threshold: r_std <= 8.0 and r_edg <= 0.006 for >= 24px
    r_edg = (edges > 0).mean(axis=1)
    quiet_band = (r_std <= 8.0) & (r_edg <= 0.006)
    pad_qb = np.pad(quiet_band.astype(np.int8), (1, 1))
    q_runs = [e - s for s, e in zip(np.where(np.diff(pad_qb) == 1)[0], np.where(np.diff(pad_qb) == -1)[0])
              if (e - s) >= 24 and 360 < (s + e) // 2 < h - 360]
    if len(q_runs) > 0:
        return False

    # Also check for multi-panel white/diagonal gutter transition across tall segments:
    if h >= 2500:
        inner_white = (gray[350:h - 350] >= 240).mean(axis=1) >= 0.92
        pad_iw = np.pad(inner_white.astype(np.int8), (1, 1))
        iw_runs = [e - s for s, e in zip(np.where(np.diff(pad_iw) == 1)[0], np.where(np.diff(pad_iw) == -1)[0]) if (e - s) >= 6]
        if len(iw_runs) > 0:
            return False
        # Check if a >= 85% diagonal white/black seam exists across any angle in (-24..+24) inside [380..h-380]
        sh, sw = max(1, h // 4), max(1, w // 4)
        sm = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        mu = cv2.blur(sm, (5, 5))
        lstd = np.sqrt(np.maximum(0.0, cv2.blur(sm * sm, (5, 5)) - mu * mu))
        ig = (((sm >= 232) | (sm <= 24)) & (lstd <= 11.0)).astype(np.uint8)
        xs = np.arange(sw) - (sw / 2.0)
        ys_b = np.arange(sh)[:, None]
        for deg in (-24, -18, -12, -6, 0, 6, 12, 18, 24):
            sl = np.tan(np.deg2rad(deg))
            yi = np.rint(ys_b + sl * xs[None, :]).astype(np.int32)
            val = (yi >= 0) & (yi < sh)
            smp = ig[np.clip(yi, 0, sh - 1), np.arange(sw)[None, :]] * val
            rf = smp.sum(axis=1) / np.maximum(val.sum(axis=1), 1)
            rf4 = cv2.blur(rf[:, None], (1, 4))[95:max(96, sh - 95), 0]
            if rf4.size > 0 and float(rf4.max()) >= 0.85:
                return False

    return True


def trim_hybrid_credits_or_drop_b(crop_rgb, is_edge_frame=False, is_tail_end=False, bubbles_local=None, ocr_hint=""):
    """Fix 8 (Issue 2 Hardened): Multilingual Credit Stripping & Hybrid Edge Trimming.
    - Splits `_STRICT_PROMO_RE` (scanlation/ad/publisher banners) from `_AUTHOR_CREDIT_RE`
      (`작화|각색|원작|그림`): NEVER drops a whole chapter-title story panel on `_AUTHOR_CREDIT_RE` alone!
    - Always runs RapidOCR on the last 2–3 frames (`is_tail_end=True`) regardless of height so tall
      (> 850 px) end-of-chapter scanlator banners are never missed.
    - Returns `(trimmed_rgb, top_trimmed, bot_trimmed, is_credit_or_trimmed_flag, should_drop_entirely, cached_ocr_text)`
      so the inward edge-walk never invokes RapidOCR twice on the same frame.
    """
    h, w = crop_rgb.shape[:2]
    if h < 40 or w < 40:
        return crop_rgb, 0, 0, True, True, ""

    # 1. Structural multi-line edge-flush watermark trimmer
    disp, wt, wb = mp._strip_edge_watermark(crop_rgb, text_boxes=bubbles_local)
    top_trim, bot_trim = wt, wb
    curr_rgb = disp
    ch, cw = curr_rgb.shape[:2]
    was_credit_touched = (wt > 0 or wb > 0)

    # 2. Check for internal bottom gap in the lower half [0.22*ch .. 0.95*ch]
    g_curr = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    r_std = g_curr.std(axis=1)
    r_white = ((g_curr >= 240).mean(axis=1) >= 0.92) & (r_std <= 14.0)
    r_black = ((g_curr <= 20).mean(axis=1) >= 0.92) & (r_std <= 14.0)
    r_gap = r_white | r_black

    pad_g = np.pad(r_gap.astype(np.int8), (1, 1))
    gaps = [(int(s), int(e)) for s, e in zip(np.where(np.diff(pad_g) == 1)[0], np.where(np.diff(pad_g) == -1)[0])
            if (e - s) >= 30 and int(0.22 * ch) <= s and e <= int(0.95 * ch)]

    cached_ocr = ocr_hint or ""

    if gaps and is_edge_frame:
        g_s, g_e = gaps[-1]
        bot_part = curr_rgb[g_e:]
        top_part = curr_rgb[:g_s]
        if bot_part.shape[0] >= 24 and top_part.shape[0] >= 120:
            # Run RapidOCR specifically on the bottom candidate block; allow footer trim on _STRICT_PROMO_RE OR _AUTHOR_CREDIT_RE
            bot_ocr = ocr_crop_rapidocr_b(bot_part)
            combined_bot_text = f"{bot_ocr} {cached_ocr}".strip()
            if _STRICT_PROMO_RE.search(combined_bot_text) or (is_tail_end and _AUTHOR_CREDIT_RE.search(combined_bot_text)):
                extra_bot = ch - g_s
                bot_trim += extra_bot
                curr_rgb = top_part
                ch = curr_rgb.shape[0]
                was_credit_touched = True

    # 3. Check if the ENTIRE remaining frame is a standalone credit/promo card
    if is_edge_frame:
        if mp._looks_like_credits_panel(curr_rgb):
            return curr_rgb, top_trim, bot_trim, True, True, cached_ocr
        # Always run OCR on the last 2-3 tail frames regardless of height (catches tall >850px scanlator banners),
        # or on compact/bubble-only edge frames.
        should_ocr = is_tail_end or (ch <= 850) or is_bubble_only_panel_b(curr_rgb, bubbles_local or [])
        if should_ocr and not cached_ocr:
            cached_ocr = ocr_crop_rapidocr_b(curr_rgb)
        if cached_ocr:
            # STRICT promo/scanlator/publisher regex drops standalone promo/credit cards
            if _STRICT_PROMO_RE.search(cached_ocr):
                if is_tail_end or ch <= 850 or is_bubble_only_panel_b(curr_rgb, bubbles_local or []):
                    return curr_rgb, top_trim, bot_trim, True, True, cached_ocr
            # NEVER drop a chapter-start title card on _AUTHOR_CREDIT_RE (작화|각색|원작|그림)!
            # Only drop on _AUTHOR_CREDIT_RE if at the very end of the chapter AND artless/zero story dialogue.
            elif is_tail_end and _AUTHOR_CREDIT_RE.search(cached_ocr):
                if (not bubbles_local) and is_bubble_only_panel_b(curr_rgb, []):
                    return curr_rgb, top_trim, bot_trim, True, True, cached_ocr

    return curr_rgb, top_trim, bot_trim, was_credit_touched, False, cached_ocr


def trim_content_span_b(strip_rgb, y1, y2, hard_mask, keep_margin=16):
    """Fix 3: Clean Gutter Trimming — store tight content spans [y_start, y_end] with
    uniform 12-20px margins rather than midpoint cuts that leave 200-450px of dead whitespace,
    while NEVER trimming into hard_mask (faces/bubbles).
    """
    H, W = strip_rgb.shape[:2]
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if y2 - y1 < 60:
        return y1, y2
    crop = strip_rgb[y1:y2]
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ch = g.shape[0]
    r_std = g.std(axis=1)
    is_bg = (((g >= 242).mean(axis=1) >= 0.96) | ((g <= 18).mean(axis=1) >= 0.96)) & (r_std <= 14.0)
    hm_local = hard_mask[y1:y2]
    is_bg = is_bg & (~hm_local)

    t = 0
    while t < ch and is_bg[t]:
        t += 1
    b = 0
    while b < ch and is_bg[ch - 1 - b]:
        b += 1

    t = max(0, t - keep_margin)
    b = max(0, b - keep_margin)
    if ch - t - b < 60:
        return y1, y2
    return y1 + t, y2 - b


def mask_slanted_corners_b(crop_rgb, angle_top, angle_bot, gutter_val_top=None, gutter_val_bot=None,
                           cy_top_local=None, cy_bot_local=None):
    """Pre-Merge Patch 1.1: Exact Centerline-Anchored Slanted Corner Masking.
    - When `cy_top_local = cy_top - fy1` is provided and `angle_top != 0`, computes the exact slanted line
      passing through `(w/2.0, cy_top_local)` with slope `tan(radians(angle_top))`:
      `y_left = cy_top_local - 0.5*w*tan(theta)`, `y_right = cy_top_local + 0.5*w*tan(theta)`
      and fills the exact opposing polygon above that line (clipped to `[0, h]`) using `bg_top`.
    - Applies the symmetric formula for `cy_bot_local = cy_bot - fy1` when `angle_bot != 0`, filling the
      opposing polygon below that line using `bg_bot`.
    - Falls back safely to edge-anchored `dy` when `cy_top_local`/`cy_bot_local` is None (e.g., squeezed merge).
    """
    if angle_top == 0 and angle_bot == 0:
        return crop_rgb
    out = crop_rgb.copy()
    h, w = out.shape[:2]

    def _sample_line_bg(y_l, y_r):
        xs = np.linspace(0, w - 1, 64).astype(np.int32)
        ys = np.clip(np.rint(np.linspace(y_l, y_r, 64)).astype(np.int32), 0, h - 1)
        med = float(np.median(out[ys, xs]))
        return (255, 255, 255) if med >= 128.0 else (0, 0, 0)

    if angle_top != 0:
        half_dy = 0.5 * float(w) * math.tan(math.radians(float(angle_top)))
        if cy_top_local is not None:
            yl = float(cy_top_local) - half_dy
            yr = float(cy_top_local) + half_dy
        else:
            dy_f = abs(2.0 * half_dy)
            if not (4.0 <= dy_f < h / 3.0):
                yl = yr = -1.0
            else:
                yl, yr = (0.0, dy_f) if angle_top > 0 else (dy_f, 0.0)
        yl_c = max(0.0, min(float(h), yl))
        yr_c = max(0.0, min(float(h), yr))
        if max(yl_c, yr_c) >= 4.0 and max(yl_c, yr_c) < h * 0.45:
            bg_top = (255, 255, 255) if gutter_val_top == 255 else ((0, 0, 0) if gutter_val_top == 0 else _sample_line_bg(yl_c, yr_c))
            pts = np.array([[0, 0], [w, 0], [w, int(round(yr_c))], [0, int(round(yl_c))]], dtype=np.int32)
            cv2.fillPoly(out, [pts], bg_top)

    if angle_bot != 0:
        half_dy = 0.5 * float(w) * math.tan(math.radians(float(angle_bot)))
        if cy_bot_local is not None:
            yl = float(cy_bot_local) - half_dy
            yr = float(cy_bot_local) + half_dy
        else:
            dy_f = abs(2.0 * half_dy)
            if not (4.0 <= dy_f < h / 3.0):
                yl = yr = float(h + 1)
            else:
                yl, yr = (float(h) - dy_f, float(h)) if angle_bot > 0 else (float(h), float(h) - dy_f)
        yl_c = max(0.0, min(float(h), yl))
        yr_c = max(0.0, min(float(h), yr))
        if (h - min(yl_c, yr_c)) >= 4.0 and (h - min(yl_c, yr_c)) < h * 0.45:
            bg_bot = (255, 255, 255) if gutter_val_bot == 255 else ((0, 0, 0) if gutter_val_bot == 0 else _sample_line_bg(yl_c, yr_c))
            pts = np.array([[0, h], [w, h], [w, int(round(yr_c))], [0, int(round(yl_c))]], dtype=np.int32)
            cv2.fillPoly(out, [pts], bg_bot)

    return out


def run_slicer_b_on_chapter(page_paths, chapter_tag="chap_test", known_ocr_map=None,
                            precomputed_faces=None, face_dt=0.0, force_split_multi=False,
                            include_crops=False, return_crops=False,
                            return_dropped_ranges=False, include_ocr=False):
    """Executes the full Phase 1/v2 upgraded Slicer_B pipeline on a chapter's raw page paths:
    - Fixed RT-DETR (`W x 1.5W`, `stride_overlap=0.20`, `stroke_px >= 80`, classes `0/1/2`) + tightened Black-Hat/White-Hat.
    - Dual box lists (`dialogue_boxes_raw` for slanted/corridor gutters; `bubbles` for flat `hard_mask`).
    - Hardened 2-deg slanted gutter sweep + 2D connected-corridor gutter finder with Gates 1A/1B/1C (`border_conf`).
    - Selective Hard-Segment Escalator: `halfbody_v1.0_n.onnx` on segments > 1,800px + `u2netp.onnx` on segments > 2,400px.
    - Tall stacked bubble-only & near-empty SFX-strip merging up to 1,150px (`len(faces) == 0`).
    - Single-pass cached RapidOCR credit stripping with `_STRICT_PROMO_RE` vs `_AUTHOR_CREDIT_RE`.
    - Tracks `cut_src_top`/`cut_src_bot`, `border_conf_top`/`border_conf_bot`, and `dropped_y_ranges`.
    """
    keep_crops = bool(include_crops or return_crops)
    t0 = time.time()
    strip, spans = reconstruct_strip_b(page_paths)
    if strip is None:
        return ([], None, [], 0.0, []) if return_dropped_ranges else ([], None, [], 0.0)
    H, W = strip.shape[:2]
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)

    # Stage 2: Overlapping W x 1.5W Tiling (stride_overlap=0.20, fast_bub=False) + tightened morphology
    faces, bubbles, face_mask, bubble_mask, hard_mask, dialogue_boxes_raw = detect_faces_and_bubbles_b(
        strip, stride_overlap=0.20, fast_bub=False, precomputed_faces=precomputed_faces
    )

    # Stage 3: Horizontal + Hardened Sheared-Angle Gutter Detection
    MIN_GUT = 24
    MIN_PANEL = 180
    MAX_PANEL = max(1350, int(1.85 * W))
    TARGET_PANEL = max(850, int(1.15 * W))
    BUBBLE_BRIDGE = 560

    slanted_cands = find_slanted_gutters_b(
        strip, hard_mask=hard_mask, faces_2d=faces, bubbles_2d=dialogue_boxes_raw, min_thick_px=MIN_GUT
    )

    g16 = gray.astype(np.int16)
    std = g16.std(axis=1)
    ink = (g16 < 205).mean(axis=1)
    edges = (cv2.Canny(gray, 60, 160) > 0).mean(axis=1)
    white = ((g16 >= 240).mean(axis=1) >= 0.94) & (std <= 16.0)
    black = ((g16 <= 20).mean(axis=1) >= 0.94) & (std <= 14.0)
    pure_gut = (white | black) & (~hard_mask)
    flat_band = (std <= 12.0) & (edges <= 0.010) & (~hard_mask)
    gut = (white | black | flat_band) & (~hard_mask)

    raw_bands, i = [], 0
    while i < H:
        if not gut[i]:
            i += 1
            continue
        j = i
        while j < H and gut[j]:
            j += 1
        raw_bands.append([i, j])
        i = j

    # Bridge floating bubbles/captions across white gutters up to BUBBLE_BRIDGE = 560px
    bands = []
    for bd in raw_bands:
        if bands:
            g0, g1 = bands[-1][1], bd[0]
            if 0 < g1 - g0 <= BUBBLE_BRIDGE and not face_mask[g0:g1].any():
                solid_art = (ink[g0:g1] >= 0.18) & (~bubble_mask[g0:g1])
                if float(solid_art.mean()) < 0.04:
                    bands[-1][1] = bd[1]
                    continue
        bands.append(bd[:])

    def _has_verified_art_panel_between(y1_p, y2_p):
        lo_p, hi_p = min(y1_p, y2_p), max(y1_p, y2_p)
        if hi_p - lo_p < 350:
            return False
        return float(ink[lo_p:hi_p].mean()) >= 0.15

    # cut_points tuples: (y_cut, angle_deg, gutter_val_255_or_0_or_None, cut_source, border_conf)
    cut_points = []
    for (a, b) in bands:
        if int(gut[a:b].sum()) < MIN_GUT:
            continue
        valid_rows = np.where(~hard_mask[a:b])[0] + a
        if len(valid_rows) == 0:
            continue
        pad_v = np.pad((~hard_mask[a:b]).astype(np.int8), (1, 1))
        runs = [(int(s + a), int(e + a)) for s, e in zip(np.where(np.diff(pad_v) == 1)[0], np.where(np.diff(pad_v) == -1)[0])]
        best_run = max(runs, key=lambda r: r[1] - r[0])
        if best_run[1] - best_run[0] >= 12:
            y_c = (best_run[0] + best_run[1]) // 2
            gv = 255 if float(np.median(gray[y_c])) >= 128.0 else 0
            is_pure_c = bool(pure_gut[y_c]) or (max(float((gray[y_c] >= 238).mean()), float((gray[y_c] <= 20).mean())) >= 0.85)
            cut_src_c = "gutter" if is_pure_c else "flat_band"
            cut_points.append((int(y_c), 0, gv, cut_src_c, 1.0))

    # Gate 1A in Stage 3: reject any slanted cut within +-240px of a horizontal gutter cut
    # (and within +-360px unless a verified artwork panel >= 350px with ink >= 0.15 sits strictly between them)
    for (cy, deg, thick, sc, gval, bconf) in slanted_cands:
        if 450 <= cy <= H - 450:
            ok_sep = True
            for (existing_y, _, _, _, _) in cut_points:
                dist_y = abs(cy - existing_y)
                if dist_y < 240:
                    ok_sep = False
                    break
                if dist_y < 360 and not _has_verified_art_panel_between(cy, existing_y):
                    ok_sep = False
                    break
            if ok_sep:
                cut_points.append((int(cy), int(deg), int(gval), "slanted", float(bconf)))

    cut_points.sort(key=lambda x: x[0])

    bounds = [(0, 0, None, "edge", 1.0)] + cut_points + [(H, 0, None, "edge", 1.0)]
    # segs format: [y_a, y_b, ang_a, ang_b, layout, gv_a, gv_b, src_a, src_b, bc_a, bc_b]
    segs = []
    for k in range(len(bounds) - 1):
        y_a, ang_a, gv_a, src_a, bc_a = bounds[k]
        y_b, ang_b, gv_b, src_b, bc_b = bounds[k + 1]
        if y_b - y_a >= 80:
            segs.append([y_a, y_b, ang_a, ang_b, "panel", gv_a, gv_b, src_a, src_b, bc_a, bc_b])
        elif segs:
            segs[-1][1] = y_b
            segs[-1][3] = ang_b
            segs[-1][6] = gv_b
            segs[-1][8] = src_b
            segs[-1][10] = bc_b
    if not segs:
        segs = [[0, H, 0, 0, "panel", None, None, "edge", "edge", 1.0, 1.0]]

    # Stage 4 & 5: Structural Break Splitting (2-deg slanted + 2D Corridor Finder) + Selective Hard-Segment Escalator
    # (halfbody_v1.0_n.onnx on >1,800px + u2netp.onnx on >2,400px) + Neck/Torso-Safe Viterbi DP
    def _find_internal_structural_cuts(ya, yb, ang_a, ang_b, gv_a, gv_b, src_a, src_b, bc_a, bc_b, faces_loc, dial_loc):
        span_h = yb - ya
        crop_s = strip[ya:yb]
        sl_loc = [c for c in find_slanted_gutters_b(
                      crop_s, hard_mask=hard_mask[ya:yb],
                      faces_2d=faces_loc, bubbles_2d=dial_loc,
                      min_fill=0.72, min_thick_px=24)
                  if 420 <= c[0] <= span_h - 420]
        sl_cands = []
        for (scy, sdeg, _sth, _ssc, sgval, sbconf) in sl_loc:
            sl_cands.append((ya + scy, sdeg, sgval, "slanted" if sdeg != 0 else "gutter", float(sbconf)))

        # Phase 1 Fix 2: 2D Connected-Corridor Gutter Finder on segments > 1,800px BEFORE DP runs!
        if span_h > 1800:
            corr_loc = find_corridor_gutters_2d_b(crop_s, faces_2d=faces_loc, dialogue_2d=dial_loc)
            for (ccy, cdeg, _cth, _csc, cgval, cbconf) in corr_loc:
                if 420 <= ccy <= span_h - 420:
                    sl_cands.append((ya + ccy, cdeg, cgval, "corridor" if cdeg != 0 else "gutter", float(cbconf)))

        # Also check clean internal white/quiet transitions (>= 92% white for >= 6 rows or quiet band >= 24px)
        g_loc = gray[ya:yb]
        gf_loc = g_loc.astype(np.float32)
        rs_loc = gf_loc.std(axis=1)
        ed_loc = (cv2.Canny(g_loc, 50, 150) > 0).mean(axis=1)
        hm_loc = hard_mask[ya:yb]
        iw_mask = ((g_loc >= 240).mean(axis=1) >= 0.92) | ((rs_loc <= 8.0) & (ed_loc <= 0.006) & (~hm_loc))
        horiz_cands = []
        pad_iw = np.pad(iw_mask.astype(np.int8), (1, 1))
        for s_r, e_r in zip(np.where(np.diff(pad_iw) == 1)[0], np.where(np.diff(pad_iw) == -1)[0]):
            cy_r = (s_r + e_r) // 2
            if (e_r - s_r) >= 8 and 450 <= cy_r <= span_h - 450:
                if not hm_loc[max(0, s_r - 2):min(span_h, e_r + 2)].all():
                    valid_r = np.where(~hm_loc[s_r:e_r])[0] + s_r
                    cut_y_r = int(valid_r[len(valid_r) // 2]) if len(valid_r) > 0 else cy_r
                    gv_r = 255 if float(np.median(g_loc[cut_y_r])) >= 128.0 else 0
                    horiz_cands.append((ya + cut_y_r, 0, gv_r, "gutter", 1.0))

        # Ultra-tall >= 5500px flat-seam fallback (0 deg only so Gates 1A/1B/1C are never bypassed!)
        if span_h >= 5500:
            sm_h, sm_w = max(1, span_h // 4), max(1, W // 4)
            sm = cv2.resize(g_loc, (sm_w, sm_h), interpolation=cv2.INTER_AREA).astype(np.float32)
            mu = cv2.blur(sm, (5, 5))
            lstd = np.sqrt(np.maximum(0.0, cv2.blur(sm * sm, (5, 5)) - mu * mu))
            ig = (((sm >= 232) | (sm <= 24)) & (lstd <= 11.0)).astype(np.uint8)
            rf = ig.mean(axis=1)
            rf4 = cv2.blur(rf[:, None], (1, 4))[110:max(111, sm_h - 110), 0]
            if rf4.size > 0 and float(rf4.max()) >= 0.85:
                y_loc_d = int(np.argmax(rf4) + 110) * 4
                gv_d = 255 if float(np.median(g_loc[min(span_h - 1, y_loc_d)])) >= 128.0 else 0
                horiz_cands.append((ya + y_loc_d, 0, gv_d, "gutter", 1.0))

        # Gate 1A Horizontal Dominance: accept 0-deg horizontal cuts FIRST, then slanted cuts only if >= 240px away
        # (and >= 360px unless a verified >= 350px artwork panel sits strictly between them)
        horiz_cands.extend([c for c in sl_cands if c[1] == 0])
        horiz_cands.sort(key=lambda x: x[0])
        chosen = []
        for c in horiz_cands:
            if (c[0] - ya) >= 420 and (yb - c[0]) >= 420 and all(abs(c[0] - k[0]) >= 420 for k in chosen):
                chosen.append(c)

        sl_only = sorted([c for c in sl_cands if c[1] != 0], key=lambda x: x[4], reverse=True)
        for c in sl_only:
            gy = c[0]
            if (gy - ya) < 420 or (yb - gy) < 420:
                continue
            ok_c = True
            for k in chosen:
                dy_k = abs(gy - k[0])
                if dy_k < 240:
                    ok_c = False
                    break
                if dy_k < 420 and not _has_verified_art_panel_between(gy, k[0]):
                    ok_c = False
                    break
            if ok_c:
                chosen.append(c)

        chosen.sort(key=lambda x: x[0])
        sub_cuts = [(ya, ang_a, gv_a, src_a, bc_a)] + chosen + [(yb, ang_b, gv_b, src_b, bc_b)]
        return sub_cuts, sl_loc

    # Base chapter row cost with chunk-seam immunity, face-anchored neck/shoulder protection & interior-artwork penalty
    seam_ys = [int(y0) for (_, y0, _) in spans[1:] if 0 < y0 < H]
    nongut_seam_ys = [sy for sy in seam_ys if not _is_in_pure_gutter_band_b(pure_gut, sy, min_band=20)]
    row_cost_base, vert_diff_s, horiz_line_frac = compute_composite_row_cost_b(
        gray, hard_mask, faces=faces, seam_ys=seam_ys, pure_gut=pure_gut
    )

    def _run_dp_with_escalator(sa, sb_y, sang_a, sang_b, sgv_a, sgv_b, ssrc_a, ssrc_b, sbc_a, sbc_b, f_sub, sal_mean_sub=None):
        """Runs selective halfbody_v1.0_n.onnx (+7.5 soft torso penalty) + u2netp saliency valley (-2.0)
        on segments > 1,800px before executing solve_dp_cuts_b.
        """
        sub_h = sb_y - sa
        if sub_h > 1800 and (len(f_sub) > 0 or sub_h > 2400):
            hb_loc = detect_halfbodies_on_segment_b(strip[sa:sb_y], conf=0.35)
        else:
            hb_loc = []
        if hb_loc or (sal_mean_sub is not None):
            rc_loc = row_cost_base[sa:sb_y].copy()
            if hb_loc:
                sub_g = gray[sa:sb_y]
                g_fill_s = np.maximum((sub_g >= 238).mean(axis=1), (sub_g <= 20).mean(axis=1))
                is_brk_s = (g_fill_s >= 0.85) | (horiz_line_frac[sa:sb_y] >= 0.55)
                hb_pen = np.zeros(sub_h, dtype=np.float32)
                for hb in hb_loc:
                    by1, by2 = max(0, int(hb[1])), min(sub_h, int(hb[3]))
                    if by2 > by1:
                        hb_pen[by1:by2] = 7.5
                hb_pen[is_brk_s] = 0.0
                rc_loc += hb_pen
            if sal_mean_sub is not None and len(sal_mean_sub) == sub_h:
                rc_loc += np.where(sal_mean_sub < 0.08, -2.0, 0.0).astype(np.float32)
        else:
            rc_loc = row_cost_base[sa:sb_y]
        dp_parts = solve_dp_cuts_b(
            rc_loc, hard_mask[sa:sb_y], 0, sub_h,
            target_h=TARGET_PANEL, min_h=max(360, int(0.45 * W)), max_h=1950
        )
        out_s = []
        for pi_idx, (pa_l, pb_l) in enumerate(dp_parts):
            pa, pb = sa + pa_l, sa + pb_l
            pa_ang = sang_a if pi_idx == 0 else 0
            pb_ang = sang_b if pi_idx == len(dp_parts) - 1 else 0
            pa_gv = sgv_a if pi_idx == 0 else None
            pb_gv = sgv_b if pi_idx == len(dp_parts) - 1 else None
            pa_src = ssrc_a if pi_idx == 0 else "dp"
            pb_src = ssrc_b if pi_idx == len(dp_parts) - 1 else "dp"
            pa_bc = sbc_a if pi_idx == 0 else 1.0
            pb_bc = sbc_b if pi_idx == len(dp_parts) - 1 else 1.0
            out_s.append([pa, pb, pa_ang, pb_ang, "panel", pa_gv, pb_gv, pa_src, pb_src, pa_bc, pb_bc])
        return out_s

    split_segs = []
    for (y_a, y_b, ang_a, ang_b, lay, gv_a, gv_b, src_a, src_b, bc_a, bc_b) in segs:
        span_h = y_b - y_a
        if span_h <= 2400:
            split_segs.append([y_a, y_b, ang_a, ang_b, "panel", gv_a, gv_b, src_a, src_b, bc_a, bc_b])
            continue

        faces_loc = [(f[0], f[1] - y_a, f[2], f[3] - y_a, f[4] if len(f) > 4 else 0.8)
                     for f in faces if f[1] >= y_a and f[3] <= y_b]
        dial_loc = [(b[0], b[1] - y_a, b[2], b[3] - y_a)
                    for b in dialogue_boxes_raw if b[1] >= y_a and b[3] <= y_b]

        sub_cuts, _ = _find_internal_structural_cuts(
            y_a, y_b, ang_a, ang_b, gv_a, gv_b, src_a, src_b, bc_a, bc_b, faces_loc, dial_loc
        )

        for sk in range(len(sub_cuts) - 1):
            sa, sang_a, sgv_a, ssrc_a, sbc_a = sub_cuts[sk]
            sb_y, sang_b, sgv_b, ssrc_b, sbc_b = sub_cuts[sk + 1]
            sub_h = sb_y - sa
            if sub_h <= 2400:
                split_segs.append([sa, sb_y, sang_a, sang_b, "panel", sgv_a, sgv_b, ssrc_a, ssrc_b, sbc_a, sbc_b])
                continue

            sub_crop = strip[sa:sb_y]
            f_sub = [(f[0], f[1] - sa, f[2], f[3] - sa, f[4] if len(f) > 4 else 0.8)
                     for f in faces if f[1] >= sa and f[3] <= sb_y]
            d_sub = [(b[0], b[1] - sa, b[2], b[3] - sa)
                     for b in dialogue_boxes_raw if b[1] >= sa and b[3] <= sb_y]
            sl_sub = [c for c in find_slanted_gutters_b(
                          sub_crop, hard_mask=hard_mask[sa:sb_y],
                          faces_2d=f_sub, bubbles_2d=d_sub,
                          min_fill=0.72, min_thick_px=24)
                      if 450 <= c[0] <= sub_h - 450]

            if 3900 <= sub_h <= 6500:
                s_max_s, sal_mean_s, s_span_s = compute_u2netp_saliency_b(sub_crop)
                u2_tuple = (s_max_s, sal_mean_s, s_span_s) if sal_mean_s is not None else None
            else:
                sal_mean_s, u2_tuple = None, None

            seams_loc_sub = [sy - sa for sy in nongut_seam_ys if sa < sy < sb_y]
            if (not force_split_multi) and (sang_a == 0 and sang_b == 0) and (3900 <= sub_h <= 6200) and is_intentional_vertical_pan_b(
                sub_crop, f_sub, d_sub, sl_sub,
                row_cost_base[sa:sb_y], vert_diff_s[sa:sb_y], horiz_line_frac[sa:sb_y],
                u2netp_sal=u2_tuple, seam_ys_local=seams_loc_sub
            ):
                split_segs.append([sa, sb_y, sang_a, sang_b, "vertical_pan", sgv_a, sgv_b, ssrc_a, ssrc_b, sbc_a, sbc_b])
            elif (not force_split_multi) and (sang_a == 0) and (6200 < sub_h <= 9200) and (len(f_sub) <= 1):
                rc_win = row_cost_base[sa + 4400 : min(sb_y - 600, sa + 5800)]
                hm_win = hard_mask[sa + 4400 : min(sb_y - 600, sa + 5800)]
                valid_v = np.where(~hm_win)[0]
                pan_isolated = False
                if len(valid_v) > 0:
                    best_off = int(valid_v[np.argmin(rc_win[valid_v])])
                    if float(rc_win[best_off]) <= 3.70:
                        split_y = sa + 4400 + best_off
                        f_p1 = [(f[0], f[1] - sa, f[2], f[3] - sa, f[4] if len(f) > 4 else 0.8)
                                for f in faces if f[1] >= sa and f[3] <= split_y]
                        d_p1 = [(b[0], b[1] - sa, b[2], b[3] - sa)
                                for b in dialogue_boxes_raw if b[1] >= sa and b[3] <= split_y]
                        u2_p1 = compute_u2netp_saliency_b(strip[sa:split_y])
                        seams_loc_p1 = [sy - sa for sy in nongut_seam_ys if sa < sy < split_y]
                        if is_intentional_vertical_pan_b(
                            strip[sa:split_y], f_p1, d_p1, [],
                            row_cost_base[sa:split_y], vert_diff_s[sa:split_y], horiz_line_frac[sa:split_y],
                            u2netp_sal=u2_p1, seam_ys_local=seams_loc_p1
                        ):
                            split_segs.append([sa, split_y, sang_a, 0, "vertical_pan", sgv_a, None, ssrc_a, "dp", sbc_a, 1.0])
                            f_tail = [(f[0], f[1] - split_y, f[2], f[3] - split_y, f[4] if len(f) > 4 else 0.8)
                                      for f in faces if f[1] >= split_y and f[3] <= sb_y]
                            split_segs.extend(_run_dp_with_escalator(split_y, sb_y, 0, sang_b, None, sgv_b, "dp", ssrc_b, 1.0, sbc_b, f_tail))
                            pan_isolated = True
                if not pan_isolated:
                    split_segs.extend(_run_dp_with_escalator(sa, sb_y, sang_a, sang_b, sgv_a, sgv_b, ssrc_a, ssrc_b, sbc_a, sbc_b, f_sub, sal_mean_sub=sal_mean_s))
            else:
                split_segs.extend(_run_dp_with_escalator(sa, sb_y, sang_a, sang_b, sgv_a, sgv_b, ssrc_a, ssrc_b, sbc_a, sbc_b, f_sub, sal_mean_sub=sal_mean_s))

    # Stage 5b (Phase 1 Fix 4 + Gate 1D): Semantic Art-Ratio Classifier for Bubble-Only, Stacked Bubble,
    # and Near-Empty Pure-White/Black SFX-Only Strips (up to 1,150 px)
    def _is_artless_or_sliver(seg):
        ya, yb = seg[0], seg[1]
        sh = yb - ya
        if sh < MIN_PANEL:
            return True
        if sh > 1150:
            return False
        crop_s = strip[ya:yb]
        f_loc = [f for f in faces if f[3] > ya and f[1] < yb]
        if len(f_loc) > 0:
            return False
        b_loc = [(b[0], b[1] - ya, b[2], b[3] - ya) for b in dialogue_boxes_raw if b[3] > ya and b[1] < yb]
        if not b_loc:
            # Gate 1D: near-empty pure-white/black SFX-only strips (art_fraction < 0.03, 0 faces, 0 dialogue)
            g_s = cv2.cvtColor(crop_s, cv2.COLOR_RGB2GRAY)
            b_px = np.concatenate([g_s[0, :], g_s[-1, :], g_s[:, 0], g_s[:, -1]])
            bg_v = int(np.median(b_px))
            if (bg_v >= 235 or bg_v <= 20) and float((np.abs(g_s.astype(np.int16) - bg_v) > 18).mean()) < 0.03:
                return True
            b_loc = [(b[0], b[1] - ya, b[2], b[3] - ya) for b in bubbles if b[3] > ya and b[1] < yb]
        return is_bubble_only_panel_b(crop_s, b_loc, faces_local=f_loc, strict_tall=(sh > 450))

    idx_s = 0
    while idx_s < len(split_segs) and len(split_segs) > 1:
        if not _is_artless_or_sliver(split_segs[idx_s]):
            idx_s += 1
            continue
        ya, yb = split_segs[idx_s][0], split_segs[idx_s][1]
        cands_n = []
        if idx_s + 1 < len(split_segs):
            cands_n.append(idx_s + 1)
        if idx_s > 0:
            cands_n.append(idx_s - 1)
        merged_ok = False
        for n_idx in cands_n:
            comb_y1 = min(ya, split_segs[n_idx][0])
            comb_y2 = max(yb, split_segs[n_idx][1])
            if comb_y2 - comb_y1 <= 2350 or (yb - ya < MIN_PANEL):
                lo_i, hi_i = min(idx_s, n_idx), max(idx_s, n_idx)
                split_segs[lo_i] = [
                    split_segs[lo_i][0],
                    split_segs[hi_i][1],
                    split_segs[lo_i][2],
                    split_segs[hi_i][3],
                    split_segs[lo_i][4] if split_segs[lo_i][4] == "vertical_pan" else split_segs[hi_i][4],
                    split_segs[lo_i][5],
                    split_segs[hi_i][6],
                    split_segs[lo_i][7],
                    split_segs[hi_i][8],
                    split_segs[lo_i][9],
                    split_segs[hi_i][10],
                ]
                del split_segs[hi_i]
                merged_ok = True
                idx_s = max(0, lo_i)
                break
        if not merged_ok:
            idx_s += 1

    # Stage 6: Clean Gutter Trimming, Clamped Slanted Overlap Expansion (dy // 2), True-Color Corner Masking,
    # and Single-Pass Cached RapidOCR Edge Trimming + Inward Walk
    def _page_for(y):
        for (nm, y0, y1) in spans:
            if y0 <= y < y1:
                return nm, y0
        return (spans[-1][0], spans[-1][1]) if spans else ("001.jpg", 0)

    n_segs = len(split_segs)
    raw_frames = []
    dropped_y_ranges = []
    for s_idx, (ya, yb, ang_a, ang_b, lay, gv_a, gv_b, src_a, src_b, bc_a, bc_b) in enumerate(split_segs):
        dy_top = int(abs(W * np.tan(np.deg2rad(ang_a)))) // 2 if ang_a != 0 else 0
        dy_bot = int(abs(W * np.tan(np.deg2rad(ang_b)))) // 2 if ang_b != 0 else 0
        ya_exp = max(0, ya - dy_top)
        yb_exp = min(H, yb + dy_bot)
        if dy_top > 0:
            for db in list(faces) + list(dialogue_boxes_raw):
                if db[3] <= ya and db[3] > ya_exp:
                    ya_exp = min(ya, int(db[3]) + 2)
                elif db[1] < ya and db[3] > ya_exp:
                    ya_exp = min(ya, int(db[3]) + 2)
        if dy_bot > 0:
            for db in list(faces) + list(dialogue_boxes_raw):
                if db[1] >= yb and db[1] < yb_exp:
                    yb_exp = max(yb, int(db[1]) - 2)
                elif db[3] > yb and db[1] < yb_exp:
                    yb_exp = max(yb, int(db[1]) - 2)

        if ang_a == 0 and ang_b == 0:
            ty1, ty2 = trim_content_span_b(strip, ya_exp, yb_exp, hard_mask, keep_margin=16)
        else:
            ty1, ty2 = ya_exp, yb_exp
            if ang_a == 0:
                ty1, _ = trim_content_span_b(strip, ya_exp, yb_exp, hard_mask, keep_margin=16)
            if ang_b == 0:
                _, ty2 = trim_content_span_b(strip, ya_exp, yb_exp, hard_mask, keep_margin=16)

        if ty2 - ty1 < 48:
            continue
        crop = strip[ty1:ty2, 0:W]
        gc = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if mp._is_blank_crop(gc):
            continue

        is_edge = (s_idx < 5) or (s_idx >= n_segs - 5)
        is_tail_end = (s_idx >= n_segs - 3)
        b_loc = [(b[0], b[1] - ty1, b[2], b[3] - ty1) for b in bubbles if b[3] > ty1 and b[1] < ty2]
        nm, y0 = _page_for(ty1)
        ocr_hint = (known_ocr_map or {}).get(nm, "")

        trimmed_crop, wt, wb, was_credit, drop_card, cached_ocr = trim_hybrid_credits_or_drop_b(
            crop, is_edge_frame=is_edge, is_tail_end=is_tail_end, bubbles_local=b_loc, ocr_hint=ocr_hint
        )
        if drop_card or trimmed_crop.shape[0] < 60:
            if drop_card:
                dropped_y_ranges.append((int(ty1), int(ty2)))
            continue

        fy1 = ty1 + wt
        fy2 = ty2 - wb
        eff_ang_a = ang_a if wt == 0 else 0
        eff_ang_b = ang_b if wb == 0 else 0
        eff_bc_a = float(bc_a) if eff_ang_a != 0 else 1.0
        eff_bc_b = float(bc_b) if eff_ang_b != 0 else 1.0
        eff_cy_top = int(ya) if eff_ang_a != 0 else int(fy1)
        eff_cy_bot = int(yb) if eff_ang_b != 0 else int(fy2)
        cy_top_loc = (eff_cy_top - fy1) if eff_ang_a != 0 else None
        cy_bot_loc = (eff_cy_bot - fy1) if eff_ang_b != 0 else None
        final_crop = mask_slanted_corners_b(
            trimmed_crop, eff_ang_a, eff_ang_b,
            gutter_val_top=gv_a, gutter_val_bot=gv_b,
            cy_top_local=cy_top_loc, cy_bot_local=cy_bot_loc
        )
        pi = int(spans and next((k for k, (_n, a, bb) in enumerate(spans) if a <= fy1 < bb), 0))
        nm, y0 = _page_for(fy1)

        raw_frames.append({
            "source_page": nm,
            "source_index": pi,
            "global_box": [0, int(fy1), int(W), int(fy2)],
            "source_box": [0, int(max(0, fy1 - y0)), int(W), int(fy2 - y0)],
            "w": int(W),
            "h": int(final_crop.shape[0]),
            "layout": lay,
            "angle_top": int(eff_ang_a),
            "angle_bot": int(eff_ang_b),
            "cy_top": int(eff_cy_top),
            "cy_bot": int(eff_cy_bot),
            "cut_src_top": src_a,
            "cut_src_bot": src_b,
            "border_conf_top": eff_bc_a,
            "border_conf_bot": eff_bc_b,
            "gutter_val_top": gv_a,
            "gutter_val_bot": gv_b,
            "was_credit": bool(was_credit),
            "ocr_text": cached_ocr,
            "crop_rgb": final_crop
        })

    # Inward walk from BOTH chapter ends using CACHED ocr_text (zero duplicate RapidOCR calls!)
    def _is_standalone_credit_frame(rf_item, is_tail=False):
        c_rgb = rf_item["crop_rgb"]
        if mp._looks_like_credits_panel(c_rgb):
            return True
        fy1, fy2 = rf_item["global_box"][1], rf_item["global_box"][3]
        b_l = [(b[0], b[1] - fy1, b[2], b[3] - fy1) for b in bubbles if b[3] > fy1 and b[1] < fy2]
        txt = rf_item.get("ocr_text", "")
        if not txt and (is_tail or c_rgb.shape[0] <= 850 or is_bubble_only_panel_b(c_rgb, b_l)):
            txt = ocr_crop_rapidocr_b(c_rgb)
            rf_item["ocr_text"] = txt
        if txt:
            if _STRICT_PROMO_RE.search(txt):
                if is_tail or c_rgb.shape[0] <= 850 or is_bubble_only_panel_b(c_rgb, b_l):
                    return True
            elif is_tail and _AUTHOR_CREDIT_RE.search(txt):
                if (not b_l) and is_bubble_only_panel_b(c_rgb, []):
                    return True
        return False

    consec_story = 0
    idx_walk = 0
    steps_fwd = 0
    while idx_walk < len(raw_frames) and steps_fwd < 5 and len(raw_frames) >= 2:
        if _is_standalone_credit_frame(raw_frames[idx_walk], is_tail=False):
            popped = raw_frames.pop(idx_walk)
            dropped_y_ranges.append((int(popped["global_box"][1]), int(popped["global_box"][3])))
            consec_story = 0
        else:
            consec_story += 1
            if consec_story >= 2:
                break
            idx_walk += 1
        steps_fwd += 1

    consec_story = 0
    steps_back = 0
    idx_back = len(raw_frames) - 1
    while idx_back >= 0 and steps_back < 5 and len(raw_frames) >= 2:
        if _is_standalone_credit_frame(raw_frames[idx_back], is_tail=True):
            popped = raw_frames.pop(idx_back)
            dropped_y_ranges.append((int(popped["global_box"][1]), int(popped["global_box"][3])))
            consec_story = 0
            idx_back -= 1
        else:
            consec_story += 1
            if consec_story >= 2:
                break
            idx_back -= 1
        steps_back += 1

    # Post-trim Bubble-Only & Stacked-Bubble (<= 1,150px) Merge Guard — RE-APPLIES mask_slanted_corners_b!
    k_f = 0
    while k_f < len(raw_frames) and len(raw_frames) > 1:
        rf = raw_frames[k_f]
        fy1, fy2 = rf["global_box"][1], rf["global_box"][3]
        fh = rf["h"]
        is_iso = (fh < MIN_PANEL)
        if not is_iso and fh <= 1150:
            f_loc = [f for f in faces if f[3] > fy1 and f[1] < fy2]
            if len(f_loc) == 0:
                sq_test = mp._squeeze_internal_whitespace(rf["crop_rgb"])
                b_loc = [(b[0], b[1] - fy1, b[2], b[3] - fy1) for b in dialogue_boxes_raw if b[3] > fy1 and b[1] < fy2]
                if not b_loc:
                    b_loc = [(b[0], b[1] - fy1, b[2], b[3] - fy1) for b in bubbles if b[3] > fy1 and b[1] < fy2]
                strict_t = (fh > 360)
                if is_bubble_only_panel_b(rf["crop_rgb"], b_loc, faces_local=f_loc, strict_tall=strict_t) or (
                    (not strict_t) and is_bubble_only_panel_b(sq_test, [], faces_local=f_loc, strict_tall=False)
                ):
                    is_iso = True
        if not is_iso:
            k_f += 1
            continue

        cand_neighbors = []
        if k_f + 1 < len(raw_frames):
            cand_neighbors.append((k_f + 1, raw_frames[k_f + 1]["global_box"][1] - fy2))
        if k_f > 0:
            cand_neighbors.append((k_f - 1, fy1 - raw_frames[k_f - 1]["global_box"][3]))
        cand_neighbors.sort(key=lambda t: t[1])

        merged_done = False
        for n_idx, _gap_d in cand_neighbors:
            lo_i, hi_i = min(k_f, n_idx), max(k_f, n_idx)
            ny1 = raw_frames[lo_i]["global_box"][1]
            ny2 = raw_frames[hi_i]["global_box"][3]
            raw_span_h = ny2 - ny1
            merged_crop = mp._squeeze_internal_whitespace(strip[ny1:ny2, 0:W])
            sq_h = int(merged_crop.shape[0])
            m_ang_t = raw_frames[lo_i]["angle_top"]
            m_ang_b = raw_frames[hi_i]["angle_bot"]
            m_cy_t = raw_frames[lo_i].get("cy_top", ny1)
            m_cy_b = raw_frames[hi_i].get("cy_bot", ny2)
            m_gv_t = raw_frames[lo_i].get("gutter_val_top")
            m_gv_b = raw_frames[hi_i].get("gutter_val_bot")
            use_cy_t = (m_cy_t - ny1) if (m_ang_t != 0 and sq_h == raw_span_h) else None
            use_cy_b = (m_cy_b - ny1) if (m_ang_b != 0 and sq_h == raw_span_h) else None
            merged_crop = mask_slanted_corners_b(
                merged_crop, m_ang_t, m_ang_b,
                gutter_val_top=m_gv_t, gutter_val_bot=m_gv_b,
                cy_top_local=use_cy_t, cy_bot_local=use_cy_b
            )
            if sq_h <= 2350 or ((ny2 - ny1) <= 2350) or (fh < MIN_PANEL and sq_h <= 2600):
                nm, y0 = _page_for(ny1)
                raw_frames[lo_i] = {
                    "source_page": nm,
                    "source_index": raw_frames[lo_i]["source_index"],
                    "global_box": [0, int(ny1), int(W), int(ny2)],
                    "source_box": [0, int(max(0, ny1 - y0)), int(W), int(ny2 - y0)],
                    "w": int(W),
                    "h": sq_h,
                    "layout": raw_frames[lo_i]["layout"] if raw_frames[lo_i]["layout"] == "vertical_pan" else raw_frames[hi_i]["layout"],
                    "angle_top": m_ang_t,
                    "angle_bot": m_ang_b,
                    "cy_top": int(m_cy_t),
                    "cy_bot": int(m_cy_b),
                    "cut_src_top": raw_frames[lo_i].get("cut_src_top", "gutter"),
                    "cut_src_bot": raw_frames[hi_i].get("cut_src_bot", "gutter"),
                    "border_conf_top": raw_frames[lo_i].get("border_conf_top", 1.0),
                    "border_conf_bot": raw_frames[hi_i].get("border_conf_bot", 1.0),
                    "gutter_val_top": m_gv_t,
                    "gutter_val_bot": m_gv_b,
                    "was_credit": raw_frames[lo_i].get("was_credit", False) or raw_frames[hi_i].get("was_credit", False),
                    "ocr_text": "",
                    "crop_rgb": merged_crop
                }
                del raw_frames[hi_i]
                k_f = max(0, lo_i)
                merged_done = True
                break
        if not merged_done:
            k_f += 1

    frames = []
    drop_keys = ["was_credit", "gutter_val_top", "gutter_val_bot"]
    if not keep_crops:
        drop_keys.append("crop_rgb")
    if not include_ocr:
        drop_keys.append("ocr_text")
    drop_set = set(drop_keys)
    for idx_f, rf in enumerate(raw_frames):
        rf_out = {k: v for k, v in rf.items() if k not in drop_set}
        rf_out["frame_id"] = f"{chapter_tag}_frame_{idx_f:05d}"
        rf_out["filename"] = f"frame_{idx_f:05d}.jpg"
        frames.append(rf_out)

    dt = (time.time() - t0) + face_dt
    if return_dropped_ranges:
        return frames, strip, spans, dt, dropped_y_ranges
    return frames, strip, spans, dt


# ============================================================================
# PHASE 1 SECTION 3: AUTOMATED QA CHECKS & FILTERED YOLO-SEG DATASET EXPORTER
# ============================================================================

def run_qa_checks_on_chapter_b(strip_rgb, frames, faces, dialogue_boxes, all_bubbles):
    """Computes the 3 new Phase 1 automated QA checks + face/bubble cuts on a sliced chapter:
    1. `NECK_CUT`: Flat cut `y` inside `[fy2, fy2 + 1.0*fh]` below a face (`conf >= 0.45`)
       when `y` is neither a gutter nor a horizontal border (`horiz_line_frac >= 0.55`).
    2. `MISSED_SLANTED_GUTTER`: Flat DP cut (`cut_src == "dp"`) through artwork when a clean
       left-to-right slanted/corridor gutter exists within `+-450 px`.
    3. `STACKED_BUBBLE_ONLY`: Any emitted frame (`<= 1,200 px`) with `0` faces and `< 5%` non-bubble art pixels.
    Returns dict with counts and per-frame flag sets.
    """
    H, W = strip_rgb.shape[:2]
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    gf = gray.astype(np.float32)
    sobel_y = np.abs(cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3))
    sobel_x = np.abs(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3))
    horiz_line_frac = ((sobel_y > 35.0) & (sobel_y > 1.8 * sobel_x)).mean(axis=1)
    gutter_fill = np.maximum((gray >= 238).mean(axis=1), (gray <= 20).mean(axis=1))

    # Pre-detect slanted gutters once on the chapter for MISSED_SLANTED_GUTTER check
    sl_ref = find_slanted_gutters_b(strip_rgb, faces_2d=faces, bubbles_2d=dialogue_boxes, min_fill=0.78, min_thick_px=20)
    sl_ys = [c[0] for c in sl_ref if c[1] != 0]

    strong_faces = [f for f in faces if (len(f) < 5 or f[4] >= 0.45)]
    neck_cuts = 0
    missed_slanted = 0
    stacked_bub_only = 0
    face_cuts = 0
    bubble_cuts = 0
    frame_flags = []

    for f in frames:
        fy1, fy2 = int(f["global_box"][1]), int(f["global_box"][3])
        fh_f = int(f["h"])
        flags = set()

        # Face cut & Bubble cut
        for (_, y1, _, y2, *_r) in faces:
            h_b = max(1.0, y2 - y1)
            if (y1 + 0.15 * h_b < fy1 < y2 - 0.15 * h_b) or (y1 + 0.15 * h_b < fy2 < y2 - 0.15 * h_b):
                flags.add("CUT_FACE")
                break
        for (_, y1, _, y2, *_r) in dialogue_boxes:
            h_b = max(1.0, y2 - y1)
            if (y1 + 0.15 * h_b < fy1 < y2 - 0.15 * h_b) or (y1 + 0.15 * h_b < fy2 < y2 - 0.15 * h_b):
                flags.add("CUT_BUBBLE")
                break

        # 1. NECK_CUT (only on flat boundaries: angle == 0)
        for y_cut, ang in ((fy1, f.get("angle_top", 0)), (fy2, f.get("angle_bot", 0))):
            if ang != 0 or y_cut <= 20 or y_cut >= H - 20:
                continue
            y_lo, y_hi = max(0, y_cut - 6), min(H, y_cut + 7)
            if float(gutter_fill[y_lo:y_hi].max()) >= 0.85 or float(horiz_line_frac[y_lo:y_hi].max()) >= 0.55:
                continue
            for (_, f_y1, _, f_y2, *_r) in strong_faces:
                fh_face = max(24.0, f_y2 - f_y1)
                if f_y2 <= y_cut <= (f_y2 + 1.0 * fh_face):
                    flags.add("NECK_CUT")
                    break

        # 2. MISSED_SLANTED_GUTTER (flat cut through artwork within +-450px of a clean slanted gutter)
        for y_cut, ang, src in ((fy1, f.get("angle_top", 0), f.get("cut_src_top", "")),
                                (fy2, f.get("angle_bot", 0), f.get("cut_src_bot", ""))):
            if ang == 0 and y_cut > 40 and y_cut < H - 40:
                y_lo, y_hi = max(0, y_cut - 6), min(H, y_cut + 7)
                is_art_cut = (float(gutter_fill[y_lo:y_hi].max()) < 0.80 and float(horiz_line_frac[y_lo:y_hi].max()) < 0.55)
                if (src == "dp" or (src == "" and is_art_cut)) and is_art_cut:
                    if any(abs(y_cut - sy) <= 450 for sy in sl_ys):
                        flags.add("MISSED_SLANTED_GUTTER")
                        break

        # 3. STACKED_BUBBLE_ONLY (emitted frame <= 1,200 px with 0 faces and < 5% non-bubble art pixels)
        if 120 <= fh_f <= 1200:
            f_loc = [fc for fc in faces if fc[3] > fy1 and fc[1] < fy2]
            if len(f_loc) == 0:
                b_loc = [(b[0], b[1] - fy1, b[2], b[3] - fy1) for b in (dialogue_boxes or all_bubbles) if b[3] > fy1 and b[1] < fy2]
                crop_f = strip_rgb[max(0, fy1):min(H, fy2)]
                if crop_f.size > 0 and is_bubble_only_panel_b(crop_f, b_loc, faces_local=f_loc, strict_tall=True):
                    flags.add("STACKED_BUBBLE_ONLY")

        if "CUT_FACE" in flags: face_cuts += 1
        if "CUT_BUBBLE" in flags: bubble_cuts += 1
        if "NECK_CUT" in flags: neck_cuts += 1
        if "MISSED_SLANTED_GUTTER" in flags: missed_slanted += 1
        if "STACKED_BUBBLE_ONLY" in flags: stacked_bub_only += 1
        frame_flags.append(flags)

    return {
        "face_cuts": face_cuts,
        "bubble_cuts": bubble_cuts,
        "neck_cuts": neck_cuts,
        "missed_slanted_gutter": missed_slanted,
        "stacked_bubble_only": stacked_bub_only,
        "frame_flags": frame_flags
    }


def _sutherland_hodgman_clip_y(poly, wy0, wy1):
    """Pre-Merge Patch 1.2: Exact Sutherland-Hodgman half-plane clipping of a 2D clockwise polygon
    against horizontal bounds `y >= wy0` and `y <= wy1`. Computes exact `x` intersection points
    whenever a diagonal edge crosses `y = wy0` or `y = wy1`.
    """
    def _clip_plane(pts, is_inside, intersect):
        if not pts:
            return []
        out = []
        prev = pts[-1]
        prev_in = is_inside(prev)
        for curr in pts:
            curr_in = is_inside(curr)
            if curr_in:
                if not prev_in:
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif prev_in:
                out.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        return out

    def _intersect_y(p1, p2, y_plane):
        x1, y1 = p1
        x2, y2 = p2
        dy = y2 - y1
        if abs(dy) < 1e-9:
            return (x1, y_plane)
        t = (y_plane - y1) / dy
        return (x1 + t * (x2 - x1), y_plane)

    clipped = _clip_plane(poly, lambda p: p[1] >= wy0, lambda a, b: _intersect_y(a, b, wy0))
    clipped = _clip_plane(clipped, lambda p: p[1] <= wy1, lambda a, b: _intersect_y(a, b, wy1))
    # Deduplicate near-identical consecutive vertices
    dedup = []
    for p in clipped:
        if not dedup or (abs(p[0] - dedup[-1][0]) > 1e-4 or abs(p[1] - dedup[-1][1]) > 1e-4):
            dedup.append(p)
    if len(dedup) >= 2 and (abs(dedup[0][0] - dedup[-1][0]) <= 1e-4 and abs(dedup[0][1] - dedup[-1][1]) <= 1e-4):
        dedup.pop()
    return dedup


def _is_valid_polygon_norm(pts, min_area=0.015):
    """Validates a normalized polygon in [0, 1]^2:
    - len(pts) >= 3
    - all coords strictly in [0.0, 1.0]
    - strictly non-self-intersecting
    - normalized shoelace area >= min_area (0.015)
    """
    n = len(pts)
    if n < 3:
        return False
    for (px, py) in pts:
        if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
            return False
    # Shoelace area
    area2 = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area2 += (x1 * y2 - x2 * y1)
    if 0.5 * abs(area2) < min_area:
        return False
    # Strict segment non-self-intersection check for non-adjacent edges
    def _ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])
    def _seg_intersect(A, B, C, D):
        c1 = _ccw(A, B, C); c2 = _ccw(A, B, D)
        c3 = _ccw(C, D, A); c4 = _ccw(C, D, B)
        return (c1 * c2 < -1e-12) and (c3 * c4 < -1e-12)
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1, b2 = pts[j], pts[(j + 1) % n]
            if _seg_intersect(a1, a2, b1, b2):
                return False
    return True


def _poly_norm_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area2 += (x1 * y2 - x2 * y1)
    return 0.5 * abs(area2)


def export_yolo_seg_dataset(chapter_dirs, out_dir, window_aspect=2.0, stride_overlap=0.50,
                            series_tag=None, return_tile_meta=False):
    """Phase 1/2 v2 Filtered YOLO-Seg Trapezoidal Polygon Auto-Exporter:
    - Uses exact `cy_top` / `cy_bot` centerlines + exact Sutherland-Hodgman half-plane clipping.
    - Hits `_DETECTOR_LRU_CACHE` 100% of the time (aligned `stride_overlap=0.20`).
    - Enforces all 5 Anti-Poisoning Guards (Section 2.2):
      1. Blank-polygon rejection (< 1.2% non-bg foreground).
      2. Dropped-credit overlap discard + unlabeled-art detection (>= 10% height, std >= 22, ink >= 0.15 -> review_hard).
      3. Degenerate 3-point corner-triangle rejection (norm area < 0.04).
      4. Marginal-slanted confidence routing (border_conf < 0.72 -> review_hard).
      5. Title-card / promo-text edge-tile exclusion (first 1 or last 2 frames matching _EXPORTER_ENDCARD_RE without story dialogue).
    """
    t_exp0 = time.time()
    out_root = Path(out_dir)
    for sub in ("gold_auto/images", "gold_auto/labels", "review_hard/images", "review_hard/labels", "review_hard/previews"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    gold_count = 0
    hard_count = 0
    total_polys = 0
    valid_polys = 0
    poly_4pt = 0
    poly_5pt = 0
    slanted_tiles = 0
    chapters_processed = 0
    blank_polys_skipped = 0
    degenerate_tri_skipped = 0
    dropped_credit_tiles_skipped = 0
    promo_edge_tiles_skipped = 0
    unlabeled_art_routed = 0
    marginal_slanted_routed = 0
    tile_records = []

    for ch_dir in chapter_dirs:
        ch_path = Path(ch_dir)
        pages = sorted([p for p in ch_path.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
        if not pages:
            continue
        s_name = series_tag or ch_path.parent.name
        ch_tag = f"{s_name}_{ch_path.name}"
        frames, strip, spans, _dt, dropped_y_ranges = run_slicer_b_on_chapter(
            pages, chapter_tag=ch_tag, return_dropped_ranges=True, include_ocr=True
        )
        if strip is None or not frames:
            continue
        chapters_processed += 1
        H, W = strip.shape[:2]
        # Patch 1.3: Guaranteed 100% LRU cache hit (aligned default stride_overlap=0.20!)
        faces, all_bubs, _, _, _, dial_bubs = detect_faces_and_bubbles_b(strip, stride_overlap=0.20, fast_bub=False)
        qa = run_qa_checks_on_chapter_b(strip, frames, faces, dial_bubs, all_bubs)
        frame_flags = qa["frame_flags"]

        # Guard 2.2.5: Pre-check first 1 and last 2 frames for promo/title-card text without story dialogue
        n_frames = len(frames)
        promo_edge_frame_idxs = set()
        edge_check_idxs = set()
        if n_frames >= 1:
            edge_check_idxs.add(0)
        if n_frames >= 2:
            edge_check_idxs.add(n_frames - 1)
        if n_frames >= 3:
            edge_check_idxs.add(n_frames - 2)
        for fi_e in edge_check_idxs:
            f_e = frames[fi_e]
            fy1_e, fy2_e = int(f_e["global_box"][1]), int(f_e["global_box"][3])
            has_dial_e = any(b[3] > fy1_e and b[1] < fy2_e for b in dial_bubs)
            if not has_dial_e:
                txt_e = f_e.get("ocr_text", "") or ""
                if not txt_e and (fy2_e - fy1_e) <= 1400:
                    txt_e = ocr_crop_rapidocr_b(strip[max(0, fy1_e):min(H, fy2_e), 0:W])
                    f_e["ocr_text"] = txt_e
                if txt_e and _EXPORTER_ENDCARD_RE.search(txt_e):
                    promo_edge_frame_idxs.add(fi_e)

        win_h = int(round(window_aspect * W))
        step = max(200, int(round(win_h * (1.0 - stride_overlap))))
        export_w = min(800, W)
        scale_e = export_w / float(W)

        wy0 = 0
        tile_idx = 0
        while wy0 < H:
            wy1 = min(H, wy0 + win_h)
            if wy1 - wy0 < int(0.5 * W):
                break
            wh = wy1 - wy0

            # Guard 2.2.2a: Discard completely any tile overlapping a dropped credit/end-card region
            overlaps_dropped = any((min(wy1, dy2) - max(wy0, dy1)) >= 20 for (dy1, dy2) in dropped_y_ranges)
            if overlaps_dropped:
                dropped_credit_tiles_skipped += 1
                tile_idx += 1
                if wy1 >= H:
                    break
                wy0 += step
                continue

            tile_rgb_full = strip[wy0:wy1, 0:W]
            t_gray_full = cv2.cvtColor(tile_rgb_full, cv2.COLOR_RGB2GRAY)
            # 4x downscaled tile gray for fast polygon blankness & unlabeled-art verification
            tg_sh, tg_sw = max(1, wh // 4), max(1, W // 4)
            t_gray_s = cv2.resize(t_gray_full, (tg_sw, tg_sh), interpolation=cv2.INTER_AREA)

            polys = []
            is_hard = False
            hard_reasons = set()
            has_slanted = False
            hits_promo_edge = False

            for f_i, f in enumerate(frames):
                fy1, fy2 = float(f["global_box"][1]), float(f["global_box"][3])
                ov = min(fy2, wy1) - max(fy1, wy0)
                if ov < max(60.0, 0.08 * wh):
                    continue
                if f_i in promo_edge_frame_idxs:
                    hits_promo_edge = True
                    break

                ang_t = float(f.get("angle_top", 0))
                ang_b = float(f.get("angle_bot", 0))
                cy_t = float(f.get("cy_top", fy1))
                cy_b = float(f.get("cy_bot", fy2))
                bc_t = float(f.get("border_conf_top", 1.0))
                bc_b = float(f.get("border_conf_bot", 1.0))
                if (ang_t != 0 and wy0 < cy_t < wy1) or (ang_b != 0 and wy0 < cy_b < wy1):
                    has_slanted = True

                # Patch 1.2: Exact global trapezoid from cy_top / cy_bot
                if ang_t != 0:
                    half_dy_t = 0.5 * float(W) * math.tan(math.radians(ang_t))
                    yt_l, yt_r = cy_t - half_dy_t, cy_t + half_dy_t
                else:
                    yt_l = yt_r = fy1
                if ang_b != 0:
                    half_dy_b = 0.5 * float(W) * math.tan(math.radians(ang_b))
                    yb_l, yb_r = cy_b - half_dy_b, cy_b + half_dy_b
                else:
                    yb_l = yb_r = fy2

                global_quad = [(0.0, yt_l), (float(W), yt_r), (float(W), yb_r), (0.0, yb_l)]
                clipped_px = _sutherland_hodgman_clip_y(global_quad, float(wy0), float(wy1))
                norm_poly = [
                    (max(0.0, min(1.0, px / float(W))), max(0.0, min(1.0, (py - float(wy0)) / float(wh))))
                    for (px, py) in clipped_px
                ]
                total_polys += 1
                if not _is_valid_polygon_norm(norm_poly, min_area=0.015):
                    continue

                # Guard 2.2.3: Degenerate 3-point corner-triangle rejection (norm area < 0.04)
                if len(norm_poly) == 3 and _poly_norm_area(norm_poly) < 0.04:
                    degenerate_tri_skipped += 1
                    continue

                # Guard 2.2.1: Blank-polygon rejection (< 1.2% non-background foreground)
                pmask = np.zeros((tg_sh, tg_sw), dtype=np.uint8)
                pts_s = np.array(
                    [[int(round(px * (tg_sw - 1))), int(round(py * (tg_sh - 1)))] for (px, py) in norm_poly],
                    dtype=np.int32
                )
                cv2.fillPoly(pmask, [pts_s], 1)
                pix_in = t_gray_s[pmask > 0]
                if pix_in.size > 0:
                    bg_med = int(np.median(pix_in))
                    fg_frac = float((np.abs(pix_in.astype(np.int16) - bg_med) > 18).mean())
                    if fg_frac < 0.012:
                        blank_polys_skipped += 1
                        continue

                valid_polys += 1
                polys.append(norm_poly)
                if len(norm_poly) == 4:
                    poly_4pt += 1
                elif len(norm_poly) >= 5:
                    poly_5pt += 1

                # Anti-error-distillation & Guard 2.2.4 routing checks
                if f.get("layout") == "vertical_pan":
                    is_hard = True; hard_reasons.add("vertical_pan")
                if (wy0 < fy1 < wy1 and f.get("cut_src_top") == "dp") or (wy0 < fy2 < wy1 and f.get("cut_src_bot") == "dp"):
                    is_hard = True; hard_reasons.add("dp_fallback")
                if (wy0 < fy1 < wy1 and f.get("cut_src_top") == "flat_band") or (wy0 < fy2 < wy1 and f.get("cut_src_bot") == "flat_band"):
                    is_hard = True; hard_reasons.add("flat_band")
                if (ang_t != 0 and wy0 < cy_t < wy1 and bc_t < 0.72) or (ang_b != 0 and wy0 < cy_b < wy1 and bc_b < 0.72):
                    is_hard = True; hard_reasons.add("marginal_slanted")
                if len(frame_flags[f_i]) > 0:
                    is_hard = True
                    for fl in frame_flags[f_i]:
                        hard_reasons.add(fl)

            # Guard 2.2.5: Skip title-card / promo-text edge tiles completely
            if hits_promo_edge:
                promo_edge_tiles_skipped += 1
                tile_idx += 1
                if wy1 >= H:
                    break
                wy0 += step
                continue

            if polys:
                # Guard 2.2.2b: Unlabeled-art check outside emitted polygons
                cov_mask = np.zeros((tg_sh, tg_sw), dtype=np.uint8)
                for q in polys:
                    pts_q = np.array(
                        [[int(round(px * (tg_sw - 1))), int(round(py * (tg_sh - 1)))] for (px, py) in q],
                        dtype=np.int32
                    )
                    cv2.fillPoly(cov_mask, [pts_q], 1)
                uncov_rows = (cov_mask.mean(axis=1) < 0.20).astype(np.int8)
                min_band_rows = max(4, int(0.10 * tg_sh))
                pad_u = np.pad(uncov_rows, (1, 1))
                for u_s, u_e in zip(np.where(np.diff(pad_u) == 1)[0], np.where(np.diff(pad_u) == -1)[0]):
                    if (u_e - u_s) >= min_band_rows:
                        band_g = t_gray_s[u_s:u_e, :]
                        b_std = float(band_g.astype(np.float32).std())
                        b_ink = float((band_g < 205).mean())
                        if b_std >= 22.0 and b_ink >= 0.15:
                            is_hard = True
                            hard_reasons.add("unlabeled_art")
                            break

                if "unlabeled_art" in hard_reasons:
                    unlabeled_art_routed += 1
                if "marginal_slanted" in hard_reasons:
                    marginal_slanted_routed += 1

                is_dark = bool(float(np.median(np.concatenate([t_gray_full[:, 0], t_gray_full[:, -1]]))) <= 45.0)
                export_h = max(1, int(round(wh * scale_e)))
                tile_rgb = cv2.resize(tile_rgb_full, (export_w, export_h), interpolation=cv2.INTER_AREA) if export_w != W else tile_rgb_full
                stem = f"{ch_tag}_t{tile_idx:04d}"
                label_lines = [
                    "0 " + " ".join(f"{px:.6f} {py:.6f}" for (px, py) in q)
                    for q in polys
                ]
                if has_slanted:
                    slanted_tiles += 1
                if not is_hard:
                    Image.fromarray(tile_rgb).save(out_root / "gold_auto/images" / f"{stem}.jpg", quality=88)
                    (out_root / "gold_auto/labels" / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    gold_count += 1
                    bucket = "gold_auto"
                else:
                    Image.fromarray(tile_rgb).save(out_root / "review_hard/images" / f"{stem}.jpg", quality=88)
                    (out_root / "review_hard/labels" / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    prev = Image.fromarray(tile_rgb.copy())
                    pdr = ImageDraw.Draw(prev)
                    for q in polys:
                        pts_px = [(int(px * (export_w - 1)), int(py * (export_h - 1))) for (px, py) in q]
                        pdr.polygon(pts_px, outline=(0, 240, 115), width=5)
                    prev.save(out_root / "review_hard/previews" / f"{stem}.jpg", quality=80)
                    hard_count += 1
                    bucket = "review_hard"

                tile_records.append({
                    "stem": stem,
                    "series": s_name,
                    "chapter": ch_path.name,
                    "bucket": bucket,
                    "n_polys": len(polys),
                    "has_slanted": has_slanted,
                    "is_dark": is_dark,
                    "hard_reasons": sorted(list(hard_reasons))
                })

            tile_idx += 1
            if wy1 >= H:
                break
            wy0 += step

    total_sec = time.time() - t_exp0
    out_stats = {
        "chapters_processed": chapters_processed,
        "gold_auto_tiles": gold_count,
        "review_hard_tiles": hard_count,
        "total_tiles": gold_count + hard_count,
        "total_polygons": total_polys,
        "valid_polygons": valid_polys,
        "poly_4pt": poly_4pt,
        "poly_5pt": poly_5pt,
        "slanted_tiles": slanted_tiles,
        "blank_polys_skipped": blank_polys_skipped,
        "degenerate_tri_skipped": degenerate_tri_skipped,
        "dropped_credit_tiles_skipped": dropped_credit_tiles_skipped,
        "promo_edge_tiles_skipped": promo_edge_tiles_skipped,
        "unlabeled_art_routed": unlabeled_art_routed,
        "marginal_slanted_routed": marginal_slanted_routed,
        "elapsed_sec": round(total_sec, 2),
        "out_dir": str(out_root)
    }
    if return_tile_meta:
        out_stats["tile_records"] = tile_records
    return out_stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Slicer_B Phase 1 CLI & YOLO-Seg Dataset Auto-Exporter")
    ap.add_argument("--export-yolo-seg", nargs="+", help="Chapter directories to export into YOLO-seg dataset")
    ap.add_argument("--out-dir", default="/tmp/yolo_seg_dataset", help="Output dataset directory")
    ap.add_argument("--window-aspect", type=float, default=2.0)
    ap.add_argument("--stride-overlap", type=float, default=0.50)
    args = ap.parse_args()
    if args.export_yolo_seg:
        stats = export_yolo_seg_dataset(
            args.export_yolo_seg, args.out_dir,
            window_aspect=args.window_aspect, stride_overlap=args.stride_overlap
        )
        print(json.dumps(stats, indent=2))
