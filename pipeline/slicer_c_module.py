#!/usr/bin/env python3
"""
Phase 4B — Slicer_C (RECAP_SLICER_ENGINE=v3):
Unified YOLO11n-seg ONNX Panel Segmentation Slicer (Pure onnxruntime + cv2 + numpy)
with Macro-Segment Pre-Gating, 2-Worker Parallel ORT Inference, Post-6b Orphan Reattachment,
Squeezed-Height Bubble-Only Parity, Post-6b 4-Pass Convergence Safety Guard, and 2,280px Non-Pan Split Cap.
- Leaves pipeline/slicer_b_module.py 100% untouched.
- Gracefully falls back to run_slicer_b_on_chapter if the ONNX model is missing or fails to load.
"""
import os
import sys
import math
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline import master_pipeline as mp
from pipeline import slicer_b_module as sb

logger = logging.getLogger("slicer_c_module")

_PANEL_SEG_MODEL_PATH = Path(
    os.environ.get(
        "RECAP_PANEL_SEG_ONNX",
        "/home/azureuser/manhwa-recap-studio-v3/pipeline/models/panel-seg/yolo11n_manhwa_panel_seg_v3_640.onnx",
    )
)
_panel_seg_sessions = None
_panel_seg_init_tried = False


def _get_panel_seg_sessions():
    """Cached 4-worker ORT session pool (4 x intra_op=2 -> 35.3 ms/tile effective on 8-vCPU VM)."""
    global _panel_seg_sessions, _panel_seg_init_tried
    if _panel_seg_init_tried:
        return _panel_seg_sessions
    _panel_seg_init_tried = True
    try:
        if _PANEL_SEG_MODEL_PATH.exists():
            import onnxruntime as ort
            n_workers = 4
            pool = []
            for _ in range(n_workers):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                pool.append(ort.InferenceSession(str(_PANEL_SEG_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]))
            _panel_seg_sessions = pool
            logger.info("Loaded YOLO11n-seg v3 ONNX (%dx intra_op=2) from %s", n_workers, _PANEL_SEG_MODEL_PATH.name)
    except Exception as e:
        logger.warning("Failed to load YOLO11n-seg v3 ONNX (%s) — will fall back to Slicer_B", e)
        _panel_seg_sessions = None
    return _panel_seg_sessions


def _get_panel_seg_session():
    """Backwards-compatible accessor returning the first session in the pool."""
    pool = _get_panel_seg_sessions()
    return pool[0] if pool else None


def _letterbox_640(img_rgb, imgsz=640, pad_val=114):
    h, w = img_rgb.shape[:2]
    scale = min(float(imgsz) / max(1, w), float(imgsz) / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    pad_x = (imgsz - nw) // 2
    pad_y = (imgsz - nh) // 2
    canvas = np.full((imgsz, imgsz, 3), pad_val, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y, nw, nh


def _nms_xyxy(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def _infer_tile_yolo11_seg(sess, tile_rgb, conf_thresh=0.65, iou_thresh=0.45, imgsz=640):
    th, tw = tile_rgb.shape[:2]
    if th < 64 or tw < 64:
        return []
    canvas, scale, pad_x, pad_y, nw, nh = _letterbox_640(tile_rgb, imgsz=imgsz, pad_val=114)
    x_in = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    inp_name = sess.get_inputs()[0].name
    out0, out1 = sess.run(None, {inp_name: x_in})

    preds = out0[0].T
    scores = preds[:, 4]
    mask_keep = scores >= conf_thresh
    if not mask_keep.any():
        return []
    preds = preds[mask_keep]
    scores = scores[mask_keep]

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes_640 = np.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], axis=1)
    keep_idx = _nms_xyxy(boxes_640, scores, iou_thresh=iou_thresh)
    if not keep_idx:
        return []

    preds = preds[keep_idx]
    scores = scores[keep_idx]
    boxes_640 = boxes_640[keep_idx]
    coeffs = preds[:, 5:]

    protos = out1[0]
    mh, mw = protos.shape[1], protos.shape[2]
    logits = np.clip(coeffs @ protos.reshape(32, -1), -20.0, 20.0)
    masks_160 = (1.0 / (1.0 + np.exp(-logits))).reshape(-1, mh, mw)

    m_scale = float(mw) / float(imgsz)
    mx0 = max(0, int(round(pad_x * m_scale)))
    my0 = max(0, int(round(pad_y * m_scale)))
    mx1 = min(mw, max(mx0 + 1, int(round((pad_x + nw) * m_scale))))
    my1 = min(mh, max(my0 + 1, int(round((pad_y + nh) * m_scale))))

    q_w = max(32, tw // 4)
    q_h = max(32, th // 4)
    sx_full = float(tw) / float(q_w)
    sy_full = float(th) / float(q_h)

    dets = []
    for k in range(len(keep_idx)):
        bx1_640, by1_640, bx2_640, by2_640 = boxes_640[k]
        m = masks_160[k]
        cbx1 = max(0, int(math.floor(bx1_640 * m_scale)))
        cby1 = max(0, int(math.floor(by1_640 * m_scale)))
        cbx2 = min(mw, int(math.ceil(bx2_640 * m_scale)))
        cby2 = min(mh, int(math.ceil(by2_640 * m_scale)))
        box_crop = np.zeros_like(m)
        if cbx2 > cbx1 and cby2 > cby1:
            box_crop[cby1:cby2, cbx1:cbx2] = m[cby1:cby2, cbx1:cbx2]
        unpad = box_crop[my0:my1, mx0:mx1]
        if unpad.size == 0:
            continue
        m_q = cv2.resize(unpad, (q_w, q_h), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_q > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 0.012 * q_w * q_h:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.015 * peri, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 3:
            continue
        pts = np.stack([approx[:, 0] * sx_full, approx[:, 1] * sy_full], axis=1)
        pts[:, 0] = np.where(pts[:, 0] <= 0.03 * tw, 0.0, np.where(pts[:, 0] >= 0.97 * tw, float(tw), pts[:, 0]))
        pts[:, 1] = np.where(pts[:, 1] <= 0.015 * th, 0.0, np.where(pts[:, 1] >= 0.985 * th, float(th), pts[:, 1]))
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(tw))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(th))

        x1_t, y1_t = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2_t, y2_t = float(pts[:, 0].max()), float(pts[:, 1].max())
        if (x2_t - x1_t) < 0.12 * tw or (y2_t - y1_t) < 48.0:
            continue

        dets.append({
            "pts": pts,
            "bbox": [x1_t, y1_t, x2_t, y2_t],
            "conf": float(scores[k]),
            "touches_top": bool(y1_t <= 0.08 * th),
            "touches_bot": bool(y2_t >= 0.92 * th),
        })
    return dets


def _extract_poly_angles(pts, W, fy1, fy2):
    ang_t, ang_b = 0, 0
    cy_t, cy_b = int(round(fy1)), int(round(fy2))
    if pts is None or len(pts) < 4:
        return ang_t, ang_b, cy_t, cy_b
    mid_y = 0.5 * (fy1 + fy2)
    n = len(pts)
    best_top = None
    best_bot = None
    for i in range(n):
        x1, y1 = float(pts[i, 0]), float(pts[i, 1])
        x2, y2 = float(pts[(i + 1) % n, 0]), float(pts[(i + 1) % n, 1])
        dx = x2 - x1
        if abs(dx) < 0.70 * W:
            continue
        if dx < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
            dx = -dx
        dy = y2 - y1
        ey_mid = 0.5 * (y1 + y2)
        deg = float(np.rad2deg(np.arctan2(dy, dx)))
        if abs(deg) > 28.0:
            continue
        if ey_mid < mid_y and (best_top is None or dx > best_top[0]):
            best_top = (dx, deg, ey_mid)
        elif ey_mid >= mid_y and (best_bot is None or dx > best_bot[0]):
            best_bot = (dx, deg, ey_mid)

    if best_top is not None and abs(best_top[1]) >= 2.0:
        ang_t = int(round(best_top[1]))
        cy_t = int(round(best_top[2]))
    if best_bot is not None and abs(best_bot[1]) >= 2.0:
        ang_b = int(round(best_bot[1]))
        cy_b = int(round(best_bot[2]))
    return ang_t, ang_b, cy_t, cy_b


def _stitch_detections(all_dets, H, W):
    """Cross-window polygon stitching & deduplication across overlapping windows inside a macro-segment."""
    all_dets.sort(key=lambda d: (d["wy0"], d["bbox"][1]))
    tracks = []

    for det in all_dets:
        dx1, dy1, dx2, dy2 = det["bbox"]
        d_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
        d_interior = (not det["touches_top"]) and (not det["touches_bot"])

        best_idx = None
        best_mode = None
        best_score = 0.0

        for idx, tr in enumerate(tracks):
            tx1, ty1, tx2, ty2 = tr["bbox"]
            if ty2 < det["wy0"] - 48.0:
                continue
            x_ov = max(0.0, min(tx2, dx2) - max(tx1, dx1))
            min_w = max(1.0, min(tx2 - tx1, dx2 - dx1))
            x_ratio = x_ov / min_w
            if x_ratio < 0.50:
                continue
            y_ov = max(0.0, min(ty2, dy2) - max(ty1, dy1))
            t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
            inter = x_ov * y_ov
            iou = inter / max(1.0, t_area + d_area - inter)
            ioa = inter / min(t_area, d_area)

            if tr["touches_bot"] and (det["touches_top"] or y_ov >= 160.0) and x_ratio >= 0.60 and dy2 > ty2 + 40.0:
                if y_ov >= 24.0 or abs(dy1 - ty2) <= 120.0:
                    sc_c = 0.8 + x_ratio
                    if sc_c > best_score:
                        best_score = sc_c
                        best_idx = idx
                        best_mode = "chain"
            elif iou >= 0.40 or ioa >= 0.55:
                if iou + 0.5 * ioa > best_score:
                    best_score = iou + 0.5 * ioa
                    best_idx = idx
                    best_mode = "dedup"

        if best_idx is None:
            tracks.append({
                "pts": det["pts"].copy(),
                "bbox": [dx1, dy1, dx2, dy2],
                "conf": det["conf"],
                "touches_top": det["touches_top"],
                "touches_bot": det["touches_bot"],
                "interior": d_interior,
                "windows": 1,
                "last_wy1": det["wy1"],
            })
        else:
            tr = tracks[best_idx]
            if best_mode == "dedup":
                if d_interior and not tr["interior"]:
                    tr["pts"] = det["pts"].copy()
                    tr["bbox"] = [dx1, dy1, dx2, dy2]
                    tr["conf"] = max(tr["conf"], det["conf"])
                    tr["touches_top"] = False
                    tr["touches_bot"] = False
                    tr["interior"] = True
                elif tr["interior"] and not d_interior:
                    tr["conf"] = max(tr["conf"], det["conf"])
                elif tr["touches_bot"] and dy2 > tr["bbox"][3] + 12.0:
                    merged_pts = np.vstack([tr["pts"], det["pts"]])
                    hull = cv2.convexHull(merged_pts.astype(np.float32)).reshape(-1, 2)
                    tr["pts"] = hull
                    tr["bbox"] = [float(hull[:, 0].min()), float(hull[:, 1].min()), float(hull[:, 0].max()), float(hull[:, 1].max())]
                    tr["touches_bot"] = det["touches_bot"]
                    tr["interior"] = (not tr["touches_top"]) and (not tr["touches_bot"])
                    tr["conf"] = max(tr["conf"], det["conf"])
                    tr["windows"] += 1
                elif det["conf"] > tr["conf"]:
                    tr["pts"] = det["pts"].copy()
                    tr["bbox"] = [dx1, dy1, dx2, dy2]
                    tr["conf"] = det["conf"]
                    tr["touches_bot"] = tr["touches_bot"] and det["touches_bot"]
                tr["last_wy1"] = max(tr["last_wy1"], det["wy1"])
            else:
                merged_pts = np.vstack([tr["pts"], det["pts"]])
                hull = cv2.convexHull(merged_pts.astype(np.float32)).reshape(-1, 2)
                tr["pts"] = hull
                tr["bbox"] = [float(hull[:, 0].min()), float(hull[:, 1].min()), float(hull[:, 0].max()), float(hull[:, 1].max())]
                tr["touches_bot"] = det["touches_bot"]
                tr["interior"] = (not tr["touches_top"]) and (not tr["touches_bot"])
                tr["conf"] = max(tr["conf"], det["conf"])
                tr["windows"] += 1
                tr["last_wy1"] = max(tr["last_wy1"], det["wy1"])

    tracks.sort(key=lambda t: (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]), reverse=True)
    kept = []
    for tr in tracks:
        tx1, ty1, tx2, ty2 = tr["bbox"]
        ta = max(1.0, (tx2 - tx1) * (ty2 - ty1))
        if any(
            (max(0.0, min(tx2, k["bbox"][2]) - max(tx1, k["bbox"][0])) *
             max(0.0, min(ty2, k["bbox"][3]) - max(ty1, k["bbox"][1]))) / ta >= 0.65
            for k in kept
        ):
            continue
        kept.append(tr)

    kept.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for i in range(len(kept) - 1):
        a, b = kept[i]["bbox"], kept[i + 1]["bbox"]
        y_ov = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        min_h = max(1.0, min(a[3] - a[1], b[3] - b[1]))
        x_ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        if (y_ov / min_h) >= 0.50 and x_ov <= 0.15 * min(a[2] - a[0], b[2] - b[0]):
            if a[0] > b[0]:
                kept[i], kept[i + 1] = kept[i + 1], kept[i]
    return kept


def _yolo_vetoes_vertical_pan(sessions, strip, sa, sb_y, W, H):
    """Phase 4D Fix 1: Veto false-positive vertical_pan candidates when YOLO11n-seg
    confidently detects >=2 separable panels inside [sa, sb_y].
    Strict True-Pan Safety Rule: never veto if any single stitched panel spans
    >= 3,000 px or >= 65% of (sb_y - sa).
    """
    seg_h = sb_y - sa
    if not sessions or seg_h < 3000:
        return False
    win_h = int(round(2.0 * W))
    stride = max(200, int(round(0.55 * win_h)))
    sess0 = sessions[0]
    dets_pan = []
    wy0 = sa
    while wy0 < sb_y:
        wy1 = min(sb_y, wy0 + win_h)
        if wy1 - wy0 < int(0.40 * W) and dets_pan:
            break
        tile = strip[wy0:wy1, 0:W]
        if tile[::8, ::8].std() >= 4.0:
            dets = _infer_tile_yolo11_seg(sess0, tile, conf_thresh=0.72, iou_thresh=0.45, imgsz=640)
            for d in dets:
                pts_g = d["pts"].copy()
                pts_g[:, 1] += float(wy0)
                bx1, by1, bx2, by2 = d["bbox"]
                dets_pan.append({
                    "pts": pts_g,
                    "bbox": [bx1, by1 + float(wy0), bx2, by2 + float(wy0)],
                    "conf": d["conf"],
                    "touches_top": d["touches_top"] and (wy0 > sa),
                    "touches_bot": d["touches_bot"] and (wy1 < sb_y),
                    "wy0": wy0,
                    "wy1": wy1,
                })
        if wy1 >= sb_y:
            break
        wy0 += stride

    if not dets_pan:
        return False
    stitched = _stitch_detections(dets_pan, H, W)
    if not stitched:
        return False

    # Strict True-Pan Safety Rule: any unbroken panel >= 3,000 px or >= 65% of segment -> keep pan!
    for tr in stitched:
        ph = float(tr["bbox"][3] - tr["bbox"][1])
        if ph >= 3000.0 or ph >= 0.65 * float(seg_h):
            return False

    # Check for >= 2 distinct high-confidence separable panels (360..2400 px, conf >= 0.75)
    cands = [
        tr for tr in stitched
        if float(tr["conf"]) >= 0.75 and 360.0 <= float(tr["bbox"][3] - tr["bbox"][1]) <= 2400.0
    ]
    if len(cands) < 2:
        return False
    cands.sort(key=lambda t: float(t["bbox"][1]))
    for i in range(len(cands)):
        ay1, ay2 = float(cands[i]["bbox"][1]), float(cands[i]["bbox"][3])
        ha = max(1.0, ay2 - ay1)
        for j in range(i + 1, len(cands)):
            by1, by2 = float(cands[j]["bbox"][1]), float(cands[j]["bbox"][3])
            hb = max(1.0, by2 - by1)
            gap = by1 - ay2
            if gap >= 20.0:
                return True
            y_ov = max(0.0, min(ay2, by2) - max(ay1, by1))
            if y_ov < 0.20 * min(ha, hb):
                return True
    return False


def run_slicer_c_on_chapter(page_paths, chapter_tag="chap_test", known_ocr_map=None,
                            precomputed_faces=None, face_dt=0.0, force_split_multi=False,
                            include_crops=False, return_crops=False,
                            return_dropped_ranges=False, include_ocr=False,
                            conf_thresh=0.65):
    """Phase 4B Slicer_C (v3): Macro-segment pure_gut pre-gating + 2-worker parallel YOLO11n-seg ONNX
    + post-6b orphan reattachment + squeezed-height bubble-only parity + 4-pass convergence safety guard.
    """
    sessions = _get_panel_seg_sessions()
    if not sessions:
        return sb.run_slicer_b_on_chapter(
            page_paths, chapter_tag=chapter_tag, known_ocr_map=known_ocr_map,
            precomputed_faces=precomputed_faces, face_dt=face_dt, force_split_multi=force_split_multi,
            include_crops=include_crops, return_crops=return_crops,
            return_dropped_ranges=return_dropped_ranges, include_ocr=include_ocr,
        )

    keep_crops = bool(include_crops or return_crops)
    t0 = time.time()

    # Stage 1: Full-chapter strip reconstruction (shared with Slicer_B)
    strip, spans = sb.reconstruct_strip_b(page_paths)
    if strip is None:
        return ([], None, [], 0.0, []) if return_dropped_ranges else ([], None, [], 0.0)
    H, W = strip.shape[:2]
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)

    # Stage 2: Shared face & speech-bubble detection
    faces, bubbles, face_mask, bubble_mask, hard_mask, dialogue_boxes_raw = sb.detect_faces_and_bubbles_b(
        strip, stride_overlap=0.20, fast_bub=False, precomputed_faces=precomputed_faces
    )

    # Fix 1.1: Macro-segment pure_gut pre-segmentation (pure white | black ONLY, NEVER flat_band!)
    MIN_GUT = 24
    MIN_PANEL = 180
    BUBBLE_BRIDGE = 560
    SPLIT_CAP = min(2280, int(2.85 * W))

    gs = gray[:, ::2]
    std_r = gs.astype(np.float32).std(axis=1)
    ink = (gs < 205).mean(axis=1)
    white_r = ((gs >= 240).mean(axis=1) >= 0.94) & (std_r <= 16.0)
    black_r = ((gs <= 20).mean(axis=1) >= 0.94) & (std_r <= 14.0)
    pure_gut = (white_r | black_r) & (~hard_mask)
    g_fill_all = np.maximum((gs >= 238).mean(axis=1), (gs <= 20).mean(axis=1))

    seam_ys = [int(y0) for (_, y0, _) in spans[1:] if 0 < y0 < H]
    nongut_seam_ys = [sy for sy in seam_ys if not sb._is_in_pure_gutter_band_b(pure_gut, sy, min_band=20)]

    raw_bands, i = [], 0
    while i < H:
        if not pure_gut[i]:
            i += 1
            continue
        j = i
        while j < H and pure_gut[j]:
            j += 1
        raw_bands.append([i, j])
        i = j

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

    pure_cuts = []
    for (a, b) in bands:
        if int(pure_gut[a:b].sum()) < MIN_GUT:
            continue
        pad_v = np.pad((~hard_mask[a:b]).astype(np.int8), (1, 1))
        runs = [(int(s + a), int(e + a)) for s, e in zip(np.where(np.diff(pad_v) == 1)[0], np.where(np.diff(pad_v) == -1)[0])]
        if runs:
            best_run = max(runs, key=lambda r: r[1] - r[0])
            if best_run[1] - best_run[0] >= 12:
                pure_cuts.append((best_run[0] + best_run[1]) // 2)

    bounds = [0] + sorted(set(pure_cuts)) + [H]
    macro_segs = []
    for k in range(len(bounds) - 1):
        sa, sb_y = bounds[k], bounds[k + 1]
        if sb_y - sa >= 80:
            macro_segs.append([sa, sb_y])
        elif macro_segs:
            macro_segs[-1][1] = sb_y
    if not macro_segs:
        macro_segs = [[0, H]]

    row_cost_base = None
    vert_diff_s = None
    horiz_line_frac = None

    def _ensure_row_cost():
        nonlocal row_cost_base, vert_diff_s, horiz_line_frac
        if row_cost_base is None:
            row_cost_base, vert_diff_s, horiz_line_frac = sb.compute_composite_row_cost_b(
                gray, hard_mask, faces=faces, seam_ys=seam_ys, pure_gut=pure_gut
            )

    def _dp_split_segment(y1_i, y2_i, ang_t=0, ang_b=0, cy_t=None, cy_b=None, gv_t=None, gv_b=None,
                          src_t="yolo", src_b="yolo", conf=0.8):
        _ensure_row_cost()
        span_h = y2_i - y1_i
        sub_crop = strip[y1_i:y2_i]
        f_sub = [(f[0], f[1] - y1_i, f[2], f[3] - y1_i, f[4] if len(f) > 4 else 0.8)
                 for f in faces if f[1] >= y1_i and f[3] <= y2_i]
        rc_loc = row_cost_base[y1_i:y2_i].copy()
        if len(f_sub) > 0 or span_h > 2200:
            hb_loc = sb.detect_halfbodies_on_segment_b(sub_crop, conf=0.35)
            if hb_loc:
                is_brk_s = (g_fill_all[y1_i:y2_i] >= 0.85) | (horiz_line_frac[y1_i:y2_i] >= 0.55)
                hb_pen = np.zeros(span_h, dtype=np.float32)
                for hb in hb_loc:
                    by1_h, by2_h = max(0, int(hb[1])), min(span_h, int(hb[3]))
                    if by2_h > by1_h:
                        hb_pen[by1_h:by2_h] = 7.5
                hb_pen[is_brk_s] = 0.0
                rc_loc += hb_pen
        dp_parts = sb.solve_dp_cuts_b(
            rc_loc, hard_mask[y1_i:y2_i], 0, span_h,
            target_h=max(850, int(1.15 * W)), min_h=max(360, int(0.45 * W)), max_h=1950
        )
        out = []
        for pi_d, (pa_l, pb_l) in enumerate(dp_parts):
            pa, pb = y1_i + pa_l, y1_i + pb_l
            out.append({
                "x1": 0, "y1": pa, "x2": W, "y2": pb, "layout": "panel",
                "ang_t": ang_t if pi_d == 0 else 0,
                "ang_b": ang_b if pi_d == len(dp_parts) - 1 else 0,
                "cy_t": (cy_t if cy_t is not None else pa) if pi_d == 0 else pa,
                "cy_b": (cy_b if cy_b is not None else pb) if pi_d == len(dp_parts) - 1 else pb,
                "gv_t": gv_t if pi_d == 0 else None,
                "gv_b": gv_b if pi_d == len(dp_parts) - 1 else None,
                "src_t": src_t if pi_d == 0 else "dp",
                "src_b": src_b if pi_d == len(dp_parts) - 1 else "dp",
                "conf": conf,
            })
        return out

    # Fix 1.2: Route each macro-segment before calling YOLO
    direct_panels = []       # (seg_order, panel_dict)
    yolo_seg_jobs = []       # list of {"seg_id", "sa", "sb", "tiles": [(wy0, wy1, tile_rgb), ...]}
    win_h = int(round(2.0 * W))
    stride = max(200, int(round(0.55 * win_h)))
    compact_cap_multi = int(round(1.15 * W))
    compact_cap_single = int(round(1.30 * W))
    medium_cap = int(round(2.05 * W))

    def _route_nonpan_seg(sa, sb_y):
        seg_h = sb_y - sa
        if seg_h < 48:
            return
        n_faces_seg = sum(1 for f in faces if f[3] > sa and f[1] < sb_y)
        eff_compact = compact_cap_single if n_faces_seg <= 1 else compact_cap_multi
        if seg_h <= eff_compact:
            ty1, ty2 = sb.trim_content_span_b(strip, sa, sb_y, hard_mask, keep_margin=16)
            if ty2 - ty1 >= 48:
                direct_panels.append({
                    "x1": 0, "y1": ty1, "x2": W, "y2": ty2, "layout": "panel",
                    "ang_t": 0, "ang_b": 0, "cy_t": ty1, "cy_b": ty2,
                    "gv_t": None, "gv_b": None, "src_t": "gutter", "src_b": "gutter", "conf": 0.99
                })
        elif seg_h <= medium_cap:
            tile = strip[sa:sb_y, 0:W]
            if tile[::8, ::8].std() >= 4.0:
                yolo_seg_jobs.append({"sa": sa, "sb": sb_y, "tiles": [(sa, sb_y, tile)]})
        else:
            tiles_s = []
            wy0 = sa
            while wy0 < sb_y:
                wy1 = min(sb_y, wy0 + win_h)
                if wy1 - wy0 < int(0.40 * W) and tiles_s:
                    break
                tile = strip[wy0:wy1, 0:W]
                if tile[::8, ::8].std() >= 4.0:
                    tiles_s.append((wy0, wy1, tile))
                if wy1 >= sb_y:
                    break
                wy0 += stride
            if tiles_s:
                yolo_seg_jobs.append({"sa": sa, "sb": sb_y, "tiles": tiles_s})

    # Phase 4C Fix 1: Localized Canny + tightened quiet-band thresholds (rs <= 8.0, ed <= 0.006, min_r=28)
    # on seg_h > 3,800px so chap_119's 5,201px dark pan stays intact (6/6 pans) and full-strip Canny is avoided!
    refined_macro_segs = []
    for sa, sb_y in macro_segs:
        seg_h = sb_y - sa
        if seg_h <= 3800:
            refined_macro_segs.append([sa, sb_y])
            continue
        g_loc = gray[sa:sb_y]
        rs_loc = std_r[sa:sb_y]
        ed_loc = (cv2.Canny(g_loc, 50, 150) > 0).mean(axis=1)
        hm_loc = hard_mask[sa:sb_y]
        iw_mask = ((g_loc >= 240).mean(axis=1) >= 0.92) | ((rs_loc <= 8.0) & (ed_loc <= 0.006) & (~hm_loc))
        pad_iw = np.pad(iw_mask.astype(np.int8), (1, 1))
        cand_ys = []
        for s_r, e_r in zip(np.where(np.diff(pad_iw) == 1)[0], np.where(np.diff(pad_iw) == -1)[0]):
            cy_r = (s_r + e_r) // 2
            w_row_frac = (g_loc[s_r:e_r] >= 240).mean(axis=1)
            w_ratio_r = float(w_row_frac.mean())
            pure_w_rows = np.where(w_row_frac >= 0.98)[0]
            min_r = 8 if w_ratio_r >= 0.90 else 28
            if (e_r - s_r) >= min_r and 450 <= cy_r <= seg_h - 450:
                solid_white_override = (
                    len(pure_w_rows) >= 40
                    and not face_mask[sa + s_r + int(pure_w_rows[0]) : sa + s_r + int(pure_w_rows[-1]) + 1].any()
                )
                if solid_white_override or not hm_loc[max(0, s_r - 2):min(seg_h, e_r + 2)].all():
                    valid_r = np.where(~hm_loc[s_r:e_r])[0] + s_r
                    if len(valid_r) > 0:
                        cut_y_r = int(valid_r[len(valid_r) // 2])
                    elif solid_white_override:
                        cut_y_r = s_r + int(pure_w_rows[len(pure_w_rows) // 2])
                    else:
                        cut_y_r = cy_r
                    cand_ys.append(sa + cut_y_r)
        if seg_h > 3800:
            f_loc_m = [(f[0], f[1] - sa, f[2], f[3] - sa) for f in faces if f[1] >= sa and f[3] <= sb_y]
            d_loc_m = [(b[0], b[1] - sa, b[2], b[3] - sa) for b in dialogue_boxes_raw if b[1] >= sa and b[3] <= sb_y]
            for c0 in sb.find_slanted_gutters_b(strip[sa:sb_y], hard_mask=hm_loc, faces_2d=f_loc_m, bubbles_2d=d_loc_m, angles=(0,), min_fill=0.72, min_thick_px=24):
                if 450 <= c0[0] <= seg_h - 450:
                    cand_ys.append(sa + int(c0[0]))
        sub_pts = [sa]
        for gy in sorted(cand_ys):
            if gy - sub_pts[-1] >= 420 and (sb_y - gy) >= 420:
                sub_pts.append(gy)
        sub_pts.append(sb_y)
        for k_s in range(len(sub_pts) - 1):
            refined_macro_segs.append([sub_pts[k_s], sub_pts[k_s + 1]])

    for sa, sb_y in refined_macro_segs:
        seg_h = sb_y - sa
        if (not force_split_multi) and (3800 <= seg_h <= 6200):
            _ensure_row_cost()
            sub_crop = strip[sa:sb_y]
            f_sub = [(f[0], f[1] - sa, f[2], f[3] - sa, f[4] if len(f) > 4 else 0.8)
                     for f in faces if f[1] >= sa and f[3] <= sb_y]
            d_sub = [(b[0], b[1] - sa, b[2], b[3] - sa)
                     for b in dialogue_boxes_raw if b[1] >= sa and b[3] <= sb_y]
            sl_sub = [c for c in sb.find_slanted_gutters_b(
                          sub_crop, hard_mask=hard_mask[sa:sb_y],
                          faces_2d=f_sub, bubbles_2d=d_sub, min_fill=0.72, min_thick_px=24)
                      if 450 <= c[0] <= seg_h - 450]
            u2_t = sb.compute_u2netp_saliency_b(sub_crop)
            seams_l = [sy - sa for sy in nongut_seam_ys if sa < sy < sb_y]
            if sb.is_intentional_vertical_pan_b(
                sub_crop, f_sub, d_sub, sl_sub,
                row_cost_base[sa:sb_y], vert_diff_s[sa:sb_y], horiz_line_frac[sa:sb_y],
                u2netp_sal=u2_t, seam_ys_local=seams_l
            ) and not _yolo_vetoes_vertical_pan(sessions, strip, sa, sb_y, W, H):
                ty1, ty2 = sb.trim_content_span_b(strip, sa, sb_y, hard_mask, keep_margin=16)
                direct_panels.append({
                    "x1": 0, "y1": ty1, "x2": W, "y2": ty2, "layout": "vertical_pan",
                    "ang_t": 0, "ang_b": 0, "cy_t": ty1, "cy_b": ty2,
                    "gv_t": None, "gv_b": None, "src_t": "pan", "src_b": "pan", "conf": 1.0
                })
                continue
        elif (not force_split_multi) and (6200 < seg_h <= 9200):
            _ensure_row_cost()
            f_sub = [(f[0], f[1] - sa, f[2], f[3] - sa, f[4] if len(f) > 4 else 0.8)
                     for f in faces if f[1] >= sa and f[3] <= sb_y]
            if len(f_sub) <= 1:
                rc_win = row_cost_base[sa + 4400 : min(sb_y - 600, sa + 5800)]
                hm_win = hard_mask[sa + 4400 : min(sb_y - 600, sa + 5800)]
                valid_v = np.where(~hm_win)[0]
                if len(valid_v) > 0:
                    best_off = int(valid_v[np.argmin(rc_win[valid_v])])
                    if float(rc_win[best_off]) <= 3.70:
                        split_y = sa + 4400 + best_off
                        f_p1 = [(f[0], f[1] - sa, f[2], f[3] - sa, f[4] if len(f) > 4 else 0.8)
                                for f in faces if f[1] >= sa and f[3] <= split_y]
                        d_p1 = [(b[0], b[1] - sa, b[2], b[3] - sa)
                                for b in dialogue_boxes_raw if b[1] >= sa and b[3] <= split_y]
                        u2_p1 = sb.compute_u2netp_saliency_b(strip[sa:split_y])
                        seams_p1 = [sy - sa for sy in nongut_seam_ys if sa < sy < split_y]
                        if sb.is_intentional_vertical_pan_b(
                            strip[sa:split_y], f_p1, d_p1, [],
                            row_cost_base[sa:split_y], vert_diff_s[sa:split_y], horiz_line_frac[sa:split_y],
                            u2netp_sal=u2_p1, seam_ys_local=seams_p1
                        ) and not _yolo_vetoes_vertical_pan(sessions, strip, sa, split_y, W, H):
                            ty1, _ = sb.trim_content_span_b(strip, sa, split_y, hard_mask, keep_margin=16)
                            direct_panels.append({
                                "x1": 0, "y1": ty1, "x2": W, "y2": split_y, "layout": "vertical_pan",
                                "ang_t": 0, "ang_b": 0, "cy_t": ty1, "cy_b": split_y,
                                "gv_t": None, "gv_b": None, "src_t": "pan", "src_b": "dp", "conf": 1.0
                            })
                            _route_nonpan_seg(split_y, sb_y)
                            continue
        _route_nonpan_seg(sa, sb_y)

    # Phase 4C: 4-worker parallel ORT inference (4 x intra_op=2 -> 35.3ms/tile effective)
    dets_by_job = [[] for _ in range(len(yolo_seg_jobs))]
    flat_tasks = []
    for j_idx, job in enumerate(yolo_seg_jobs):
        sa_j, sb_j = job["sa"], job["sb"]
        for (wy0, wy1, tile) in job["tiles"]:
            flat_tasks.append((j_idx, sa_j, sb_j, wy0, wy1, tile))

    if flat_tasks:
        n_w = min(len(sessions), len(flat_tasks))
        if n_w <= 1:
            for (j_idx, sa_j, sb_j, wy0, wy1, tile) in flat_tasks:
                dets = _infer_tile_yolo11_seg(sessions[0], tile, conf_thresh=conf_thresh, iou_thresh=0.45, imgsz=640)
                for d in dets:
                    pts_g = d["pts"].copy()
                    pts_g[:, 1] += float(wy0)
                    bx1, by1, bx2, by2 = d["bbox"]
                    dets_by_job[j_idx].append({
                        "pts": pts_g,
                        "bbox": [bx1, by1 + float(wy0), bx2, by2 + float(wy0)],
                        "conf": d["conf"],
                        "touches_top": d["touches_top"] and (wy0 > sa_j),
                        "touches_bot": d["touches_bot"] and (wy1 < sb_j),
                        "wy0": wy0,
                        "wy1": wy1,
                    })
        else:
            from concurrent.futures import ThreadPoolExecutor
            def _run_worker(args):
                w_id, chunk = args
                s_w = sessions[w_id]
                out_w = []
                for (j_idx, sa_j, sb_j, wy0, wy1, tile) in chunk:
                    dets = _infer_tile_yolo11_seg(s_w, tile, conf_thresh=conf_thresh, iou_thresh=0.45, imgsz=640)
                    for d in dets:
                        pts_g = d["pts"].copy()
                        pts_g[:, 1] += float(wy0)
                        bx1, by1, bx2, by2 = d["bbox"]
                        out_w.append((j_idx, {
                            "pts": pts_g,
                            "bbox": [bx1, by1 + float(wy0), bx2, by2 + float(wy0)],
                            "conf": d["conf"],
                            "touches_top": d["touches_top"] and (wy0 > sa_j),
                            "touches_bot": d["touches_bot"] and (wy1 < sb_j),
                            "wy0": wy0,
                            "wy1": wy1,
                        }))
                return out_w
            chunks = [flat_tasks[i::n_w] for i in range(n_w)]
            with ThreadPoolExecutor(max_workers=n_w) as ex:
                for res_list in ex.map(_run_worker, [(i, chunks[i]) for i in range(n_w)]):
                    for j_idx, item in res_list:
                        dets_by_job[j_idx].append(item)

    # Assemble stitched panels from all YOLO macro-segments + direct panels
    panels = list(direct_panels)
    for j_idx, job in enumerate(yolo_seg_jobs):
        j_dets = dets_by_job[j_idx]
        if not j_dets:
            # Fallback if YOLO returned 0 detections inside this macro-segment
            sa_j, sb_j = job["sa"], job["sb"]
            ty1, ty2 = sb.trim_content_span_b(strip, sa_j, sb_j, hard_mask, keep_margin=16)
            if ty2 - ty1 >= 180:
                if (ty2 - ty1) > SPLIT_CAP:
                    panels.extend(_dp_split_segment(ty1, ty2, src_t="gap_dp", src_b="gap_dp", conf=0.5))
                else:
                    panels.append({
                        "x1": 0, "y1": ty1, "x2": W, "y2": ty2, "layout": "panel",
                        "ang_t": 0, "ang_b": 0, "cy_t": ty1, "cy_b": ty2,
                        "gv_t": None, "gv_b": None, "src_t": "gap_recover", "src_b": "gap_recover", "conf": 0.5
                    })
            continue

        stitched_j = _stitch_detections(j_dets, H, W)
        for tr in stitched_j:
            tx1, ty1, tx2, ty2 = tr["bbox"]
            pw, ph = tx2 - tx1, ty2 - ty1
            if ph < 48:
                continue
            if pw >= 0.78 * W:
                x1_i, x2_i = 0, W
            else:
                x1_i = max(0, int(round(tx1)) - 10)
                x2_i = min(W, int(round(tx2)) + 10)
            y1_i = max(0, int(round(ty1)))
            y2_i = min(H, int(round(ty2)))

            ang_t, ang_b, cy_t, cy_b = _extract_poly_angles(tr["pts"], W, y1_i, y2_i)
            if x1_i > 0 or x2_i < W:
                ang_t, ang_b = 0, 0
                cy_t, cy_b = y1_i, y2_i
            gv_t = (255 if float(np.median(gray[max(0, min(H - 1, cy_t))])) >= 128.0 else 0) if ang_t != 0 else None
            gv_b = (255 if float(np.median(gray[max(0, min(H - 1, cy_b))])) >= 128.0 else 0) if ang_b != 0 else None
            src_t = "yolo_slanted" if ang_t != 0 else "yolo"
            src_b = "yolo_slanted" if ang_b != 0 else "yolo"

            # Fix 5.1: Tighter non-pan split cap (2,280px / 2.85*W)
            if (y2_i - y1_i) > SPLIT_CAP and x1_i == 0 and x2_i == W:
                panels.extend(_dp_split_segment(y1_i, y2_i, ang_t, ang_b, cy_t, cy_b, gv_t, gv_b, src_t, src_b, tr["conf"]))
            else:
                panels.append({
                    "x1": x1_i, "y1": y1_i, "x2": x2_i, "y2": y2_i, "layout": "panel",
                    "ang_t": ang_t, "ang_b": ang_b, "cy_t": cy_t, "cy_b": cy_b,
                    "gv_t": gv_t, "gv_b": gv_b, "src_t": src_t, "src_b": src_b, "conf": tr["conf"]
                })

    panels.sort(key=lambda p: (p["y1"], p["x1"]))

    # Stage 4b: Unpredicted Artwork Gap Fallback (>= 600px uncovered regions with std >= 22 & ink >= 0.15)
    cov_y = np.zeros(H, dtype=bool)
    for p in panels:
        cov_y[max(0, p["y1"]):min(H, p["y2"])] = True
    pad_u = np.pad((~cov_y).astype(np.int8), (1, 1))
    recovered = []
    for g_s, g_e in zip(np.where(np.diff(pad_u) == 1)[0], np.where(np.diff(pad_u) == -1)[0]):
        if (g_e - g_s) < 600:
            continue
        ty1, ty2 = sb.trim_content_span_b(strip, int(g_s), int(g_e), hard_mask, keep_margin=16)
        if ty2 - ty1 < 220:
            continue
        sub_g = gray[ty1:ty2]
        if float(sub_g.astype(np.float32).std()) < 22.0 or float((sub_g < 205).mean()) < 0.15:
            continue
        if (ty2 - ty1) > SPLIT_CAP:
            recovered.extend(_dp_split_segment(ty1, ty2, src_t="gap_dp", src_b="gap_dp", conf=0.5))
        else:
            recovered.append({
                "x1": 0, "y1": ty1, "x2": W, "y2": ty2, "layout": "panel",
                "ang_t": 0, "ang_b": 0, "cy_t": ty1, "cy_b": ty2,
                "gv_t": None, "gv_b": None, "src_t": "gap_recover", "src_b": "gap_recover", "conf": 0.5
            })
    if recovered:
        panels.extend(recovered)
        panels.sort(key=lambda p: (p["y1"], p["x1"]))

    # Fix 2.1 & 2.2: Horizontal + Vertical Orphaned Gutter-Bubble Reattachment (<= 750px)
    if panels and dialogue_boxes_raw:
        for db in dialogue_boxes_raw:
            bx1, by1, bx2, by2 = float(db[0]), float(db[1]), float(db[2]), float(db[3])
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            max_ov_frac = 0.0
            for p in panels:
                ix = max(0.0, min(bx2, float(p["x2"])) - max(bx1, float(p["x1"])))
                iy = max(0.0, min(by2, float(p["y2"])) - max(by1, float(p["y1"])))
                max_ov_frac = max(max_ov_frac, (ix * iy) / b_area)
            if max_ov_frac >= 0.40:
                continue

            # 1. Check horizontal/Y-overlapping inset panels first!
            y_ov_best = None
            best_y_ov = 0.0
            for pi, p in enumerate(panels):
                y_ov = min(by2, float(p["y2"])) - max(by1, float(p["y1"]))
                if y_ov > best_y_ov:
                    best_y_ov = y_ov
                    y_ov_best = pi
            if y_ov_best is not None and best_y_ov > 0.0:
                p = panels[y_ov_best]
                p["x1"] = min(p["x1"], max(0, int(math.floor(bx1)) - 6))
                p["x2"] = max(p["x2"], min(W, int(math.ceil(bx2)) + 6))
                p["y1"] = min(p["y1"], max(0, int(math.floor(by1)) - 6))
                p["y2"] = max(p["y2"], min(H, int(math.ceil(by2)) + 6))
                if p["ang_t"] == 0:
                    p["cy_t"] = p["y1"]
                if p["ang_b"] == 0:
                    p["cy_b"] = p["y2"]
                continue

            # 2. Otherwise attach vertically to nearest panel within <= 750px
            above_idx, dist_above = None, 1e9
            below_idx, dist_below = None, 1e9
            for pi, p in enumerate(panels):
                if p["y2"] <= by1 + 24:
                    d_a = max(0.0, by1 - p["y2"])
                    if d_a < dist_above:
                        dist_above, above_idx = d_a, pi
                if p["y1"] >= by2 - 24:
                    d_b = max(0.0, p["y1"] - by2)
                    if d_b < dist_below:
                        dist_below, below_idx = d_b, pi
            chosen_side = None
            if below_idx is not None and dist_below <= 750 and (above_idx is None or dist_below <= dist_above + 80):
                chosen_side = ("below", below_idx)
            elif above_idx is not None and dist_above <= 750:
                chosen_side = ("above", above_idx)
            elif below_idx is not None and dist_below <= 750:
                chosen_side = ("below", below_idx)

            if chosen_side is not None:
                side, idx_p = chosen_side
                p = panels[idx_p]
                if side == "below":
                    new_y1 = max(0, int(math.floor(by1)) - 6)
                    if new_y1 < p["y1"]:
                        p["y1"] = new_y1
                        if p["ang_t"] == 0:
                            p["cy_t"] = new_y1
                else:
                    new_y2 = min(H, int(math.ceil(by2)) + 6)
                    if new_y2 > p["y2"]:
                        p["y2"] = new_y2
                        if p["ang_b"] == 0:
                            p["cy_b"] = new_y2
                p["x1"] = min(p["x1"], max(0, int(math.floor(bx1)) - 6))
                p["x2"] = max(p["x2"], min(W, int(math.ceil(bx2)) + 6))

    # Fix 5.2: Re-split any non-pan panel expanded past 2,380px (if squeezed height is also > 2,350px)
    final_panels = []
    for p in panels:
        ph = p["y2"] - p["y1"]
        if p["layout"] != "vertical_pan" and ph > 2380 and p["x1"] == 0 and p["x2"] == W:
            sq_test = mp._squeeze_internal_whitespace(strip[p["y1"]:p["y2"], 0:W])
            if sq_test.shape[0] > 2350:
                final_panels.extend(_dp_split_segment(
                    p["y1"], p["y2"], p["ang_t"], p["ang_b"], p["cy_t"], p["cy_b"],
                    p["gv_t"], p["gv_b"], p["src_t"], p["src_b"], p["conf"]
                ))
                continue
        final_panels.append(p)
    panels = final_panels

    # Stage 5 & 6: Slanted Corner Masking + Cached RapidOCR Edge Credit Stripping & Inward Walk
    def _page_for(y):
        for (nm, y0, y1) in spans:
            if y0 <= y < y1:
                return nm, y0
        return (spans[-1][0], spans[-1][1]) if spans else ("001.jpg", 0)

    def _build_crop(px1, py1, px2, py2, ang_t, ang_b, cy_t, cy_b, gv_t, gv_b, squeeze=False):
        raw_c = strip[py1:py2, px1:px2]
        raw_h = raw_c.shape[0]
        c_out = mp._squeeze_internal_whitespace(raw_c) if squeeze else raw_c
        sq_h = c_out.shape[0]
        use_cy_t = (cy_t - py1) if (ang_t != 0 and sq_h == raw_h) else None
        use_cy_b = (cy_b - py1) if (ang_b != 0 and sq_h == raw_h) else None
        return sb.mask_slanted_corners_b(
            c_out, ang_t, ang_b, gutter_val_top=gv_t, gutter_val_bot=gv_b,
            cy_top_local=use_cy_t, cy_bot_local=use_cy_b
        )

    n_p = len(panels)
    raw_frames = []
    dropped_y_ranges = []
    for s_idx, p in enumerate(panels):
        px1, py1, px2, py2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        if py2 - py1 < 48 or px2 - px1 < 48:
            continue
        crop = strip[py1:py2, px1:px2]
        gc = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if mp._is_blank_crop(gc):
            continue

        is_edge = (s_idx < 5) or (s_idx >= n_p - 5)
        is_tail_end = (s_idx >= n_p - 3)
        b_loc = [(b[0] - px1, b[1] - py1, b[2] - px1, b[3] - py1) for b in bubbles if b[3] > py1 and b[1] < py2]
        nm, y0 = _page_for(py1)
        ocr_hint = (known_ocr_map or {}).get(nm, "")

        trimmed_crop, wt, wb, was_credit, drop_card, cached_ocr = sb.trim_hybrid_credits_or_drop_b(
            crop, is_edge_frame=is_edge, is_tail_end=is_tail_end, bubbles_local=b_loc, ocr_hint=ocr_hint
        )
        if drop_card or trimmed_crop.shape[0] < 60:
            if drop_card:
                dropped_y_ranges.append((py1, py2))
            continue

        fy1 = py1 + wt
        fy2 = py2 - wb
        eff_ang_t = p["ang_t"] if wt == 0 else 0
        eff_ang_b = p["ang_b"] if wb == 0 else 0
        eff_cy_t = int(p["cy_t"]) if eff_ang_t != 0 else int(fy1)
        eff_cy_b = int(p["cy_b"]) if eff_ang_b != 0 else int(fy2)
        cy_t_loc = (eff_cy_t - fy1) if eff_ang_t != 0 else None
        cy_b_loc = (eff_cy_b - fy1) if eff_ang_b != 0 else None
        final_crop = sb.mask_slanted_corners_b(
            trimmed_crop, eff_ang_t, eff_ang_b,
            gutter_val_top=p["gv_t"], gutter_val_bot=p["gv_b"],
            cy_top_local=cy_t_loc, cy_bot_local=cy_b_loc
        )
        pi = int(spans and next((k for k, (_n, a, bb) in enumerate(spans) if a <= fy1 < bb), 0))
        nm, y0 = _page_for(fy1)

        raw_frames.append({
            "source_page": nm,
            "source_index": pi,
            "global_box": [px1, int(fy1), px2, int(fy2)],
            "source_box": [px1, int(max(0, fy1 - y0)), px2, int(fy2 - y0)],
            "w": int(final_crop.shape[1]),
            "h": int(final_crop.shape[0]),
            "layout": p["layout"],
            "angle_top": int(eff_ang_t),
            "angle_bot": int(eff_ang_b),
            "cy_top": int(eff_cy_t),
            "cy_bot": int(eff_cy_b),
            "cut_src_top": p["src_t"],
            "cut_src_bot": p["src_b"],
            "border_conf_top": round(float(p["conf"]), 4),
            "border_conf_bot": round(float(p["conf"]), 4),
            "gutter_val_top": p["gv_t"],
            "gutter_val_bot": p["gv_b"],
            "was_credit": bool(was_credit),
            "ocr_text": cached_ocr,
            "crop_rgb": final_crop,
        })

    # Inward walk from BOTH chapter ends using cached ocr_text
    def _is_standalone_credit_frame(rf_item, is_tail=False):
        c_rgb = rf_item["crop_rgb"]
        if mp._looks_like_credits_panel(c_rgb):
            return True
        fy1, fy2 = rf_item["global_box"][1], rf_item["global_box"][3]
        b_l = [(b[0], b[1] - fy1, b[2], b[3] - fy1) for b in bubbles if b[3] > fy1 and b[1] < fy2]
        txt = rf_item.get("ocr_text", "")
        if not txt and (is_tail or c_rgb.shape[0] <= 850 or sb.is_bubble_only_panel_b(c_rgb, b_l)):
            txt = sb.ocr_crop_rapidocr_b(c_rgb)
            rf_item["ocr_text"] = txt
        if txt:
            if sb._STRICT_PROMO_RE.search(txt):
                if is_tail or c_rgb.shape[0] <= 850 or sb.is_bubble_only_panel_b(c_rgb, b_l):
                    return True
            elif is_tail and sb._AUTHOR_CREDIT_RE.search(txt):
                if (not b_l) and sb.is_bubble_only_panel_b(c_rgb, []):
                    return True
        return False

    consec_story, idx_walk, steps_fwd = 0, 0, 0
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

    consec_story, steps_back, idx_back = 0, 0, len(raw_frames) - 1
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

    # Fix 3: Stage 6b Squeezed-Height Check (`STACKED_BUBBLE_ONLY` Parity with Slicer_B)
    k_f = 0
    while k_f < len(raw_frames) and len(raw_frames) > 1:
        rf = raw_frames[k_f]
        fx1, fy1, fx2, fy2 = rf["global_box"]
        fh = rf["h"]
        is_iso = (fh < MIN_PANEL)
        if not is_iso and fh <= 1150:
            f_loc = [f for f in faces if f[3] > fy1 and f[1] < fy2]
            if len(f_loc) == 0:
                b_loc = [(b[0] - fx1, b[1] - fy1, b[2] - fx1, b[3] - fy1) for b in dialogue_boxes_raw if b[3] > fy1 and b[1] < fy2]
                if not b_loc:
                    # Gate 1D: pure-white/black SFX-only strips
                    g_s = cv2.cvtColor(rf["crop_rgb"], cv2.COLOR_RGB2GRAY)
                    b_px = np.concatenate([g_s[0, :], g_s[-1, :], g_s[:, 0], g_s[:, -1]])
                    bg_v = int(np.median(b_px))
                    if (bg_v >= 235 or bg_v <= 20) and float((np.abs(g_s.astype(np.int16) - bg_v) > 18).mean()) < 0.03:
                        is_iso = True
                    b_loc = [(b[0] - fx1, b[1] - fy1, b[2] - fx1, b[3] - fy1) for b in bubbles if b[3] > fy1 and b[1] < fy2]
                if not is_iso:
                    strict_t = (fh > 360)
                    sq_test = mp._squeeze_internal_whitespace(rf["crop_rgb"]) if not strict_t else None
                    if sb.is_bubble_only_panel_b(rf["crop_rgb"], b_loc, faces_local=f_loc, strict_tall=strict_t) or (
                        (not strict_t) and sb.is_bubble_only_panel_b(sq_test, [], faces_local=f_loc, strict_tall=False)
                    ):
                        is_iso = True
        if not is_iso:
            k_f += 1
            continue

        cand_neighbors = []
        if k_f + 1 < len(raw_frames):
            cand_neighbors.append((k_f + 1, abs(raw_frames[k_f + 1]["global_box"][1] - fy2)))
        if k_f > 0:
            cand_neighbors.append((k_f - 1, abs(fy1 - raw_frames[k_f - 1]["global_box"][3])))
        cand_neighbors.sort(key=lambda t: t[1])

        merged_done = False
        for n_idx, _gap_d in cand_neighbors:
            lo_i, hi_i = min(k_f, n_idx), max(k_f, n_idx)
            nx1 = min(raw_frames[lo_i]["global_box"][0], raw_frames[hi_i]["global_box"][0])
            nx2 = max(raw_frames[lo_i]["global_box"][2], raw_frames[hi_i]["global_box"][2])
            ny1 = min(raw_frames[lo_i]["global_box"][1], raw_frames[hi_i]["global_box"][1])
            ny2 = max(raw_frames[lo_i]["global_box"][3], raw_frames[hi_i]["global_box"][3])
            raw_span_h = ny2 - ny1
            if raw_span_h > 3600 and fh >= MIN_PANEL:
                continue
            m_ang_t = raw_frames[lo_i]["angle_top"]
            m_ang_b = raw_frames[hi_i]["angle_bot"]
            m_cy_t = raw_frames[lo_i].get("cy_top", ny1)
            m_cy_b = raw_frames[hi_i].get("cy_bot", ny2)
            m_gv_t = raw_frames[lo_i].get("gutter_val_top")
            m_gv_b = raw_frames[hi_i].get("gutter_val_bot")
            merged_crop = _build_crop(nx1, ny1, nx2, ny2, m_ang_t, m_ang_b, m_cy_t, m_cy_b, m_gv_t, m_gv_b, squeeze=True)
            sq_h = int(merged_crop.shape[0])
            if sq_h <= 2440 or raw_span_h <= 2440 or (fh < MIN_PANEL and sq_h <= 2650):
                nm, y0 = _page_for(ny1)
                raw_frames[lo_i] = {
                    "source_page": nm,
                    "source_index": raw_frames[lo_i]["source_index"],
                    "global_box": [int(nx1), int(ny1), int(nx2), int(ny2)],
                    "source_box": [int(nx1), int(max(0, ny1 - y0)), int(nx2), int(ny2 - y0)],
                    "w": int(merged_crop.shape[1]),
                    "h": sq_h,
                    "layout": raw_frames[lo_i]["layout"] if raw_frames[lo_i]["layout"] == "vertical_pan" else raw_frames[hi_i]["layout"],
                    "angle_top": m_ang_t,
                    "angle_bot": m_ang_b,
                    "cy_top": int(m_cy_t),
                    "cy_bot": int(m_cy_b),
                    "cut_src_top": raw_frames[lo_i].get("cut_src_top", "yolo"),
                    "cut_src_bot": raw_frames[hi_i].get("cut_src_bot", "yolo"),
                    "border_conf_top": raw_frames[lo_i].get("border_conf_top", 1.0),
                    "border_conf_bot": raw_frames[hi_i].get("border_conf_bot", 1.0),
                    "gutter_val_top": m_gv_t,
                    "gutter_val_bot": m_gv_b,
                    "was_credit": raw_frames[lo_i].get("was_credit", False) or raw_frames[hi_i].get("was_credit", False),
                    "ocr_text": "",
                    "crop_rgb": merged_crop,
                }
                del raw_frames[hi_i]
                k_f = max(0, lo_i)
                merged_done = True
                break
        if not merged_done:
            k_f += 1

    # Fix 2.3: Post-Stage-6b Orphaned Dialogue Sweep across surviving raw_frames
    modified_idxs = set()
    if raw_frames and dialogue_boxes_raw:
        for db in dialogue_boxes_raw:
            bx1, by1, bx2, by2 = float(db[0]), float(db[1]), float(db[2]), float(db[3])
            if any(min(by2, dy2) - max(by1, dy1) > 0.5 * max(1.0, by2 - by1) for (dy1, dy2) in dropped_y_ranges):
                continue
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            max_ov = 0.0
            for rf in raw_frames:
                fx1, fy1, fx2, fy2 = rf["global_box"]
                ix = max(0.0, min(bx2, float(fx2)) - max(bx1, float(fx1)))
                iy = max(0.0, min(by2, float(fy2)) - max(by1, float(fy1)))
                max_ov = max(max_ov, (ix * iy) / b_area)
            if max_ov >= 0.40:
                continue
            # Find nearest surviving frame (prefer Y-overlapping, else closest vertical distance <= 800px)
            best_fi = None
            best_dist = 1e9
            for fi, rf in enumerate(raw_frames):
                fx1, fy1, fx2, fy2 = rf["global_box"]
                if min(by2, fy2) - max(by1, fy1) > 0:
                    best_fi = fi
                    best_dist = 0.0
                    break
                d_v = (fy1 - by2) if by2 <= fy1 else (by1 - fy2)
                if 0 <= d_v < best_dist:
                    best_dist = d_v
                    best_fi = fi
            if best_fi is not None and best_dist <= 800.0:
                rf = raw_frames[best_fi]
                fx1, fy1, fx2, fy2 = rf["global_box"]
                nx1 = min(fx1, max(0, int(math.floor(bx1)) - 6))
                nx2 = max(fx2, min(W, int(math.ceil(bx2)) + 6))
                ny1 = min(fy1, max(0, int(math.floor(by1)) - 6))
                ny2 = max(fy2, min(H, int(math.ceil(by2)) + 6))
                rf["global_box"] = [nx1, ny1, nx2, ny2]
                if ny1 < fy1 and rf["angle_top"] == 0:
                    rf["cy_top"] = ny1
                if ny2 > fy2 and rf["angle_bot"] == 0:
                    rf["cy_bot"] = ny2
                modified_idxs.add(best_fi)

    # Fix 4 + Phase 4C Fix 2: Post-Stage-6b 4-Pass Final Convergence Safety Guard + Post-Fix-4 Height Guard
    _ensure_row_cost()
    protect_boxes = list(faces) + list(dialogue_boxes_raw)
    strong_faces = [f for f in faces if (len(f) < 5 or f[4] >= 0.45)]

    def _converge_box(fy1, fy2, ang_t, ang_b, max_passes=4):
        changed_any = False
        for _pass in range(max_passes):
            changed = False
            if ang_b == 0 and 20 < fy2 < H - 20:
                yl, yh = max(0, fy2 - 6), min(H, fy2 + 7)
                if float(g_fill_all[yl:yh].max()) < 0.85 and float(horiz_line_frac[yl:yh].max()) < 0.55:
                    for fc in strong_faces:
                        f_y1, f_y2 = float(fc[1]), float(fc[3])
                        fh_f = max(24.0, f_y2 - f_y1)
                        if f_y2 <= fy2 <= (f_y2 + 1.0 * fh_f):
                            new_y2 = min(H, int(math.ceil(f_y2 + 1.05 * fh_f)) + 4)
                            if new_y2 > fy2:
                                fy2 = new_y2
                                changed = True
                            break
            if ang_t == 0 and 20 < fy1 < H - 20:
                yl, yh = max(0, fy1 - 6), min(H, fy1 + 7)
                if float(g_fill_all[yl:yh].max()) < 0.85 and float(horiz_line_frac[yl:yh].max()) < 0.55:
                    for fc in strong_faces:
                        f_y1, f_y2 = float(fc[1]), float(fc[3])
                        fh_f = max(24.0, f_y2 - f_y1)
                        if f_y2 <= fy1 <= (f_y2 + 1.0 * fh_f):
                            # Prefer shifting fy1 DOWN past the previous panel's neck zone rather than
                            # expanding UP into the previous panel (which duplicated prev frames into >2600px monsters).
                            new_y1_down = int(math.ceil(f_y2 + 1.05 * fh_f)) + 4
                            cuts_prot_down = any(
                                float(b[1]) + 0.12 * max(1.0, float(b[3]) - float(b[1])) < new_y1_down < float(b[3]) - 0.12 * max(1.0, float(b[3]) - float(b[1]))
                                for b in protect_boxes
                            )
                            if new_y1_down <= fy2 - 260 and not cuts_prot_down:
                                if new_y1_down > fy1:
                                    fy1 = new_y1_down
                                    changed = True
                            else:
                                new_y1 = max(0, int(math.floor(f_y1)) - 4)
                                if new_y1 < fy1:
                                    fy1 = new_y1
                                    changed = True
                            break
            for b in protect_boxes:
                by1, by2 = float(b[1]), float(b[3])
                bh = max(1.0, by2 - by1)
                if by1 + 0.12 * bh < fy1 < by2 - 0.12 * bh:
                    new_y1 = max(0, int(math.floor(by1)) - 6)
                    if new_y1 < fy1:
                        fy1 = new_y1
                        ang_t = 0
                        changed = True
                if by1 + 0.12 * bh < fy2 < by2 - 0.12 * bh:
                    new_y2 = min(H, int(math.ceil(by2)) + 6)
                    if new_y2 > fy2:
                        fy2 = new_y2
                        ang_b = 0
                        changed = True
            if changed:
                changed_any = True
            else:
                break
        return int(fy1), int(fy2), int(ang_t), int(ang_b), changed_any

    final_raw_frames = []
    for fi, rf in enumerate(raw_frames):
        fx1, fy1, fx2, fy2 = rf["global_box"]
        ang_t, ang_b = rf["angle_top"], rf["angle_bot"]
        fy1, fy2, ang_t, ang_b, changed_any = _converge_box(fy1, fy2, ang_t, ang_b, max_passes=4)
        is_non_pan = (rf["layout"] != "vertical_pan")
        needs_update = changed_any or (fi in modified_idxs) or (is_non_pan and rf["h"] > 2400)

        if not needs_update:
            final_raw_frames.append(rf)
            continue

        cy_t_eff = int(rf["cy_top"]) if ang_t != 0 else int(fy1)
        cy_b_eff = int(rf["cy_bot"]) if ang_b != 0 else int(fy2)
        do_squeeze = is_non_pan and ((fy2 - fy1) > 2400 or rf["h"] > 2440)
        new_crop = _build_crop(
            fx1, fy1, fx2, fy2, ang_t, ang_b, cy_t_eff, cy_b_eff,
            rf.get("gutter_val_top"), rf.get("gutter_val_bot"), squeeze=do_squeeze
        )

        # Phase 4C/4D Fix 2: If non-pan squeezed height is still > 2400px (even if slightly inset),
        # split cleanly via _dp_split_segment with safe zero-face/zero-bubble fallback + squeeze on sub-parts!
        if is_non_pan and int(new_crop.shape[0]) > 2400:
            sub_parts = _dp_split_segment(
                fy1, fy2, ang_t, ang_b, cy_t_eff, cy_b_eff,
                rf.get("gutter_val_top"), rf.get("gutter_val_bot"),
                rf.get("cut_src_top", "yolo"), rf.get("cut_src_bot", "yolo"),
                rf.get("border_conf_top", 0.8)
            )
            # Phase 4D Fix 2.3: If any DP part is still > 2480px (e.g. morphology bloat in hard_mask),
            # find a strict zero-face-cut, zero-bubble-cut split row minimizing row_cost_base inside [y1+420, y2-420].
            if any((sp["y2"] - sp["y1"]) > 2480 for sp in sub_parts):
                strict_prot = np.zeros(H, dtype=bool)
                for fc in faces:
                    strict_prot[max(0, int(math.floor(fc[1])) - 6):min(H, int(math.ceil(fc[3])) + 6)] = True
                for db in dialogue_boxes_raw:
                    strict_prot[max(0, int(math.floor(db[1])) - 6):min(H, int(math.ceil(db[3])) + 6)] = True
                refined_parts = []
                for sp in sub_parts:
                    if (sp["y2"] - sp["y1"]) > 2480:
                        lo_s, hi_s = sp["y1"] + 420, sp["y2"] - 420
                        valid_rows = (np.where(~strict_prot[lo_s:hi_s])[0] + lo_s) if hi_s > lo_s else np.array([], dtype=np.int64)
                        if len(valid_rows) > 0:
                            best_y = int(valid_rows[int(np.argmin(row_cost_base[valid_rows]))])
                            refined_parts.append({**sp, "y2": best_y, "ang_b": 0, "cy_b": best_y, "gv_b": None, "src_b": "dp_safe"})
                            refined_parts.append({**sp, "y1": best_y, "ang_t": 0, "cy_t": best_y, "gv_t": None, "src_t": "dp_safe"})
                            continue
                    refined_parts.append(sp)
                sub_parts = refined_parts
            for sp in sub_parts:
                sy1, sy2, sat, sab, _ = _converge_box(sp["y1"], sp["y2"], sp["ang_t"], sp["ang_b"], max_passes=2)
                scyt = int(sp["cy_t"]) if sat != 0 else int(sy1)
                scyb = int(sp["cy_b"]) if sab != 0 else int(sy2)
                scrop = _build_crop(
                    fx1, sy1, fx2, sy2, sat, sab, scyt, scyb,
                    sp["gv_t"], sp["gv_b"], squeeze=(sy2 - sy1 > 2350)
                )
                nm_s, y0_s = _page_for(sy1)
                pi_s = int(spans and next((k for k, (_n, a, bb) in enumerate(spans) if a <= sy1 < bb), 0))
                final_raw_frames.append({
                    "source_page": nm_s,
                    "source_index": pi_s,
                    "global_box": [int(fx1), int(sy1), int(fx2), int(sy2)],
                    "source_box": [int(fx1), int(max(0, sy1 - y0_s)), int(fx2), int(sy2 - y0_s)],
                    "w": int(scrop.shape[1]),
                    "h": int(scrop.shape[0]),
                    "layout": "panel",
                    "angle_top": int(sat),
                    "angle_bot": int(sab),
                    "cy_top": int(scyt),
                    "cy_bot": int(scyb),
                    "cut_src_top": sp["src_t"],
                    "cut_src_bot": sp["src_b"],
                    "border_conf_top": rf.get("border_conf_top", 1.0),
                    "border_conf_bot": rf.get("border_conf_bot", 1.0),
                    "gutter_val_top": sp["gv_t"],
                    "gutter_val_bot": sp["gv_b"],
                    "was_credit": rf.get("was_credit", False),
                    "ocr_text": "",
                    "crop_rgb": scrop,
                })
        else:
            rf["global_box"] = [fx1, int(fy1), fx2, int(fy2)]
            rf["angle_top"] = int(ang_t)
            rf["angle_bot"] = int(ang_b)
            rf["cy_top"] = int(cy_t_eff)
            rf["cy_bot"] = int(cy_b_eff)
            nm, y0 = _page_for(fy1)
            rf["source_page"] = nm
            rf["source_box"] = [fx1, int(max(0, fy1 - y0)), fx2, int(fy2 - y0)]
            rf["w"] = int(new_crop.shape[1])
            rf["h"] = int(new_crop.shape[0])
            rf["crop_rgb"] = new_crop
            final_raw_frames.append(rf)
    raw_frames = final_raw_frames

    # Phase 4E: Post-Fix-4 Deduplication & Overlap Resolution (zero-orphan, zero-face/bubble-cut safe)
    # 1) Prune nested/heavily-contained duplicate panels (e.g. from Fix 2.3 / Fix 4 expansions or sliding windows)
    if len(raw_frames) > 1:
        def _bub_covered_by(db, box):
            bx1, by1, bx2, by2 = float(db[0]), float(db[1]), float(db[2]), float(db[3])
            ba = max(1.0, (bx2 - bx1) * (by2 - by1))
            ix = max(0.0, min(bx2, float(box[2])) - max(bx1, float(box[0])))
            iy = max(0.0, min(by2, float(box[3])) - max(by1, float(box[1])))
            return (ix * iy) / ba >= 0.40

        order_desc = sorted(
            range(len(raw_frames)),
            key=lambda idx: (
                1 if raw_frames[idx]["layout"] == "vertical_pan" else 0,
                max(1, (raw_frames[idx]["global_box"][2] - raw_frames[idx]["global_box"][0]) *
                    (raw_frames[idx]["global_box"][3] - raw_frames[idx]["global_box"][1])),
            ),
            reverse=True,
        )
        kept_idxs = []
        for idx in order_desc:
            bx1, by1, bx2, by2 = raw_frames[idx]["global_box"]
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            ba = float(bw * bh)
            subsumed_by = None
            for ki in kept_idxs:
                kx1, ky1, kx2, ky2 = raw_frames[ki]["global_box"]
                ix = max(0, min(bx2, kx2) - max(bx1, kx1))
                iy = max(0, min(by2, ky2) - max(by1, ky1))
                if ix <= 0 or iy <= 0:
                    continue
                area_ov = float(ix * iy) / ba
                y_ov_frac = float(iy) / float(bh)
                x_ov_frac = float(ix) / float(bw)
                if area_ov >= 0.65 or (y_ov_frac >= 0.70 and x_ov_frac >= 0.55):
                    subsumed_by = ki
                    break
            if subsumed_by is not None:
                # Verify zero dialogue bubbles are orphaned by dropping idx
                kbox = raw_frames[subsumed_by]["global_box"]
                jbox = raw_frames[idx]["global_box"]
                safe_to_drop = True
                for db in dialogue_boxes_raw:
                    if _bub_covered_by(db, jbox) and not _bub_covered_by(db, kbox):
                        if not any(_bub_covered_by(db, raw_frames[other_k]["global_box"]) for other_k in kept_idxs):
                            safe_to_drop = False
                            break
                if safe_to_drop:
                    continue
            kept_idxs.append(idx)

        raw_frames = [raw_frames[i] for i in sorted(kept_idxs, key=lambda i: (raw_frames[i]["global_box"][1], raw_frames[i]["global_box"][0]))]

    # 2) Resolve remaining >= 30% vertical overlaps between adjacent full-width non-pan frames
    if len(raw_frames) > 1:
        strict_prot_ov = None
        dedup_frames = []
        idx_r = 0
        while idx_r < len(raw_frames):
            cur = raw_frames[idx_r]
            if not dedup_frames:
                dedup_frames.append(cur)
                idx_r += 1
                continue
            prev = dedup_frames[-1]
            if prev["layout"] == "vertical_pan" or cur["layout"] == "vertical_pan":
                dedup_frames.append(cur)
                idx_r += 1
                continue
            ax1, ay1, ax2, ay2 = prev["global_box"]
            bx1, by1, bx2, by2 = cur["global_box"]
            x_ov = max(0, min(ax2, bx2) - max(ax1, bx1))
            min_w = max(1, min(ax2 - ax1, bx2 - bx1))
            y_ov = max(0, min(ay2, by2) - max(ay1, by1))
            min_h = max(1, min(ay2 - ay1, by2 - by1))
            if (x_ov / float(min_w)) >= 0.55 and (y_ov / float(min_h)) >= 0.30:
                ny1, ny2 = min(ay1, by1), max(ay2, by2)
                nx1, nx2 = min(ax1, bx1), max(ax2, bx2)
                if (ny2 - ny1) <= 2350:
                    # Merge overlapping double-detection into one clean frame
                    m_at = prev["angle_top"] if ay1 <= by1 else cur["angle_top"]
                    m_ab = cur["angle_bot"] if by2 >= ay2 else prev["angle_bot"]
                    m_cyt = int(prev["cy_top"]) if (ay1 <= by1 and m_at != 0) else (int(cur["cy_top"]) if m_at != 0 else int(ny1))
                    m_cyb = int(cur["cy_bot"]) if (by2 >= ay2 and m_ab != 0) else (int(prev["cy_bot"]) if m_ab != 0 else int(ny2))
                    m_gvt = prev.get("gutter_val_top") if ay1 <= by1 else cur.get("gutter_val_top")
                    m_gvb = cur.get("gutter_val_bot") if by2 >= ay2 else prev.get("gutter_val_bot")
                    mcrop = _build_crop(nx1, ny1, nx2, ny2, m_at, m_ab, m_cyt, m_cyb, m_gvt, m_gvb, squeeze=(ny2 - ny1 > 2200))
                    nm_m, y0_m = _page_for(ny1)
                    prev["global_box"] = [int(nx1), int(ny1), int(nx2), int(ny2)]
                    prev["source_page"] = nm_m
                    prev["source_box"] = [int(nx1), int(max(0, ny1 - y0_m)), int(nx2), int(ny2 - y0_m)]
                    prev["w"] = int(mcrop.shape[1])
                    prev["h"] = int(mcrop.shape[0])
                    prev["angle_top"] = int(m_at)
                    prev["angle_bot"] = int(m_ab)
                    prev["cy_top"] = int(m_cyt)
                    prev["cy_bot"] = int(m_cyb)
                    prev["crop_rgb"] = mcrop
                    idx_r += 1
                    continue
                elif prev["angle_bot"] == 0 and cur["angle_top"] == 0:
                    # Combined > 2350px: trim shared boundary at a zero-face/zero-bubble/zero-neck row
                    if strict_prot_ov is None:
                        strict_prot_ov = np.zeros(H, dtype=bool)
                        for fc in faces:
                            fh_f = max(24.0, float(fc[3]) - float(fc[1]))
                            strict_prot_ov[max(0, int(math.floor(fc[1])) - 6):min(H, int(math.ceil(fc[3] + 1.05 * fh_f)) + 6)] = True
                        for db in dialogue_boxes_raw:
                            strict_prot_ov[max(0, int(math.floor(db[1])) - 6):min(H, int(math.ceil(db[3])) + 6)] = True
                    lo_t = max(max(ay1, by1), ay1 + 260)
                    hi_t = min(min(ay2, by2), by2 - 260)
                    if hi_t > lo_t:
                        vrows = np.where(~strict_prot_ov[lo_t:hi_t])[0] + lo_t
                        if len(vrows) > 0:
                            cut_y = int(vrows[int(np.argmin(row_cost_base[vrows]))])
                            acrop = _build_crop(ax1, ay1, ax2, cut_y, prev["angle_top"], 0,
                                                int(prev["cy_top"]) if prev["angle_top"] != 0 else int(ay1), cut_y,
                                                prev.get("gutter_val_top"), None, squeeze=(cut_y - ay1 > 2350))
                            nm_a, y0_a = _page_for(ay1)
                            prev["global_box"] = [int(ax1), int(ay1), int(ax2), int(cut_y)]
                            prev["source_box"] = [int(ax1), int(max(0, ay1 - y0_a)), int(ax2), int(cut_y - y0_a)]
                            prev["w"] = int(acrop.shape[1]); prev["h"] = int(acrop.shape[0])
                            prev["cy_bot"] = int(cut_y); prev["crop_rgb"] = acrop

                            bcrop = _build_crop(bx1, cut_y, bx2, by2, 0, cur["angle_bot"],
                                                cut_y, int(cur["cy_bot"]) if cur["angle_bot"] != 0 else int(by2),
                                                None, cur.get("gutter_val_bot"), squeeze=(by2 - cut_y > 2350))
                            nm_b, y0_b = _page_for(cut_y)
                            cur["global_box"] = [int(bx1), int(cut_y), int(bx2), int(by2)]
                            cur["source_page"] = nm_b
                            cur["source_box"] = [int(bx1), int(max(0, cut_y - y0_b)), int(bx2), int(by2 - y0_b)]
                            cur["w"] = int(bcrop.shape[1]); cur["h"] = int(bcrop.shape[0])
                            cur["cy_top"] = int(cut_y); cur["crop_rgb"] = bcrop
            dedup_frames.append(cur)
            idx_r += 1
        raw_frames = dedup_frames

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
