from pathlib import Path
from pipeline.production import *
import re
from pathlib import Path
from pipeline.production import *

CONFIDENCE_CUTOFF = 0.70
SYMBOL_RATIO_LIMIT = 0.35

class _TextRegion:
    def __init__(self, text, confidence, x_min, y_min, y_max, x_max):
        self.text = text
        self.confidence = confidence
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.x_max = x_max

def _is_slash_or_math_artifact(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    if re.fullmatch(r'[/\-\\—_\s]+', s):
        return True
    if re.search(r'[/\\—\-]{2,}', s) and not re.search(r'[a-zA-Z0-9]', s):
        return True
    if re.fullmatch(r'[*+\\/\-—=]\s*[0-9A-Za-z]{0,3}', s):
        return True
    if re.fullmatch(r'[*+\\/\-—=]+', s):
        return True
    return False

def _symbol_ratio_exceeded(text: str, max_ratio: float = SYMBOL_RATIO_LIMIT) -> bool:
    if not text:
        return True
    alnum_count = len(re.findall(r'[a-zA-Z0-9]', text))
    if alnum_count == 0:
        return True
    non_alnum_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    ratio = non_alnum_count / float(alnum_count)
    return ratio > max_ratio

def _is_graphic_logo(region: '_TextRegion', img_h: int = 0, img_w: int = 0) -> bool:
    box_w = region.x_max - region.x_min
    box_h = region.y_max - region.y_min
    if box_h <= 0 or box_w <= 0:
        return False
    text_lower = region.text.lower()
    if re.search(r'\bsouls?\s+lac(?:ing|e)\b', text_lower):
        return True
    if img_h > 0 and img_w > 0:
        aspect_ratio = box_w / float(box_h)
        area_ratio = (box_w * box_h) / float(img_w * img_h)
        if (area_ratio > 0.15 or aspect_ratio > 5.0 or box_h > img_h * 0.4) and region.confidence < 0.85:
            if not re.fullmatch(r'[\w\s.,!\'\"]+', region.text) or region.confidence < 0.75:
                return True
    return False

def _clean_and_normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r'\s*\b(minus|dash|underscore)\b\s*$', '...', text, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)
    t = re.sub(r'\bHO[0O]\b', 'HOO', t)
    t = re.sub(r'\bHO\s+O\b', 'HOO', t)
    t = re.sub(r'\bgood-curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)
    t = re.sub(r'\bgood\s+curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)
    t = re.sub(r'\bB\s+to\s+be\s+continued\.*', 'To Be Continued...', t, flags=re.IGNORECASE)
    t = re.sub(r'^\s*B\s+to\s+be\b(?!\s+continued)', 'To Be Continued', t, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)
    return t

def _sort_regions_reading_order(regions, is_ui_box=False):
    if not regions:
        return regions
    if is_ui_box:
        return sorted(regions, key=lambda r: (r.y_min, r.x_min))
    heights = [r.y_max - r.y_min for r in regions]
    mean_height = sum(heights) / len(heights) if heights else 20.0
    vertical_tolerance = max(mean_height * 0.4, 10.0)
    remaining = sorted(regions, key=lambda r: r.y_min)
    rows = []
    for r in remaining:
        placed = False
        for row in rows:
            row_y = sum(rr.y_min for rr in row) / len(row)
            if abs(r.y_min - row_y) < vertical_tolerance:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])
    rows.sort(key=lambda row: sum(rr.y_min for rr in row) / len(row))
    ordered = []
    for row in rows:
        row.sort(key=lambda rr: rr.x_min)
        ordered.extend(row)
    return ordered

def _merge_regions(regions, is_ui_box=False):
    if not regions:
        return "", 0.0, 0
    sorted_regions = _sort_regions_reading_order(regions, is_ui_box=is_ui_box)
    lines = []
    current_line = [sorted_regions[0]]
    for region in sorted_regions[1:]:
        prev = current_line[-1]
        vertical_gap = abs(region.y_min - prev.y_min)
        mean_h = (region.y_max - region.y_min + prev.y_max - prev.y_min) / 2
        threshold = max(mean_h * 0.2, 4.0) if is_ui_box else max(mean_h * 0.5, 10.0)
        if vertical_gap < threshold:
            current_line.append(region)
        else:
            lines.append(current_line)
            current_line = [region]
    lines.append(current_line)
    text_parts = []
    all_confidences = []
    for line in lines:
        line_sorted = sorted(line, key=lambda r: r.x_min)
        line_text = " ".join(_clean_and_normalize_ocr_text(r.text.strip()) for r in line_sorted if r.text.strip())
        if line_text:
            text_parts.append(line_text)
        for r in line_sorted:
            if r.confidence > 0:
                all_confidences.append(r.confidence)
    merged_text = _clean_and_normalize_ocr_text(" ".join(text_parts))
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    return merged_text, round(avg_confidence, 4), len(sorted_regions)


def test_paddleocr_filters():
    # 1. Slash & math symbol filter
    assert _is_slash_or_math_artifact("///\\/\\//")
    assert _is_slash_or_math_artifact("*1C5")
    assert _is_slash_or_math_artifact("—")
    assert _is_slash_or_math_artifact("-")
    assert not _is_slash_or_math_artifact("Hello World")

    # 2. Symbol ratio limit (> 0.35)
    assert _symbol_ratio_exceeded("!!!???@@@")
    assert _symbol_ratio_exceeded("A!!!#$")
    assert not _symbol_ratio_exceeded("Hello World!")

    # 3. Graphic logo & title card exclusion
    reg_logo = _TextRegion("Souls Lacing", 0.65, 50, 50, 200, 500)
    assert _is_graphic_logo(reg_logo, img_h=1000, img_w=1000)

    reg_normal = _TextRegion("Normal dialogue here", 0.95, 10, 10, 30, 200)
    assert not _is_graphic_logo(reg_normal, img_h=1000, img_w=1000)

    # 4. Text normalization and end-card
    assert _clean_and_normalize_ocr_text("good-curdling dash") == "blood-curdling..."
    assert _clean_and_normalize_ocr_text("HO0 HO O") == "HOO HOO"
    assert _clean_and_normalize_ocr_text("B to be continued...") == "To Be Continued..."

    # 5. Line height & sorting for UI quest cards
    regions = [
        _TextRegion("Line 2: Sit-ups 100/100", 0.90, 10, 50, 70, 200),
        _TextRegion("Line 1: Push-ups 100/100", 0.90, 10, 10, 30, 200),
    ]
    merged, avg_conf, count = _merge_regions(regions, is_ui_box=True)
    assert merged == "Line 1: Push-ups 100/100 Line 2: Sit-ups 100/100"
