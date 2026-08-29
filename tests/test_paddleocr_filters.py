import os
import importlib.util
import sys
from pathlib import Path

import numpy as np

os.environ["SKIP_OCR_INIT"] = "1"

# Load the real filter/merge functions dynamically from
# mini-services/paddleocr-service/main.py — the same pattern used by
# tests/test_paddleocr_parser.py. This file used to locally re-implement
# these functions verbatim instead of importing them, which meant it
# validated a stale fork that had already drifted from production (e.g. a
# local CONFIDENCE_CUTOFF of 0.70 vs. the real 0.40) and would not catch a
# regression in the actual cascade.
main_path = Path(__file__).parent.parent / "mini-services" / "paddleocr-service" / "main.py"
spec = importlib.util.spec_from_file_location("paddleocr_main", main_path)
paddleocr_main = importlib.util.module_from_spec(spec)
sys.modules["paddleocr_main"] = paddleocr_main
spec.loader.exec_module(paddleocr_main)

_TextRegion = paddleocr_main._TextRegion
_is_slash_or_math_artifact = paddleocr_main._is_slash_or_math_artifact
_symbol_ratio_exceeded = paddleocr_main._symbol_ratio_exceeded
_is_graphic_logo = paddleocr_main._is_graphic_logo
_clean_and_normalize_ocr_text = paddleocr_main._clean_and_normalize_ocr_text
_merge_regions = paddleocr_main._merge_regions


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


def test_ocr_cascade_skips_expensive_passes_when_zero_regions_detected(monkeypatch):
    """A blank/silent panel (detector finds zero text regions) must return
    immediately instead of running the 3 preprocessing variants + Tesseract
    fallback cascade. Manhwa chapters are frequently 80%+ silent panels, so
    running that full cascade on every one of them was the single biggest
    driver of pipeline slowness — this is a regression guard for the fix.
    """
    call_count = {"ocr": 0, "tesseract": 0}

    def fake_run_ocr_on_image(img, options=None):
        call_count["ocr"] += 1
        return []  # zero detected regions, every call

    def fake_run_tesseract_ocr(img):
        call_count["tesseract"] += 1
        return "", 0.0

    monkeypatch.setattr(paddleocr_main, "_run_ocr_on_image", fake_run_ocr_on_image)
    monkeypatch.setattr(paddleocr_main, "_run_tesseract_ocr", fake_run_tesseract_ocr)
    monkeypatch.setattr(paddleocr_main, "_detect_ui_card_or_borders", lambda img: False)

    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    text, conf, regions, status, quality, candidates, reason = paddleocr_main._ocr_with_cascade(blank_img)

    assert call_count["ocr"] == 1, "expected exactly one OCR pass (standard only) for a zero-region panel"
    assert call_count["tesseract"] == 0, "Tesseract fallback should not run when the detector found zero regions"
    assert regions == 0
    assert text == ""
    assert "skipped_cascade_zero_regions_detected" in reason
