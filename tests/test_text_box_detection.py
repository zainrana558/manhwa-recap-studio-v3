"""Tests for _detect_text_boxes' tall-image band-splitting logic (added
alongside the switch from a Manga109-only YOLO panel/text detector to
ogkalu/comic-text-and-bubble-detector, an RT-DETR-v2 model trained on
Manga/Webtoon/Manhua/Western comics with "Tall Webtoons split
vertically" as an explicit training preprocessing step -- this mirrors
that same preprocessing at inference time instead of feeding the model
a single extreme-aspect-ratio image its training distribution didn't
contain examples of).

These tests mock _detect_text_boxes_raw directly rather than requiring
the actual ~170MB RT-DETR-v2 model (not available in this environment;
huggingface.co is not reachable from here), so they verify the
band-splitting, coordinate-offsetting, and de-duplication logic in
isolation from actual detection accuracy, which can only be validated
against real content on a real deployment.
"""
import numpy as np
from pipeline.master_pipeline import _detect_text_boxes, TALL_IMAGE_SPLIT_ASPECT_RATIO
import pipeline.master_pipeline as mp


def test_short_image_skips_band_splitting(monkeypatch):
    """An image at or below the split threshold should be detected in
    one pass -- no banding needed."""
    calls = []

    def fake_raw(img_gray):
        calls.append(img_gray.shape)
        return [(1, 1, 10, 10)]

    monkeypatch.setattr(mp, "_detect_text_boxes_raw", fake_raw)
    short_gray = np.zeros((100, 100), dtype=np.uint8)  # aspect ratio 1.0
    boxes = _detect_text_boxes(short_gray)
    assert len(calls) == 1, "should call the raw detector exactly once, no banding"
    assert calls[0] == (100, 100), "should pass the whole image unmodified"
    assert boxes == [(1, 1, 10, 10)]


def test_tall_image_gets_split_into_overlapping_bands(monkeypatch):
    """An image well above the split threshold should be broken into
    multiple bands, each no taller than width * TALL_IMAGE_SPLIT_ASPECT_RATIO,
    with correctly offset coordinates in the final result."""
    call_shapes = []

    def fake_raw(img_gray):
        h, w = img_gray.shape[:2]
        call_shapes.append((w, h))
        # A box near the top of whatever band it's given
        return [(10, 5, w - 10, 30)]

    monkeypatch.setattr(mp, "_detect_text_boxes_raw", fake_raw)
    tall_gray = np.zeros((2500, 500), dtype=np.uint8)  # aspect ratio 5.0
    boxes = _detect_text_boxes(tall_gray)

    assert len(call_shapes) > 1, "a 5.0-aspect-ratio image must be split into multiple bands"
    for w, h in call_shapes:
        assert h <= w * TALL_IMAGE_SPLIT_ASPECT_RATIO + 1, "no band should exceed the split aspect ratio"

    # Every returned box must be correctly offset into the ORIGINAL
    # image's coordinate space, not left relative to its own band.
    for (x1, y1, x2, y2) in boxes:
        assert 0 <= y1 < 2500
        assert 0 <= y2 <= 2500
    # Boxes from different (non-overlapping-content) bands must all be
    # kept, not accidentally merged into one.
    assert len(boxes) == len(call_shapes), "each band's distinct detection should survive to the final result"


def test_duplicate_detection_in_overlap_zone_is_deduplicated(monkeypatch):
    """The same real text box, sitting in the overlap zone between two
    adjacent bands, will legitimately be detected by both bands (after
    coordinate offsetting, at nearly the same final location) -- this
    must collapse to one box, not two, or the content mask would just
    get a slightly-redundant fill (harmless) but the underlying
    assumption ("each detection is one distinct text region") would be
    silently wrong for any future caller counting detections.
    """
    call_count = [0]

    def fake_raw(img_gray):
        call_count[0] += 1
        w = img_gray.shape[1]
        if call_count[0] == 1:
            return [(10, 850, w - 10, 900)]  # band 1: near its own bottom
        elif call_count[0] == 2:
            return [(10, 50, w - 10, 100)]   # band 2: same real content, near its own top
        return []

    monkeypatch.setattr(mp, "_detect_text_boxes_raw", fake_raw)
    tall_gray = np.zeros((1900, 500), dtype=np.uint8)  # aspect ratio 3.8 -> splits into 2 bands
    boxes = _detect_text_boxes(tall_gray)
    assert len(boxes) == 1, f"expected exactly 1 deduplicated box, got {boxes}"
    assert boxes[0] == (10, 850, 490, 900)


def test_no_detections_returns_empty_list(monkeypatch):
    monkeypatch.setattr(mp, "_detect_text_boxes_raw", lambda img_gray: [])
    tall_gray = np.zeros((2000, 400), dtype=np.uint8)
    assert _detect_text_boxes(tall_gray) == []
