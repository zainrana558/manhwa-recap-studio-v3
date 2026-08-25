"""
test_image_slicing.py
======================
Unit and integration tests for the PIL/OpenCV image slicing engine refactoring in
pipeline/master_pipeline.py.

Tests cover:
1. Speech Bubble Protection & Minimum Vertical Padding (+30px & Bubble Union Check)
2. Featureless & Visual Effect Frame Suppression (Speed lines / Ice shards / Aura gradients)
3. UI Window Aspect Ratio & Linear Vertical Scroll Framing
4. Clean Panel Boundary Edge Cleaning (Stray top/bottom adjacent panel bleed trimming)
"""

import os
import sys
import numpy as np
import cv2
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
import pipeline.master_pipeline as mp


def test_speech_bubble_union_and_padding():
    """Verify that horizontal cut lines intersecting speech bubbles move below the bubble
    and dynamic 30px padding is applied to protect dialogue.
    """
    h, w = 1000, 800
    img_gray = np.full((h, w), 255, dtype=np.uint8)

    # Panel 1 art region
    img_gray[50:400, 50:750] = 50
    # Speech bubble with dialogue near cut line boundary (380 to 415)
    img_gray[380:415, 200:400] = 255
    cv2.putText(img_gray, "HAAAH!!", (210, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    # Panel 2 art region
    img_gray[500:900, 50:750] = 100

    content_mask = mp._build_dilated_content_mask(img_gray)

    # Test Bubble Union Check
    cut_y = mp._check_bubble_union(content_mask, 410, img_gray.shape)
    assert cut_y >= 416, f"Cut line {cut_y} should be moved below speech bubble (>=416)"

    # Test Dynamic Padding
    box = (50, 50, 700, 360)  # y1=50, y2=410
    padded_box = mp._apply_dynamic_padding(box, content_mask, img_gray.shape, padding_px=30)
    assert padded_box[1] <= 50 and (padded_box[1] + padded_box[3]) >= 420, (
        f"Padded box {padded_box} should expand vertically to cover bubble and padding"
    )


def test_featureless_vfx_frame_suppression():
    """Verify that speed line, particle effect, and aura background slices are identified
    as featureless VFX, while dialogue action panels are preserved.
    """
    h, w = 300, 600
    # Speed lines / gradient slice candidate
    vfx_img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        vfx_img[i, :] = int(100 + 100 * (i / h))
    cv2.line(vfx_img, (0, 0), (w, h), 255, 2)
    cv2.line(vfx_img, (100, 0), (w, h - 100), 255, 2)

    assert mp._is_featureless_vfx(vfx_img), "Speed line slice candidate should be suppressed as featureless VFX"

    # Action panel slice with dialogue text and subject
    action_img = np.full((h, w), 255, dtype=np.uint8)
    action_img[50:250, 50:550] = 40
    action_img[70:130, 80:450] = 255
    cv2.putText(action_img, "I'M AN AWAKENED!", (90, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    assert not mp._is_featureless_vfx(action_img), "Dialogue action panel slice should NOT be suppressed as featureless VFX"


def test_ui_card_aspect_ratio_and_scrolling():
    """Verify that tall rectangular system/quest UI panels (aspect > 1.8) generate
    vertical scroll sub-frames maintaining full resolution width rather than squishing.
    """
    # 1000h x 400w tall UI card (aspect ratio height/width = 2.5)
    ui_crop = Image.new("RGB", (400, 1000), (30, 30, 60))
    scroll_frames = mp._generate_ui_card_scroll_frames(ui_crop, num_scroll_frames=4)

    assert len(scroll_frames) == 4, "Should produce 4 vertical scroll sub-frames"
    for frame in scroll_frames:
        assert frame.size == (1920, 1080), f"Frame canvas size should be 1920x1080, got {frame.size}"


def test_clean_panel_boundary_edge_trimming():
    """Verify that stray top/bottom edge fragments from adjacent panels (e.g. fire panel bleed)
    are cleaned before finalizing slice output.
    """
    h, w = 200, 400
    panel_gray = np.full((h, w), 100, dtype=np.uint8)
    panel_rgb = np.full((h, w, 3), 100, dtype=np.uint8)

    # Stray top border fragment rows (5px white bleed)
    panel_gray[:5, :] = 255
    panel_rgb[:5, :] = 255

    # Stray bottom border fragment rows (5px black bleed)
    panel_gray[195:, :] = 0
    panel_rgb[195:, :] = 0

    cleaned_rgb, cleaned_gray = mp._clean_panel_boundary_edges(panel_rgb, panel_gray, max_scan_px=15)
    assert cleaned_gray.shape[0] == 190, f"Expected 190 rows after trimming stray top/bottom 5px fragments, got {cleaned_gray.shape[0]}"
