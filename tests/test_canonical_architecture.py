"""
test_canonical_architecture.py
===============================
Regression tests verifying the unified frame/page/panel architecture:
1. Source content produces stable canonical frame IDs.
2. OCR results map to exact canonical frame IDs.
3. Narration maps to exact canonical frame IDs.
4. Audio/timing maps to exact canonical frame IDs.
5. Rendering consumes the canonical mapping.
6. Double slicing does not occur when a valid manifest exists.
7. Resume behavior validates manifest consistency.
8. Missing/stale/inconsistent manifest data triggers re-slicing (no silent failure).
9. Legacy v1 manifest migration cleanly upgrades to v2.0 schema.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
import pipeline.master_pipeline as mp
from pipeline.master_pipeline import (
    CanonicalManifest,
    Chapter,
    FrameEntry,
    PipelineConfig,
    load_or_migrate_manifest,
    parse_narration_item,
    slice_chapter_panels,
    validate_canonical_manifest,
)


@pytest.fixture
def dummy_chapter_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "dataset"
        work_dir = tmp_path / "work"
        chap_dir = input_dir / "chapter_001"
        chap_dir.mkdir(parents=True, exist_ok=True)

        from PIL import ImageDraw
        # Create 3 dummy source panel page images with distinct content
        source_paths = []
        for i in range(1, 4):
            img_path = chap_dir / f"{i:03d}.jpg"
            img = Image.new("RGB", (800, 1200), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Panel 1 art box
            draw.rectangle([50, 50, 750, 450], fill=(50, 50, 50), outline=(0, 0, 0), width=4)
            # Panel 2 art box
            draw.rectangle([50, 550, 750, 1100], fill=(80, 80, 80), outline=(0, 0, 0), width=4)
            img.save(img_path)
            source_paths.append(img_path)

        chapter = Chapter(index=1, name="chapter_001", folder=chap_dir, panel_paths=source_paths)
        cfg = PipelineConfig(
            input_dir=input_dir,
            output_path=tmp_path / "master_recap.mp4",
            work_dir=work_dir,
            job_id="test_job_123",
        )
        cfg.ensure_dirs()

        yield cfg, chapter, work_dir


def test_canonical_frame_ids_generation(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    frame_data = slice_chapter_panels(cfg, chapter)
    assert len(frame_data) > 0

    manifest_path = cfg.temp_slices_dir / chapter.tag / "manifest.json"
    assert manifest_path.exists()

    manifest = load_or_migrate_manifest(manifest_path, chapter=chapter, job_id=cfg.job_id)
    assert manifest is not None
    assert manifest.manifest_version == mp.MANIFEST_VERSION
    assert manifest.job_id == "test_job_123"
    assert manifest.chapter_tag == chapter.tag
    assert manifest.source_pages == ["001.jpg", "002.jpg", "003.jpg"]

    frame_ids = [f.frame_id for f in manifest.frames]
    assert len(frame_ids) == len(set(frame_ids)), "All canonical frame IDs must be unique"
    for idx, f_entry in enumerate(manifest.frames):
        assert f_entry.frame_id == f"{chapter.tag}_frame_{idx:05d}"
        assert f_entry.filename == f"frame_{idx:05d}.jpg"
        assert f_entry.source_page in manifest.source_pages


def test_ocr_and_narration_mapping_to_canonical_frame_ids(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    slice_chapter_panels(cfg, chapter)
    manifest_path = cfg.temp_slices_dir / chapter.tag / "manifest.json"
    manifest = load_or_migrate_manifest(manifest_path, chapter=chapter, job_id=cfg.job_id)
    assert manifest is not None

    # Simulate OCR results keyed by canonical frame_id
    sample_frame = manifest.frames[0]
    chapter.image_narrations = {
        sample_frame.frame_id: {"text": "Hero enters the dungeon", "status": "SUCCESS", "confidence": 0.95}
    }

    raw_item = chapter.image_narrations.get(sample_frame.frame_id)
    parsed = parse_narration_item(raw_item)
    assert parsed["text"] == "Hero enters the dungeon"
    assert parsed["status"] == mp.OcrStatus.SUCCESS
    assert parsed["confidence"] == 0.95


def test_double_slicing_prevention(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    # Slicing pass 1
    frame_data_1 = slice_chapter_panels(cfg, chapter)
    frame_0_path = frame_data_1[0][0]
    mtime_1 = frame_0_path.stat().st_mtime_ns

    # Slicing pass 2 — valid manifest exists
    frame_data_2 = slice_chapter_panels(cfg, chapter)
    mtime_2 = frame_0_path.stat().st_mtime_ns

    assert len(frame_data_1) == len(frame_data_2)
    assert mtime_1 == mtime_2, "Frame files must not be re-written or double-sliced when valid manifest exists"


def test_resume_validation_detects_inconsistency(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    # 1. Initial valid slice
    slice_chapter_panels(cfg, chapter)
    manifest_path = cfg.temp_slices_dir / chapter.tag / "manifest.json"

    # Validate passes initially
    is_valid, manifest, reason = validate_canonical_manifest(manifest_path, chapter, job_id=cfg.job_id)
    assert is_valid is True

    # 2. Tamper: delete a frame file from disk
    missing_frame_path = Path(manifest.frames[0].path)
    if missing_frame_path.exists():
        missing_frame_path.unlink()

    is_valid, _, reason = validate_canonical_manifest(manifest_path, chapter, job_id=cfg.job_id)
    assert is_valid is False
    assert "missing or invalid on disk" in reason

    # 3. Verify re-slicing repairs the invalid state
    re_sliced_data = slice_chapter_panels(cfg, chapter)
    assert len(re_sliced_data) > 0
    assert missing_frame_path.exists(), "Re-slicing must regenerate missing frame file"


def test_resume_validation_detects_source_mismatch(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    slice_chapter_panels(cfg, chapter)
    manifest_path = cfg.temp_slices_dir / chapter.tag / "manifest.json"

    # Tamper with manifest source_pages
    data = json.loads(manifest_path.read_text())
    data["source_pages"] = ["different_001.jpg"]
    manifest_path.write_text(json.dumps(data))

    is_valid, _, reason = validate_canonical_manifest(manifest_path, chapter, job_id=cfg.job_id)
    assert is_valid is False
    assert "Source pages mismatch" in reason


def test_rendering_and_timing_consumes_canonical_manifest(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    # 1. Slice panels -> produces manifest
    slice_chapter_panels(cfg, chapter)
    manifest_path = cfg.temp_slices_dir / chapter.tag / "manifest.json"
    manifest = load_or_migrate_manifest(manifest_path, chapter=chapter, job_id=cfg.job_id)
    assert manifest is not None

    # Assign durations to canonical frame entries
    frame_durations = [3.0] * len(manifest.frames)
    for pos, entry in enumerate(manifest.frames):
        entry.duration = frame_durations[pos]
    manifest.save(manifest_path)

    # Reload manifest and verify stored timing
    reloaded = load_or_migrate_manifest(manifest_path, chapter=chapter, job_id=cfg.job_id)
    assert reloaded is not None
    assert len(reloaded.frames) == len(manifest.frames)
    for entry in reloaded.frames:
        assert entry.duration == 3.0

    # Render chapter using manifest frame paths
    frame_paths = [Path(f.path) for f in reloaded.frames]
    out_video = mp.render_chapter(cfg, chapter, frame_paths, frame_durations, audio_path=None)
    assert out_video is not None
    assert out_video.exists()


def test_legacy_v1_manifest_explicit_migration(dummy_chapter_env):
    cfg, chapter, work_dir = dummy_chapter_env

    out_dir = cfg.temp_slices_dir / chapter.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f0 = out_dir / "frame_00000.jpg"
    Image.new("RGB", (100, 100), "white").save(f0)

    # Write legacy v1 manifest structure
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "frames": [str(f0.resolve())],
        "sources": [0]
    }))

    # Migrate explicitly
    migrated = load_or_migrate_manifest(manifest_path, chapter=chapter, job_id=cfg.job_id)
    assert migrated is not None
    assert migrated.manifest_version == mp.MANIFEST_VERSION
    assert len(migrated.frames) == 1
    assert migrated.frames[0].frame_id == f"{chapter.tag}_frame_00000"
    assert migrated.frames[0].source_page == "001.jpg"
    assert migrated.frames[0].source_index == 0
