from pathlib import Path
from PIL import Image, ImageDraw
from pipeline.master_pipeline import (
    PipelineConfig,
    Chapter,
    slice_chapter_panels,
    discover_chapters,
)


def test_slice_chapter_panels_corrupt_image_graceful_skip(tmp_path: Path):
    work_dir = tmp_path / "work"
    output_path = tmp_path / "output" / "out.mp4"

    cfg = PipelineConfig(
        input_dir=tmp_path / "dataset",
        output_path=output_path,
        work_dir=work_dir,
    )
    cfg.ensure_dirs()

    # Create 1 valid non-blank image and 1 corrupt image
    chap_dir = tmp_path / "dataset" / "chapter_001"
    chap_dir.mkdir(parents=True)

    valid_img = chap_dir / "001.jpg"
    img = Image.new("RGB", (400, 400), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 350], fill="black")
    draw.text((100, 100), "Panel Dialogue Text Here", fill="white")
    img.save(valid_img)

    corrupt_img = chap_dir / "002.jpg"
    corrupt_img.write_bytes(b"CORRUPT_BYTES")

    chapter = Chapter(
        index=1,
        name="chapter_001",
        folder=chap_dir,
        panel_paths=[valid_img, corrupt_img],
    )

    # slice_chapter_panels should skip corrupt_img and return sliced frame for valid_img
    frame_data = slice_chapter_panels(cfg, chapter)
    assert len(frame_data) >= 1
    assert all(fp.exists() for fp, _ in frame_data)


def test_discover_chapters_handles_empty_folders(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    empty_chap = dataset_dir / "chapter_001"
    empty_chap.mkdir()

    cfg = PipelineConfig(
        input_dir=dataset_dir,
        output_path=tmp_path / "out.mp4",
        work_dir=tmp_path / "work",
    )

    chapters = discover_chapters(cfg)
    assert chapters == []
