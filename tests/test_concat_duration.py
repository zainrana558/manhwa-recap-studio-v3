import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from pipeline.master_pipeline import render_chapter, PipelineConfig, Chapter

class TestConcatDuration(unittest.TestCase):
    def test_render_chapter_concat_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "work"
            input_dir = Path(tmpdir) / "dataset"
            output_path = Path(tmpdir) / "output.mp4"

            cfg = PipelineConfig(
                input_dir=input_dir,
                output_path=output_path,
                total_chapters=1,
                bgm_path=None,
                work_dir=work_dir,
            )
            cfg.ensure_dirs()

            # Create dummy frame images
            img_dir = work_dir / "temp_slices" / "chap_001"
            img_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            for i in range(3):
                fp = img_dir / f"frame_{i:05d}.jpg"
                Image.new("RGB", (1920, 1080), "black").save(fp)
                frame_paths.append(fp)

            frame_durations = [3.0, 4.0, 5.0]  # Expected sum = 12.0s
            chapter = Chapter(index=1, name="chapter_001", folder=input_dir / "chapter_001")

            out_video = render_chapter(cfg, chapter, frame_paths, frame_durations, audio_path=None)
            self.assertIsNotNone(out_video)
            self.assertTrue(out_video.exists())

            import subprocess
            res = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(out_video)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            actual_dur = float(res.stdout.strip())

            # Verify the duration matches within 3.0 seconds (video QA tolerance is 5.0s)
            expected_dur = sum(frame_durations)
            self.assertAlmostEqual(actual_dur, expected_dur, delta=3.0)

if __name__ == "__main__":
    unittest.main()
