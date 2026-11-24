"""Regression tests for all defects uncovered during the 338-chapter Nano Machine run."""

import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath("."))
import pipeline.master_pipeline as mp


def test_kokoro_worker_pool_reuses_bounded_workers_and_shuts_down(monkeypatch, tmp_path):
    spawned = []

    class FakeStdin:
        def write(self, data):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    class FakeStdout:
        def __init__(self, out_ref):
            self._out_ref = out_ref
            self._sent_ready = False

        def readline(self):
            if not self._sent_ready:
                self._sent_ready = True
                return json.dumps({"ready": True}) + "\n"
            out = self._out_ref[0]
            if out is None:
                return ""
            Path(out).write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 256)
            self._out_ref[0] = None
            return json.dumps({"ok": True, "sentences": [{"text": "Hello world", "start": 0.0, "end": 0.8}]}) + "\n"

    class FakeProc:
        def __init__(self):
            self.pid = 90000 + len(spawned)
            self._pending = [None]
            self.stdin = self
            self.stdout = FakeStdout(self._pending)
            self._rc = None
            spawned.append(self)

        def write(self, payload):
            req = json.loads(payload)
            self._pending[0] = req["out_wav"]

        def flush(self):
            pass

        def close(self):
            pass

        def poll(self):
            return self._rc

        def wait(self, timeout=None):
            self._rc = 0
            return 0

        def terminate(self):
            self._rc = 0

        def kill(self):
            self._rc = -9

    mp._shutdown_kokoro_pool()
    fake_script = tmp_path / "kokoro_tts.py"
    fake_script.write_text("# mock")
    fake_model_dir = tmp_path / "kokoro_models"
    fake_model_dir.mkdir()
    (fake_model_dir / "kokoro-v1.0.onnx").write_bytes(b"onnx")
    (fake_model_dir / "voices-v1.0.bin").write_bytes(b"bin")
    monkeypatch.setenv("RECAP_KOKORO_WORKERS", "2")
    monkeypatch.setattr(mp, "_kokoro_python", lambda: Path("/usr/bin/python3"))
    monkeypatch.setattr(mp, "KOKORO_TTS_SCRIPT", fake_script)
    monkeypatch.setattr(mp, "KOKORO_MODEL_DIR", fake_model_dir)
    monkeypatch.setattr(mp.subprocess, "Popen", lambda *a, **kw: FakeProc())

    # Simulate 3 sequential chapters each opening a fresh ThreadPoolExecutor of 4 threads
    for ch in range(3):
        def _job(idx):
            out = tmp_path / f"ch{ch}_seg{idx}.wav"
            res = mp._synthesize_with_kokoro("Hello world", out, voice="af_heart")
            return res is not None

        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(_job, range(6)))
        assert all(results)

    # Across all 3 chapters x 4 threads, total spawned workers must stay <= RECAP_KOKORO_WORKERS (2)
    assert len(spawned) <= 2, f"Expected <=2 pooled workers, got {len(spawned)}"
    mp._shutdown_kokoro_pool()
    assert mp._kokoro_pool_singleton is None and all(p.poll() == 0 for p in spawned)


def test_promo_banner_and_credits_filtered_dropped_while_chibi_art_kept(tmp_path):
    # Verify _clean_source_text & _PROMO_CARD_RE handle every leaked watermark/promo form
    assert mp._clean_source_text(
        "THIS MANHWA CHAPTER ENDS ON NOVEL CHAPTER 340 - FINAL CHAPTER (END) "
        "CONTINUE READING THE NOVEL AT ASURASCANS.COM/EBOOKS"
    ) == ""
    assert mp._clean_source_text("·Red ice Studi") == ""
    assert mp._clean_source_text("lans.co") == ""
    assert mp._clean_source_text("reapscans.co.") == ""
    assert mp._clean_source_text("WE WON! TN: WAIGONG REFERS TO EXTERNAL QI") == "WE WON!"

    # Create 5 chapter frames:
    # 0, 1: normal dialogue
    # 2: genuine silent chibi line-art panel (white paper + crisp strokes, NO credits_filtered)
    # 3: white novel-promo end card
    # 4: purple ASURASCANS.COM end card whose OCR returned "" with credits_filtered=True
    chap_dir = tmp_path / "chapter_001"
    chap_dir.mkdir()
    frames_dir = tmp_path / "slices"
    frames_dir.mkdir()

    frame_paths = []
    for i in range(5):
        fp = frames_dir / f"frame_{i:05d}.jpg"
        img = Image.new("RGB", (600, 400), (255, 255, 255) if i != 4 else (90, 40, 140))
        if i == 2:
            d = ImageDraw.Draw(img)
            d.ellipse([220, 120, 380, 280], outline=(10, 10, 10), width=4)
        img.save(fp)
        frame_paths.append(fp)

    narrations = {
        "frame_00000.jpg": {"image": "frame_00000.jpg", "text": "WE MUST ATTACK!", "status": "SUCCESS"},
        "frame_00001.jpg": {"image": "frame_00001.jpg", "text": "STAND BACK!", "status": "SUCCESS"},
        "frame_00002.jpg": {"image": "frame_00002.jpg", "text": "", "status": "NO_TEXT"},
        "frame_00003.jpg": {
            "image": "frame_00003.jpg",
            "text": "THIS MANHWA CHAPTER ENDS ON NOVEL CHAPTER 340 ASURASCANS.COM/EBOOKS",
            "status": "SUCCESS",
        },
        "frame_00004.jpg": {
            "image": "frame_00004.jpg",
            "text": "",
            "status": "NO_TEXT",
            "credits_filtered": True,
        },
    }

    # Replicate the exact _definite_promo_or_credit + _boundary_credits contract from master_pipeline
    # and also test via actual chapter preparation in _ocr_frames_into_narration_json
    def _get_ne(i):
        return narrations[frame_paths[i].name]

    assert not _get_ne(2).get("credits_filtered")
    assert _get_ne(4).get("credits_filtered") is True
    assert mp._PROMO_CARD_RE.search(_get_ne(3)["text"])

    # Mock /ocr/batch and verify _ocr_frames_into_narration_json persists credits_filtered=True
    chap = mp.Chapter(index=1, name="chapter_001", folder=chap_dir, panel_paths=[], image_narrations={})
    entries = [
        mp.FrameEntry(frame_id=f"chap_001_frame_{i:05d}", filename=frame_paths[i].name,
                      source_page="001.jpg", source_index=0, path=str(frame_paths[i]))
        for i in range(5)
    ]
    mock_resp = json.dumps({
        "results": [
            {"index": 0, "text": "WE MUST ATTACK!", "status": "SUCCESS", "selection_reason": "accepted"},
            {"index": 1, "text": "STAND BACK!", "status": "SUCCESS", "selection_reason": "accepted"},
            {"index": 2, "text": "", "status": "NO_TEXT", "selection_reason": "skipped_fallback_zero_regions_detected"},
            {"index": 3, "text": narrations["frame_00003.jpg"]["text"], "status": "SUCCESS", "selection_reason": "accepted"},
            {"index": 4, "text": "", "status": "NO_TEXT", "selection_reason": "credits_filtered"},
        ]
    }).encode()

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return mock_resp

    cfg = mp.PipelineConfig(input_dir=tmp_path, output_path=tmp_path / "out.mp4", work_dir=tmp_path / "w")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
        ok = mp._ocr_frames_into_narration_json(cfg, chap, entries)
    assert ok is True
    saved = json.loads((chap_dir / "narration.json").read_text())
    assert saved[2].get("credits_filtered") is None
    assert saved[3].get("credits_filtered") is True and saved[3]["text"] == ""
    assert saved[4].get("credits_filtered") is True and saved[4]["text"] == ""


def test_run_slice_only_monotonic_progress(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    for ch in range(1, 6):
        d = ds / f"chapter_{ch:03d}"
        d.mkdir(parents=True)
        img = Image.new("RGB", (400, 400), (20, 20, 20))
        img.save(d / "001.jpg")
    prog_file = tmp_path / "progress.json"
    cfg = mp.PipelineConfig(
        input_dir=ds, output_path=tmp_path / "out.mp4",
        work_dir=tmp_path / "work", progress_file=prog_file, slice_only=True,
    )

    seen_units = []
    orig_write = cfg.write_progress

    def _rec(stage, ch_idx, total, msg, frac=0.0, substage=None):
        if stage == "slice":
            seen_units.append(ch_idx + max(0.0, min(1.0, frac)))
        return orig_write(stage, ch_idx, total, msg, frac=frac, substage=substage)

    monkeypatch.setattr(cfg, "write_progress", _rec)
    monkeypatch.setenv("RECAP_FRAME_MODE", "page")

    def _fake_frame(c, chapter, out_dir):
        fp = out_dir / "frame_00000.jpg"
        Image.new("RGB", (400, 400), (30, 30, 30)).save(fp)
        return [mp.FrameEntry(frame_id=f"{chapter.tag}_frame_00000", filename="frame_00000.jpg",
                              source_page="001.jpg", source_index=0, path=str(fp))]

    def _fake_ocr(c, chapter, fe, phase_cb=None):
        time.sleep(0.03)
        if phase_cb:
            phase_cb("ocr", 0.7)
        (chapter.folder / "narration.json").write_text("[]")
        return True

    monkeypatch.setattr(mp, "_frame_pages_reference_style", _fake_frame)
    monkeypatch.setattr(mp, "_ocr_frames_into_narration_json", _fake_ocr)

    mp.run_slice_only(cfg)
    assert len(seen_units) >= 5
    for a, b in zip(seen_units, seen_units[1:]):
        assert b >= a - 1e-9, f"Slice progress regressed backwards: {a} -> {b} in {seen_units}"


def test_filler_slice_and_sfx_filter_guards(tmp_path):
    import cv2

    # 1. Skinny vertical sliver (110x900 out of 800px column) with no text box -> dropped
    sliver = np.full((900, 110, 3), 220, dtype=np.uint8)
    sliver[200:700, 30:80] = 60
    sliver_g = cv2.cvtColor(sliver, cv2.COLOR_RGB2GRAY)
    assert mp._is_filler_slice_crop(sliver, sliver_g, orig_w=800, text_boxes=None) is True
    # Overlapping text box protects even a narrow slice
    assert mp._is_filler_slice_crop(sliver, sliver_g, orig_w=800, text_boxes=[(10, 20, 90, 70, 0.9)]) is False

    # 2. Wide colorful establishing panel (780x500) -> preserved even without text
    wide = np.zeros((500, 780, 3), dtype=np.uint8)
    wide[:, :, 0] = 40
    wide[:, :, 2] = 160
    wide[120:380, 180:600] = (120, 180, 210)
    wide_g = cv2.cvtColor(wide, cv2.COLOR_RGB2GRAY)
    assert mp._is_filler_slice_crop(wide, wide_g, orig_w=800, text_boxes=None) is False

    # 3. _is_empty_sfx_filler geometry & sparse white Hangul-SFX checks
    blank_fp = tmp_path / "sfx_white.jpg"
    Image.new("RGB", (600, 400), (248, 248, 248)).save(blank_fp)
    dark_fp = tmp_path / "sword_action.jpg"
    dark_arr = np.full((400, 600, 3), 25, dtype=np.uint8)
    dark_arr[50:350, 100:500] = (30, 80, 160)
    Image.fromarray(dark_arr).save(dark_fp)

    fe_skinny = mp.FrameEntry("c_f0", "frame_00000.jpg", str(blank_fp), "001.jpg", 0, source_box=[0, 0, 140, 700])
    fe_short = mp.FrameEntry("c_f1", "frame_00001.jpg", str(blank_fp), "001.jpg", 0, source_box=[0, 0, 700, 120])
    fe_normal = mp.FrameEntry("c_f2", "frame_00002.jpg", str(blank_fp), "001.jpg", 0, source_box=[0, 0, 750, 650])

    assert mp._is_empty_sfx_filler(blank_fp, fe_skinny, {"text": "", "status": "NO_TEXT"}) is True
    assert mp._is_empty_sfx_filler(blank_fp, fe_short, {"text": "", "status": "NO_TEXT"}) is True
    assert mp._is_empty_sfx_filler(blank_fp, fe_normal, {"text": "", "status": "NO_TEXT", "sfx_or_noise": True}) is True
    # Colorful action art panel with sfx_or_noise must NOT be dropped
    assert mp._is_empty_sfx_filler(dark_fp, fe_normal, {"text": "", "status": "NO_TEXT", "sfx_or_noise": True}) is False
    # Dialogue always protects
    assert mp._is_empty_sfx_filler(blank_fp, fe_skinny, {"text": "ATTACK!", "status": "SUCCESS"}) is False


def test_cross_chapter_dedup_dhash_and_bounded_window():
    img_a = np.zeros((120, 200, 3), dtype=np.uint8)
    img_a[20:100, 40:160] = (200, 120, 40)
    hx = mp._dhash_from_array(img_a)
    assert hx is not None and len(hx) == 128

    fe = mp.FrameEntry(
        frame_id="chap_001_frame_00000",
        filename="frame_00000.jpg",
        path="/tmp/frame_00000.jpg",
        source_page="001.jpg",
        source_index=0,
        dhash=hx,
    )
    reloaded = mp.FrameEntry.from_dict(fe.to_dict())
    assert reloaded.dhash == hx

    mp._RECAP_SEEN_HASHES.clear()
    mp._RECAP_SEEN_SET.clear()
    mp._record_seen_hashes([111, 222, 333])
    assert 222 in mp._RECAP_SEEN_SET and len(mp._RECAP_SEEN_HASHES) == 3
    mp._RECAP_SEEN_HASHES.clear()
    mp._RECAP_SEEN_SET.clear()


def test_edge_cases_slivers_eviction_floors_and_env_toggles(monkeypatch, tmp_path):
    import cv2

    # A. Narrow vertical speech-bubble column (175x650) with char blobs when text_boxes=None -> PROTECTED
    bubble_col = np.full((650, 175, 3), 252, dtype=np.uint8)
    for r in range(8):
        y = 60 + r * 35
        for c in range(4):
            x = 25 + c * 28
            cv2.rectangle(bubble_col, (x, y), (x + 14, y + 18), (15, 15, 15), -1)
    bubble_g = cv2.cvtColor(bubble_col, cv2.COLOR_RGB2GRAY)
    assert mp._is_filler_slice_crop(bubble_col, bubble_g, orig_w=800, text_boxes=None) is False

    # B. RECAP_DROP_FILLER_SLICES=0 disables sliver dropping
    sliver = np.full((900, 110, 3), 210, dtype=np.uint8)
    sliver[200:700, 45:65] = 40
    sliver_g = cv2.cvtColor(sliver, cv2.COLOR_RGB2GRAY)
    monkeypatch.setenv("RECAP_DROP_FILLER_SLICES", "0")
    assert mp._is_filler_slice_crop(sliver, sliver_g, orig_w=800, text_boxes=None) is False
    monkeypatch.setenv("RECAP_DROP_FILLER_SLICES", "1")
    assert mp._is_filler_slice_crop(sliver, sliver_g, orig_w=800, text_boxes=None) is True

    # C. _is_empty_sfx_filler edge inputs: float box, degenerate zero-width box, visual caption, scroll_group, env=0
    fp = tmp_path / "f.jpg"
    Image.new("RGB", (600, 400), (248, 248, 248)).save(fp)
    fe_float = mp.FrameEntry("f_flt", "frame_00000.jpg", str(fp), "001.jpg", 0, source_box=[0.0, 0.0, 140.5, 700.5])
    assert mp._is_empty_sfx_filler(fp, fe_float, {"text": "", "status": "NO_TEXT"}) is True
    fe_zero = mp.FrameEntry("f_zero", "frame_00001.jpg", str(fp), "001.jpg", 0, source_box=[100, 50, 100, 300])
    assert mp._is_empty_sfx_filler(fp, fe_zero, {"text": "", "status": "NO_TEXT"}) is False
    fe_scroll = mp.FrameEntry("f_scr", "frame_00002.jpg", str(fp), "001.jpg", 0, scroll_group="g1", source_box=[0, 0, 140, 700])
    assert mp._is_empty_sfx_filler(fp, fe_scroll, {"text": "", "status": "NO_TEXT"}) is False
    assert mp._is_empty_sfx_filler(fp, fe_float, {"text": "", "status": "NO_TEXT", "visual": "A sword swings."}) is False
    monkeypatch.setenv("RECAP_DROP_SFX_FILLER", "0")
    assert mp._is_empty_sfx_filler(fp, fe_float, {"text": "", "status": "NO_TEXT"}) is False
    monkeypatch.setenv("RECAP_DROP_SFX_FILLER", "1")

    # D. Sliding window eviction with duplicate entries preserves active hashes in _RECAP_SEEN_SET
    from collections import deque
    small_q = deque(maxlen=3)
    monkeypatch.setattr(mp, "_RECAP_SEEN_HASHES", small_q)
    monkeypatch.setattr(mp, "_RECAP_SEEN_SET", set())
    mp._record_seen_hashes([10, 20, 10, 30])  # pushes out first 10; deque holds [20, 10, 30]
    assert list(mp._RECAP_SEEN_HASHES) == [20, 10, 30]
    assert 10 in mp._RECAP_SEEN_SET and 20 in mp._RECAP_SEEN_SET and 30 in mp._RECAP_SEEN_SET
    mp._record_seen_hashes([40])  # pushes out 20
    assert 20 not in mp._RECAP_SEEN_SET and 10 in mp._RECAP_SEEN_SET


def test_kokoro_voice_honors_explicit_edge_tts_andrew(monkeypatch, tmp_path):
    monkeypatch.setenv("RECAP_TTS_ENGINE", "kokoro")
    monkeypatch.setenv("RECAP_KOKORO_VOICE", "am_michael")

    # Explicit Edge-TTS voice ("Andrew") must NOT be overridden by Kokoro!
    cfg_andrew = mp.PipelineConfig(input_dir=tmp_path, output_path=tmp_path / "out.mp4", voice="en-US-AndrewMultilingualNeural")
    assert mp._kokoro_voice_requested(cfg_andrew) is None

    # Explicit Kokoro voice ("bm_george") uses Kokoro
    cfg_kok = mp.PipelineConfig(input_dir=tmp_path, output_path=tmp_path / "out.mp4", voice="bm_george")
    assert mp._kokoro_voice_requested(cfg_kok) == "bm_george"

    # Unset / CLI default falls back to RECAP_KOKORO_VOICE when RECAP_TTS_ENGINE=kokoro
    cfg_def = mp.PipelineConfig(input_dir=tmp_path, output_path=tmp_path / "out.mp4")
    assert mp._kokoro_voice_requested(cfg_def) == "am_michael"

    # mgeko / WEBTOON REDICE cards & solo "to" cleaned out
    assert mp._clean_source_text("UPDATES New COMICS Active COMMUNITY") == ""
    assert mp._clean_source_text("WEBTOON 71813248SHAPTREDICEINC-MA") == ""
    assert mp._clean_source_text("WEBTOON") == ""
    assert mp._clean_source_text("to") == ""


def test_solo_leveling_regressions_garble_webtoon_and_trimmed_source_box(tmp_path):
    # 1. Mixed digit+letter non-word garble dropped, legitimate ordinals/units/dialogue kept
    assert mp._clean_source_text("9NIO9") == ""
    assert mp._clean_source_text("to WEBTOON") == ""
    assert mp._clean_source_text("1ST PLACE!") == "1ST PLACE!"
    assert mp._clean_source_text("100KG MAX") == "100KG MAX"
    assert mp._clean_source_text("STOP!") == "STOP!"
    assert "Push-ups 100/100" in mp._clean_source_text("Line 1: Push-ups 100/100")

    # 2. _frame_reconstructed_strip records trimmed source_box width so skinny strips
    # (e.g. 220px wide art inside a 720px strip) are caught by _is_empty_sfx_filler.
    from PIL import Image
    import numpy as np
    strip = np.full((700, 720, 3), 255, dtype=np.uint8)
    # Paint a 220px-wide colorful vertical panel in the middle (x=250..470, y=50..650)
    rng = np.random.default_rng(7)
    strip[50:650, 250:470] = rng.integers(30, 200, size=(600, 220, 3), dtype=np.uint8)
    img_path = tmp_path / "001.jpg"
    Image.fromarray(strip).save(img_path)

    ch = mp.Chapter(index=1, name="ch1", folder=tmp_path, panel_paths=[img_path])
    cfg = mp.PipelineConfig(input_dir=tmp_path, output_path=tmp_path / "out.mp4", work_dir=tmp_path / "work")
    cfg.ensure_dirs()
    out_dir = tmp_path / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = mp._frame_reconstructed_strip(cfg, ch, out_dir)
    assert len(entries) >= 1
    fe = entries[0]
    bw = fe.source_box[2] - fe.source_box[0]
    bh = fe.source_box[3] - fe.source_box[1]
    # _border_trim_amounts trims up to 10% per side (72px each on 720px strip -> 576px);
    # previously source_box stayed untrimmed at 720px.
    assert bw < 600, f"expected horizontally trimmed source_box width < 600, got {bw} (box={fe.source_box})"
    assert bh >= 580
    # And a skinny 240x620 trimmed source_box is caught by _is_empty_sfx_filler
    fe_skinny = mp.FrameEntry("c_f_sk", fe.filename, fe.path, fe.source_page, fe.source_index, source_box=[240, 40, 480, 660])
    assert mp._is_empty_sfx_filler(Path(fe.path), fe_skinny, {"text": ""}) is True
