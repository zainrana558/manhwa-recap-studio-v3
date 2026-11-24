import os
os.environ["SKIP_OCR_INIT"] = "1"

import asyncio
import json
import sys
import pytest

service_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if service_dir not in sys.path:
    sys.path.insert(0, service_dir)

import main


def _run_pipeline(text, slug="nano-machine"):
    main._set_series_slug(slug)
    t = main._repair_and_denoise(main._clean_and_normalize_ocr_text(text))
    return main._apply_known_corrections(t, slug)


def test_clean_dict_excludes_capitalized_names():
    # If wamerican is installed, lowercase common words exist while surnames like kim/han/martel do not
    if main._CLEAN_DICT:
        assert "defending" in main._CLEAN_DICT
        assert "ultimate" in main._CLEAN_DICT
        assert "kim" not in main._CLEAN_DICT
        assert "martel" not in main._CLEAN_DICT


@pytest.mark.parametrize("bad,expected", [
    ("UIGHTNING", "LIGHTNING"),
    ("FIIGHTING", "FIGHTING"),
    ("TRAINNG", "TRAINING"),
    ("ATTACCK", "ATTACK"),
    ("WATCHNG", "WATCHING"),
    ("ULTMATE", "ULTIMATE"),
    ("ULTINATE", "ULTIMATE"),
    ("SUPERVISOK", "SUPERVISOR"),
    ("DELIGHITAUL", "DELIGHTFUL"),
    ("MARTEAL", "MARTIAL"),
    ("SEURET", "SECRET"),
    ("SABERT", "SABER"),
    ("EICHT", "EIGHT"),
    ("BEUEVE", "BELIEVE"),
    ("DLUINE", "DIVINE"),
    ("PAHACE", "PALACE"),
    ("DIMINE", "DIVINE"),
    ("GRBAT", "GREAT"),
    ("GUARDANS", "GUARDIANS"),
])
def test_garble_spell_and_visual_repairs(bad, expected):
    out = _run_pipeline(f"THE {bad} TECHNIQUE")
    assert expected in out.split(), f"{bad} -> {out!r} (expected {expected})"


@pytest.mark.parametrize("glued,expected", [
    ("DEFENDINGTHAT", "DEFENDING THAT"),
    ("SPLITTINGUP", "SPLITTING UP"),
    ("STEALTHMODE", "STEALTH MODE"),
    ("FORMLESSSWORD", "FORMLESS SWORD"),
    ("MURIMALLIANCE", "MURIM ALLIANCE"),
    ("BURNEDWHOLE", "BURNED WHOLE"),
])
def test_desegment_runon_words(glued, expected):
    out = _run_pipeline(f"WE ARE {glued} NOW")
    assert expected in out, f"{glued} -> {out!r}"


def test_digit_as_word_fixes_without_corrupting_numbers():
    assert "SECT IS STILL STANDING" in _run_pipeline("THE SIX MARTIAL ARTS SECT 15 STILL STANDING!")
    assert "LEVEL IS BELOW 19%" in _run_pipeline("WARNING! THE USER'S ENERGY LEVEL 15 BELOW 19%.").upper()
    assert "I CAN'T" in _run_pipeline("1 CAN'T BELIEVE IT")
    assert "GOD" in _run_pipeline("OH MY G0D!")
    # Real numbers must NEVER be altered
    assert "CHAPTER 15" in _run_pipeline("CHAPTER 15 IS OUT")
    assert "LEVEL 15 REACHED" in _run_pipeline("LEVEL 15 REACHED NOW")
    assert "FOR 15 YEARS" in _run_pipeline("HE TRAINED FOR 15 YEARS")


def test_names_and_sfx_preserved():
    for safe in ["SUNBAE", "GONGJA", "JAIHUAN", "SEOLAH", "GARAM", "YEOWOON", "CHEON"]:
        out = _run_pipeline(f"LOOK AT {safe} NOW")
        assert safe in out, f"Protected name {safe} corrupted -> {out!r}"
    for sfx in ["FWOOSH", "KRAKOOM", "AAARGH", "KEKEKE", "GRRR"]:
        out = _run_pipeline(f"{sfx}!")
        assert sfx in out, f"SFX {sfx} corrupted -> {out!r}"


def test_contextual_mlm_feeus_and_conflicts():
    out1 = _run_pipeline("IT FEEUS SO STRANGE TODAY")
    assert "FEELS" in out1, f"Expected FEELS in {out1!r}"
    out_bloom = _run_pipeline("THE CLOOMING LOTUS PETALS FELL")
    assert "BLOOMING" in out_bloom, f"Expected BLOOMING in {out_bloom!r}"
    out_loom = _run_pipeline("A CLOOMING SHADOW APPEARED")
    assert "LOOMING" in out_loom, f"Expected LOOMING in {out_loom!r}"
    out2 = _run_pipeline("HIIM IS MY SWORD")
    assert out2.startswith("THIS IS"), f"Expected THIS in {out2!r}"
    out3 = _run_pipeline("HE USED HIIM OWN POWER")
    assert "USED HIS OWN" in out3, f"Expected HIS in {out3!r}"
    out4 = _run_pipeline("WHAT SIH GOING ON HERE")
    assert "WHAT IS GOING" in out4, f"Expected IS in {out4!r}"


def test_self_learning_sequence_alignment_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_CORR_LEARN_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_CORR_DIRS", [main._CORR_DIRS[0], str(tmp_path)])
    monkeypatch.setattr(main, "_GLOBAL_LEARN_PATH", str(tmp_path / "_global.learned.jsonl"))
    main._corr_cache.clear()

    # Should NOT learn shifted garbage when compound splits on left and right
    main._learn_ocr_correction(
        "POISONOUSENERGY ONOHS HERE",
        "POISONOUS ENERGY HERE",
        "test-series",
        source="tight-unclip",
    )
    lp, xp, _rp = main._learned_paths("test-series")
    rows = main._read_jsonl(lp)
    assert all(r["from"] != "ONOHS" for r in rows), f"Shifted garbage learned: {rows}"

    # Should learn anchored 1-to-1 fix and promote after 2 tight-unclip hits
    for _ in range(2):
        main._learn_ocr_correction(
            "HE TRIED XZQPT HIS SWORD",
            "HE TRIED USING HIS SWORD",
            "test-series",
            source="tight-unclip",
        )
    corr = main._load_corrections("test-series")
    assert corr["tokens"].get("XZQPT") == "USING", f"Failed to promote: {corr}"

    # ContextVar isolation across async tasks
    async def _check_ctx():
        async def task(slug):
            main._set_series_slug(slug)
            await asyncio.sleep(0.01)
            return main._get_series_slug()
        r1, r2 = await asyncio.gather(task("series-a"), task("series-b"))
        assert (r1, r2) == ("series-a", "series-b")

    asyncio.run(_check_ctx())


def test_sort_regions_single_line_y_jitter():
    # Words on a single line whose bounding boxes jitter in y_min by 1-3px
    regs = [
        main._TextRegion("WHAT", 0.99, 29, 52, 79, 125),
        main._TextRegion("IS", 0.99, 132, 54, 78, 186),
        main._TextRegion("HAPPENING", 0.99, 194, 52, 79, 384),
        main._TextRegion("NOW?!", 0.99, 400, 51, 81, 520),
    ]
    ordered = main._sort_regions_reading_order(regs)
    assert [r.text for r in ordered] == ["WHAT", "IS", "HAPPENING", "NOW?!"]


def test_parseq_crop_vision_rescue(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    # Render real stylized/sheared word "SHOULD" where RapidOCR gave garble "CTNOHS"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    img = Image.new("RGB", (540, 90), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((20, 28), "WHAT", fill=(20, 20, 20), font=font)
    d.text((130, 28), "SHOULD", fill=(20, 20, 20), font=font, stroke_width=1, stroke_fill=(80, 80, 80))
    d.text((295, 28), "I DO?", fill=(20, 20, 20), font=font)
    arr = np.array(img)

    fake_regions = [
        main._TextRegion("WHAT", 0.98, 20, 26, 62, 118),
        main._TextRegion("CTNOHS", 0.86, 128, 26, 62, 285),  # garble mimicking spiky font
        main._TextRegion("I DO?", 0.98, 292, 26, 62, 395),
    ]

    monkeypatch.setattr(main, "_CORR_LEARN_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_CORR_DIRS", [main._CORR_DIRS[0], str(tmp_path)])
    monkeypatch.setattr(main, "_GLOBAL_LEARN_PATH", str(tmp_path / "_global.learned.jsonl"))
    main._corr_cache.clear()
    main._set_series_slug("test-parseq-series")

    # Exercise _recover_regions_with_parseq directly on pixel boxes
    pq = main._recover_regions_with_parseq(arr, fake_regions)
    assert pq is not None, "PARSeq failed to recover garbled box"
    text, conf, count, improved = pq
    assert improved == 1
    assert "SHOULD" in text, f"Expected SHOULD from pixel pixels -> {text!r}"
    assert "WHAT" in text and "DO?" in text

    # Also verify _ocr_with_cascade integrates it and logs parseq-crop into learned.jsonl
    monkeypatch.setattr(main, "rapidocr_engine", object())
    monkeypatch.setattr(main, "_run_rapidocr_on_image", lambda *a, **kw: fake_regions)
    out_text, out_conf, out_cnt, status, q, cands, reason = main._ocr_with_cascade(arr)
    assert status == "SUCCESS"
    assert "SHOULD" in out_text
    assert any(c.get("provider") == "parseq" for c in cands), cands
    lp, _xp, _rp = main._learned_paths("test-parseq-series")
    rows = main._read_jsonl(lp)
    assert any(r["from"] == "CTNOHS" and r["to"] == "SHOULD" and "parseq-crop" in r.get("src", {}) for r in rows), rows


def test_noise_only_panels_dropped():
    for junk in ("0", "1", "?!", "ER", "TE", "I!", "$A", "034-4532-5893"):
        assert main._looks_like_ocr_noise(junk) is True, f"Expected noise for {junk!r}"
        merged, conf, cnt = main._merge_regions([main._TextRegion(junk, 0.90, 10, 10, 40, 60)])
        assert (merged, cnt) == ("", 0), f"Noise region {junk!r} leaked: {(merged, cnt)}"

    # Real 2-letter words must still pass
    for real_word in ("NO", "OK", "HI", "AH", "OH"):
        assert main._looks_like_ocr_noise(real_word) is False


def test_hyphen_line_wrap_and_margin_footnote_strip():
    assert main._clean_and_normalize_ocr_text("MAR- TIAL ARTS ANCES- TOR") == "MARTIAL ARTS ANCESTOR"
    assert main._clean_and_normalize_ocr_text("BLOOD- LINE") == "BLOODLINE"
    cleaned = main._clean_and_normalize_ocr_text(
        "HE IS ALREADY DEAD. TN: WAIGONG REFERS TO EXTERNAL MARTIAL ARTS"
    )
    assert cleaned == "HE IS ALREADY DEAD."


def test_dot_chains_glyphs_and_period_glued_repairs():
    assert "Its as if-" in main._clean_and_normalize_ocr_text("Its.as.if-")
    assert main._clean_and_normalize_ocr_text("NO.HE'S STRONG") == "NO. HE'S STRONG"
    res = main._clean_and_normalize_ocr_text("drΩT02TM mxo.nx.axexaxxaxo WAIT!")
    assert "WAIT!" in res and "mxo.nx" not in res and "Ω" not in res


def test_reading_order_separates_margin_column_from_bubble():
    # Speech bubble (right column x=220..420) vs left margin footnote column (x=10..90)
    regs = [
        main._TextRegion("AL-", 0.95, 220, 40, 70, 320),
        main._TextRegion("TN: EXTERNAL", 0.92, 10, 55, 80, 100),
        main._TextRegion("READY?!", 0.95, 220, 82, 112, 340),
    ]
    ordered = main._sort_regions_reading_order(regs)
    texts = [r.text for r in ordered]
    # Margin TN: and bubble AL- / READY?! must NOT interleave AL- -> TN: -> READY?!
    assert texts.index("AL-") + 1 == texts.index("READY?!"), f"Interleaved columns: {texts}"


def test_promo_and_scanlator_credits_filtered_signal(monkeypatch):
    import numpy as np

    promo = (
        "THIS MANHWA CHAPTER ENDS ON NOVEL CHAPTER 340 - "
        "FINAL CHAPTER (END) OF FUTURE (REVISED) "
        "CONTINUE READING THE NOVEL AT ASURASCANS.COM/EBOOKS"
    )
    assert main._PROMO_BANNER_RE.search(promo)
    assert main._apply_known_corrections(promo, "nano-machine") == ""
    assert main._apply_known_corrections("Read at asurascans.com for the fastest releases", "nano-machine") == ""
    assert main._apply_known_corrections("·Red ice Studi", "nano-machine") == ""

    regs = [
        main._TextRegion("READ AT", 0.99, 50, 40, 90, 260),
        main._TextRegion("ASURASCANS.COM FOR THE FASTEST RELEASES", 0.99, 30, 100, 150, 480),
    ]
    assert main._raw_regions_are_credits(regs) is True

    monkeypatch.setattr(main, "rapidocr_engine", object())
    monkeypatch.setattr(main, "_run_rapidocr_on_image", lambda *a, **kw: regs)
    monkeypatch.setattr(main, "_run_ocr_on_image", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not fall back")))
    img = np.full((200, 500, 3), 240, dtype=np.uint8)
    text, conf, cnt, status, q, cands, reason = main._ocr_with_cascade(img)
    assert text == "" and cnt == 0 and status == "NO_TEXT" and reason == "credits_filtered"


def test_exemplar_rejects_semi_garble_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_CORR_LEARN_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_CORR_DIRS", [main._CORR_DIRS[0], str(tmp_path)])
    monkeypatch.setattr(main, "_GLOBAL_LEARN_PATH", str(tmp_path / "_global.learned.jsonl"))
    main._corr_cache.clear()
    main._learn_ocr_correction("BAD GARBLE", "HBLP ME QXKZ", "test-exemplar-gate", source="got-ocr2")
    _lp, xp, _rp = main._learned_paths("test-exemplar-gate")
    rows = main._read_jsonl(xp)
    assert not any(r.get("to") == "HBLP ME QXKZ" for r in rows), rows


def test_noise_or_sfx_only_reason_distinguished_from_empty_panel(monkeypatch):
    import numpy as np

    img = np.full((240, 600, 3), 245, dtype=np.uint8)
    monkeypatch.setattr(main, "rapidocr_engine", object())

    # 1. Detector saw raw glyph ("0" from Hangul circle) that filtered out as noise
    monkeypatch.setattr(main, "_run_rapidocr_on_image", lambda *a, **kw: [main._TextRegion("0", 0.91, 20, 20, 60, 60)])
    t, c, n, st, q, cands, rsn = main._ocr_with_cascade(img)
    assert (t, n, st) == ("", 0, "NO_TEXT")
    assert rsn.startswith("noise_or_sfx_only:"), rsn

    # 2. Detector found literally zero regions
    monkeypatch.setattr(main, "_run_rapidocr_on_image", lambda *a, **kw: [])
    t2, c2, n2, st2, q2, cands2, rsn2 = main._ocr_with_cascade(img)
    assert (t2, n2) == ("", 0)
    assert rsn2.startswith("skipped_fallback_zero_regions_detected:"), rsn2


def test_mixed_dialogue_with_noise_and_credit_precedence(monkeypatch):
    import numpy as np

    img = np.full((240, 600, 3), 245, dtype=np.uint8)
    monkeypatch.setattr(main, "rapidocr_engine", object())

    # Mixed dialogue + Hangul circle noise ("0"): dialogue MUST survive cleanly as SUCCESS
    monkeypatch.setattr(main, "_run_rapidocr_on_image", lambda *a, **kw: [
        main._TextRegion("STAND BACK!", 0.96, 40, 20, 70, 340),
        main._TextRegion("0", 0.88, 400, 120, 155, 435),
    ])
    t, c, n, st, q, cands, rsn = main._ocr_with_cascade(img)
    assert t == "STAND BACK!" and n == 1 and st == "SUCCESS"
    assert not rsn.startswith("noise_or_sfx_only:")

    # Credits card MUST take precedence over noise_or_sfx_only
    monkeypatch.setattr(main, "_run_rapidocr_on_image", lambda *a, **kw: [
        main._TextRegion("READ AT ASURASCANS.COM FOR THE FASTEST RELEASES", 0.98, 20, 20, 80, 560),
    ])
    t2, c2, n2, st2, q2, cands2, rsn2 = main._ocr_with_cascade(img)
    assert t2 == "" and n2 == 0 and st2 == "NO_TEXT" and rsn2 == "credits_filtered"


def test_solo_leveling_mgeko_webtoon_and_solo_function_word(monkeypatch):
    import numpy as np

    # 1. mgeko.cc & D&C WEBTOON / REDICE promo cards are filtered cleanly
    for txt in (
        "UPDATES New COMICS Active COMMUNITY",
        "read at mgeko.cc fastest updates",
        "WEBTOON",
        "WEBTOON 71813248SHAPTREDICEINC-MA",
    ):
        assert main._apply_known_corrections(txt, "solo-leveling") == "", f"Promo leaked: {txt!r}"

    # 2. Solo bare function word hallucination ("to", "the") -> "", while dialogue & "NO!" stay intact
    assert main._merge_regions([main._TextRegion("to", 0.90, 20, 20, 45, 50)]) == ("", 0.0, 0)
    assert main._merge_regions([main._TextRegion("the", 0.90, 20, 20, 45, 50)]) == ("", 0.0, 0)
    merged_real, _, cnt_real = main._merge_regions([
        main._TextRegion("I NEED", 0.95, 20, 20, 45, 90),
        main._TextRegion("TO GO!", 0.95, 95, 20, 45, 160),
    ])
    assert merged_real == "I NEED TO GO!" and cnt_real == 2
