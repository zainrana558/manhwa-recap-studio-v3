import sys
import json
from typing import List, Dict, Any

# Import internal functions from PaddleOCR service and production pipeline
sys.path.append("mini-services/paddleocr-service")
sys.path.append("pipeline")

# Mock cv2 and numpy if not installed in benchmark env
class _DummyCv2:
    def cvtColor(self, *args, **kwargs): return args[0]
    def Canny(self, *args, **kwargs): return args[0]
    def findContours(self, *args, **kwargs): return [], None
    def approxPolyDP(self, *args, **kwargs): return []
    def arcLength(self, *args, **kwargs): return 0.0
    def contourArea(self, *args, **kwargs): return 0.0
sys.modules.setdefault("cv2", _DummyCv2())

import main
from main import (
    _TextRegion,
    _is_slash_or_math_artifact,
    _symbol_ratio_exceeded,
    _is_graphic_logo,
    _clean_and_normalize_ocr_text,
    _merge_regions,
    CONFIDENCE_CUTOFF,
)
from production import clean_ocr_text_with_sfx


def run_benchmark():
    test_cases = [
        {
            "id": "speed_lines_and_lens_flares",
            "raw_candidates": [
                (_TextRegion("///\\/\\//", 0.85, 10, 10, 30, 100), "Slash & Math Filter"),
                (_TextRegion("*1C5", 0.72, 50, 50, 70, 120), "Slash & Math Filter"),
                (_TextRegion("—", 0.90, 80, 80, 95, 150), "Slash & Math Filter"),
                (_TextRegion("Background Art Flare", 0.54, 20, 20, 40, 200), "Confidence Score Cutoff"),
            ],
            "expected_text": ""
        },
        {
            "id": "high_density_quest_board",
            "raw_candidates": [
                (_TextRegion("[QUEST NOTIFICATION]", 0.92, 10, 10, 30, 200), None),
                (_TextRegion("1. Push-ups: 100/100", 0.88, 10, 40, 60, 250), None),
                (_TextRegion("2. Sit-ups: 100/100", 0.89, 10, 70, 90, 250), None),
                (_TextRegion("3. Squats: 100/100", 0.91, 10, 100, 120, 250), None),
                (_TextRegion("4. Run: 10km", 0.90, 10, 130, 150, 200), None),
            ],
            "expected_text": "[QUEST NOTIFICATION] 1. Push-ups: 100/100 2. Sit-ups: 100/100 3. Squats: 100/100 4. Run: 10km"
        },
        {
            "id": "ellipsis_and_mangled_words",
            "raw_candidates": [
                (_TextRegion("A good-curdling scream filled the air dash", 0.95, 10, 10, 30, 300), None),
                (_TextRegion("HO0 HO O! B to be continued...", 0.92, 10, 40, 60, 300), None),
            ],
            "expected_text": "A blood-curdling scream filled the air... HOO HOO! To Be Continued..."
        },
        {
            "id": "graphic_logo_title_card",
            "raw_candidates": [
                (_TextRegion("Souls Lacing", 0.65, 50, 50, 200, 500), "Graphic Logo Exclusion"),
            ],
            "expected_text": ""
        }
    ]

    print("================================================================")
    print("           OCR FILTERING & POST-PROCESSING BENCHMARK            ")
    print("================================================================\n")

    summary_log = []

    for test in test_cases:
        test_id = test["id"]
        filtered_regions = []
        filtered_reasons = []

        for region, expected_reason in test["raw_candidates"]:
            # Evaluate filters
            if region.confidence < CONFIDENCE_CUTOFF:
                filtered_reasons.append((region.text, f"Low Confidence (<{CONFIDENCE_CUTOFF})"))
                continue
            if _is_slash_or_math_artifact(region.text):
                filtered_reasons.append((region.text, "Slash & Math Artifact Filter"))
                continue
            if _symbol_ratio_exceeded(region.text):
                filtered_reasons.append((region.text, "Symbol Ratio Limit (>0.35)"))
                continue
            if _is_graphic_logo(region, img_h=1000, img_w=1000):
                filtered_reasons.append((region.text, "Graphic Logo Exclusion"))
                continue

            filtered_regions.append(region)

        merged_text, avg_conf, count = _merge_regions(filtered_regions, is_ui_box=(test_id == "high_density_quest_board"))
        final_tts_script = clean_ocr_text_with_sfx(merged_text)

        print(f"--- Test Case: {test_id} ---")
        print(f"Filtered Garbage Strings:")
        if filtered_reasons:
            for text_str, reason in filtered_reasons:
                print(f"  [X] Dropped '{text_str}' -> Reason: {reason}")
        else:
            print("  (None dropped)")

        print(f"Parsed Text Output:")
        print(f"  Raw Merged:  '{merged_text}'")
        print(f"  Final TTS:   '{final_tts_script}'")
        print(f"  Expected:    '{test['expected_text']}'")
        print(f"  Match:       {'PASSED' if final_tts_script == test['expected_text'] else 'FAILED'}\n")

        summary_log.append({
            "test_id": test_id,
            "filtered": filtered_reasons,
            "final_tts": final_tts_script,
            "passed": final_tts_script == test["expected_text"]
        })

    print("================================================================")
    print("                      BENCHMARK COMPLETE                        ")
    print("================================================================")
    return summary_log


if __name__ == "__main__":
    run_benchmark()
