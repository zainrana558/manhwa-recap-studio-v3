import importlib.util
import sys
from pathlib import Path

# Load parse_ocr_results dynamically from mini-services/paddleocr-service/main.py
main_path = Path(__file__).parent.parent / "mini-services" / "paddleocr-service" / "main.py"
spec = importlib.util.spec_from_file_location("paddleocr_main", main_path)
paddleocr_main = importlib.util.module_from_spec(spec)
sys.modules["paddleocr_main"] = paddleocr_main
spec.loader.exec_module(paddleocr_main)

parse_ocr_results = paddleocr_main.parse_ocr_results


class MockPaddle3OCRResult:
    """Mock PaddleOCR 3.x OCRResult object."""
    def __init__(self, texts, scores, boxes):
        self.rec_texts = texts
        self.rec_scores = scores
        self.rec_boxes = boxes


def test_parse_ocr_results_paddleocr3():
    # Test PaddleOCR 3.x OCRResult dataclass / object format
    mock_result = MockPaddle3OCRResult(
        texts=["System Alert", "Quest Completed!"],
        scores=[0.98, 0.95],
        boxes=[[[10, 10], [100, 10], [100, 30], [10, 30]], [[10, 40], [150, 40], [150, 60], [10, 60]]]
    )
    parsed = parse_ocr_results([mock_result])

    assert len(parsed) == 2
    assert parsed[0]["text"] == "System Alert"
    assert parsed[0]["confidence"] == 0.98
    assert parsed[0]["box"] == [[10, 10], [100, 10], [100, 30], [10, 30]]

    assert parsed[1]["text"] == "Quest Completed!"
    assert parsed[1]["confidence"] == 0.95
    assert parsed[1]["box"] == [[10, 40], [150, 40], [150, 60], [10, 60]]


def test_parse_ocr_results_legacy():
    # Test Legacy tuple/list fallback: [[[box], (text, score)]]
    legacy_result = [
        [
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("Hello World", 0.92)],
            [[[10, 40], [120, 40], [120, 60], [10, 60]], ("Next Line", 0.88)]
        ]
    ]
    parsed = parse_ocr_results(legacy_result)

    assert len(parsed) == 2
    assert parsed[0]["text"] == "Hello World"
    assert parsed[0]["confidence"] == 0.92
    assert parsed[0]["box"] == [[10, 10], [100, 10], [100, 30], [10, 30]]

    assert parsed[1]["text"] == "Next Line"
    assert parsed[1]["confidence"] == 0.88
    assert parsed[1]["box"] == [[10, 40], [120, 40], [120, 60], [10, 60]]


def test_parse_ocr_results_empty():
    assert parse_ocr_results([]) == []
    assert parse_ocr_results(None) == []
