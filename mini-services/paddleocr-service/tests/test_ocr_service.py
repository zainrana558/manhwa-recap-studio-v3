import base64
import io
import os
import sys
import threading
import time
import pytest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Ensure paddleocr-service directory is in sys.path
service_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if service_dir not in sys.path:
    sys.path.insert(0, service_dir)

import main
from main import app, ServiceState


@pytest.fixture
def test_client():
    return TestClient(app)


def create_dummy_base64_image():
    img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_readiness_and_health_states(test_client, monkeypatch):
    """Test health and readiness endpoints across INITIALIZING, FAILED, and READY states."""
    # State 1: INITIALIZING
    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.INITIALIZING)
    monkeypatch.setattr(main, "MODEL_READY", False)
    monkeypatch.setattr(main, "INIT_ERROR", None)

    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "initializing"
    assert data["ready"] is False
    assert data["state"] == ServiceState.INITIALIZING

    resp_ready = test_client.get("/ready")
    assert resp_ready.status_code == 503
    assert resp_ready.json()["detail"]["status"] == "not_ready"
    assert resp_ready.json()["detail"]["state"] == ServiceState.INITIALIZING

    # State 2: FAILED
    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.FAILED)
    monkeypatch.setattr(main, "MODEL_READY", False)
    monkeypatch.setattr(main, "INIT_ERROR", "Warmup failed")

    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["ready"] is False
    assert data["state"] == ServiceState.FAILED
    assert data["error"] == "Warmup failed"

    resp_ready = test_client.get("/ready")
    assert resp_ready.status_code == 503
    assert resp_ready.json()["detail"]["status"] == "not_ready"

    # State 3: READY
    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.READY)
    monkeypatch.setattr(main, "MODEL_READY", True)
    monkeypatch.setattr(main, "MODEL_NAME", "PP-OCRv5")
    monkeypatch.setattr(main, "INIT_ERROR", None)

    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["state"] == ServiceState.READY

    resp_ready = test_client.get("/ready")
    assert resp_ready.status_code == 200
    data_ready = resp_ready.json()
    assert data_ready["status"] == "ready"
    assert data_ready["ready"] is True
    assert data_ready["state"] == ServiceState.READY


def test_requests_rejected_when_not_ready(test_client, monkeypatch):
    """Ensure OCR endpoints return 503 when service is not ready."""
    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.FAILED)
    monkeypatch.setattr(main, "MODEL_READY", False)

    b64 = create_dummy_base64_image()

    resp = test_client.post("/ocr/base64", json={"image": b64})
    assert resp.status_code == 503
    assert "not ready" in resp.json()["detail"]

    resp_batch = test_client.post("/ocr/batch", json={"images": ["/nonexistent.png"]})
    assert resp_batch.status_code == 503
    assert "not ready" in resp_batch.json()["detail"]


def test_normal_ocr_sequence(test_client, monkeypatch):
    """Test normal sequential OCR requests when service is READY."""
    class FakeOCR:
        def predict(self, img, **kwargs):
            return [{
                "rec_texts": ["Hello World"],
                "rec_scores": [0.95],
                "rec_boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            }]

    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.READY)
    monkeypatch.setattr(main, "MODEL_READY", True)
    monkeypatch.setattr(main, "MODEL_NAME", "PP-OCRv5")
    monkeypatch.setattr(main, "ocr", FakeOCR())

    b64 = create_dummy_base64_image()

    resp = test_client.post("/ocr/base64", json={"image": b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello World"
    assert data["confidence"] > 0.90
    assert data["status"] == "SUCCESS"
    assert data["regions"] == 1


def test_inference_exception_handling(test_client, monkeypatch):
    """Test that native inference exceptions yield structured FAILED status and do not crash or return fake success."""
    class CrashingOCR:
        def predict(self, img, **kwargs):
            raise RuntimeError("Native C++ predictor exception simulation")

    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.READY)
    monkeypatch.setattr(main, "MODEL_READY", True)
    monkeypatch.setattr(main, "MODEL_NAME", "PP-OCRv5")
    monkeypatch.setattr(main, "ocr", CrashingOCR())

    b64 = create_dummy_base64_image()

    resp = test_client.post("/ocr/base64", json={"image": b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAILED"
    assert data["text"] == ""
    assert "ocr_cascade_exception" in data["selection_reason"] or "ocr_exception" in data["selection_reason"]


def test_concurrent_requests_serialization(test_client, monkeypatch):
    """Verify that concurrent OCR requests serialize access to the shared predictor using _inference_lock."""
    active_inferences = 0
    max_concurrent_inferences = 0
    lock_concurrency_check = threading.Lock()

    class SlowOCR:
        def predict(self, img, **kwargs):
            nonlocal active_inferences, max_concurrent_inferences
            with lock_concurrency_check:
                active_inferences += 1
                if active_inferences > max_concurrent_inferences:
                    max_concurrent_inferences = active_inferences

            time.sleep(0.05)  # Simulate native execution duration

            with lock_concurrency_check:
                active_inferences -= 1

            return [{
                "rec_texts": ["Concurrent Test"],
                "rec_scores": [0.90],
                "rec_boxes": [[[0, 0], [5, 0], [5, 5], [0, 5]]],
            }]

    monkeypatch.setattr(main, "SERVICE_STATE", ServiceState.READY)
    monkeypatch.setattr(main, "MODEL_READY", True)
    monkeypatch.setattr(main, "MODEL_NAME", "PP-OCRv5")
    monkeypatch.setattr(main, "ocr", SlowOCR())

    b64 = create_dummy_base64_image()

    threads = []
    results = [None] * 5

    def make_request(idx):
        res = test_client.post("/ocr/base64", json={"image": b64})
        results[idx] = res

    for i in range(5):
        t = threading.Thread(target=make_request, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for res in results:
        assert res is not None
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"

    assert max_concurrent_inferences == 1


def test_warmup_validation(monkeypatch):
    """Test _run_warmup success and failure paths."""
    class ValidOCR:
        def predict(self, img):
            return []

    class InvalidOCR:
        def predict(self, img):
            raise RuntimeError("Warmup crash")

    assert main._run_warmup(ValidOCR()) is True
    assert main._run_warmup(InvalidOCR()) is False
    assert main._run_warmup(None) is False
