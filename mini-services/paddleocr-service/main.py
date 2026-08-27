"""
PaddleOCR PP-OCRv5 Mini-Service

A FastAPI service providing high-accuracy OCR for manhwa/manga recap pipelines.
Uses PaddleOCR with PP-OCRv5 (falls back to PP-OCRv4) for text extraction
from speech bubbles and captions.

Port: 3002
"""

import os
# Prevent OpenMP and C++ thread collisions & PIR interpreter SIGSEGV
os.environ["FLAGS_enable_pir_api"] = "0"          # Disable experimental PIR interpreter
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import base64
import io
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.requests import Request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID:%(process)d] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paddleocr-service")

# ---------------------------------------------------------------------------
# Service Readiness State & Synchronization Locks
# ---------------------------------------------------------------------------

class ServiceState:
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


_init_lock = threading.Lock()
_inference_lock = threading.Lock()

ocr = None  # type: Any
MODEL_NAME = "unknown"  # type: str
MODEL_READY = False  # type: bool
SERVICE_STATE = ServiceState.INITIALIZING  # type: str
INIT_ERROR = None  # type: Optional[str]


def _run_warmup(ocr_obj: Any) -> bool:
    """Perform a lightweight real inference warmup on a dummy image tensor.

    Acquires _inference_lock to ensure thread safety.
    Returns True if warmup inference succeeds, False otherwise.
    """
    if ocr_obj is None:
        return False
    logger.info("Starting lightweight real inference warmup...")
    t_start = time.perf_counter()
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        with _inference_lock:
            if hasattr(ocr_obj, "predict") and callable(getattr(ocr_obj, "predict")):
                try:
                    _ = ocr_obj.predict(dummy_img)
                except TypeError:
                    _ = ocr_obj.predict(dummy_img)
            elif hasattr(ocr_obj, "ocr") and callable(getattr(ocr_obj, "ocr")):
                _ = ocr_obj.ocr(dummy_img)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info("Real inference warmup succeeded in %.2f ms", elapsed_ms)
        return True
    except Exception as exc:
        logger.error("Warmup inference failed: %s", exc, exc_info=True)
        return False


def _init_ocr() -> None:
    """Attempt to initialise PaddleOCR PP-OCRv5, falling back to PP-OCRv4.

    Runs under _init_lock to prevent race conditions during model initialization.
    Validates model with _run_warmup before marking the service READY.
    """
    global ocr, MODEL_NAME, MODEL_READY, SERVICE_STATE, INIT_ERROR

    with _init_lock:
        SERVICE_STATE = ServiceState.INITIALIZING
        MODEL_READY = False
        INIT_ERROR = None
        logger.info("Beginning PaddleOCR model initialization sequence...")

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        MODEL_INIT_RETRIES = 5

        def _try_init(ocr_version: str) -> Any:
            from paddleocr import PaddleOCR
            import re as _re
            last_exc = None  # type: Optional[Exception]
            delay = 10
            for attempt in range(1, MODEL_INIT_RETRIES + 1):
                init_kwargs = {
                    "ocr_version": ocr_version,
                    "lang": "en",
                    "use_angle_cls": True,
                    "det_db_thresh": 0.3,
                    "det_limit_side_len": 1216,
                    "cpu_threads": 1,
                    "enable_mkldnn": False,
                    "use_gpu": False,
                }
                attempt_exc = None  # type: Optional[Exception]
                for _ in range(len(init_kwargs) + 1):
                    try:
                        return PaddleOCR(**init_kwargs)
                    except Exception as exc:
                        m = _re.match(r"Unknown argument:\s*(\w+)", str(exc))
                        if m and m.group(1) in init_kwargs:
                            bad_kwarg = m.group(1)
                            logger.warning("%s: dropping unsupported constructor kwarg '%s' (%s)", ocr_version, bad_kwarg, exc)
                            del init_kwargs[bad_kwarg]
                            attempt_exc = exc
                            continue
                        attempt_exc = exc
                        break
                last_exc = attempt_exc
                if attempt < MODEL_INIT_RETRIES:
                    logger.warning(
                        "%s init attempt %d/%d failed (%s) — retrying in %ds",
                        ocr_version, attempt, MODEL_INIT_RETRIES, last_exc, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
            raise last_exc  # type: ignore

        cand_ocr = None
        cand_name = "unknown"

        try:
            cand_ocr = _try_init("PP-OCRv5")
            cand_name = "PP-OCRv5"
            logger.info("PP-OCRv5 loaded constructor successfully")
        except Exception as exc_v5:
            logger.warning("PP-OCRv5 init failed after retries (%s), falling back to PP-OCRv4", exc_v5)
            try:
                cand_ocr = _try_init("PP-OCRv4")
                cand_name = "PP-OCRv4"
                logger.info("PP-OCRv4 loaded constructor successfully (fallback)")
            except Exception as exc_v4:
                logger.error("Both PP-OCRv5 and PP-OCRv4 failed to initialise: v5=%s, v4=%s", exc_v5, exc_v4)
                ocr = None
                MODEL_NAME = "unknown"
                MODEL_READY = False
                SERVICE_STATE = ServiceState.FAILED
                INIT_ERROR = f"PP-OCRv5 error: {exc_v5}; PP-OCRv4 error: {exc_v4}"
                return

        # Perform real inference warmup validation
        if _run_warmup(cand_ocr):
            ocr = cand_ocr
            MODEL_NAME = cand_name
            MODEL_READY = True
            SERVICE_STATE = ServiceState.READY
            INIT_ERROR = None
            logger.info("PaddleOCR state transition: %s -> READY (model: %s)", ServiceState.INITIALIZING, MODEL_NAME)
        else:
            ocr = None
            MODEL_NAME = "unknown"
            MODEL_READY = False
            SERVICE_STATE = ServiceState.FAILED
            INIT_ERROR = f"{cand_name} loaded but real inference warmup failed"
            logger.error("PaddleOCR state transition: %s -> FAILED (%s)", ServiceState.INITIALIZING, INIT_ERROR)


# Run initialisation at module load so the model is ready before requests arrive.
_init_ocr()

# If startup init failed outright, run background retry loop.
if SERVICE_STATE != ServiceState.READY:
    def _background_retry_loop():
        # type: () -> None
        backoff_sec = 60
        max_backoff_sec = 600
        while SERVICE_STATE != ServiceState.READY:
            time.sleep(backoff_sec)
            logger.info("Retrying PaddleOCR initialisation in background (current state: %s)...", SERVICE_STATE)
            _init_ocr()
            if SERVICE_STATE != ServiceState.READY:
                backoff_sec = min(backoff_sec * 2, max_backoff_sec)

    threading.Thread(target=_background_retry_loop, daemon=True).start()
    logger.warning("PaddleOCR not ready at startup (state: %s) — background retry loop started", SERVICE_STATE)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PaddleOCR Service",
    description="OCR engine for manhwa/manga recap pipeline using PaddleOCR PP-OCRv5",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Sanitize validation error payloads before JSON-encoding them.
    Prevents a 500 (UnicodeDecodeError) when a malformed request body
    contains non-UTF8 bytes, e.g. multipart data sent to a JSON-only route.
    """
    def sanitize(obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    return JSONResponse(
        status_code=422,
        content={"detail": sanitize(exc.errors())},
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class OCRErrorInfo(BaseModel):
    """Per-image error details when a single image in a batch fails."""
    index: int = 0
    path: str = ""
    error: str = ""


class OCRResult(BaseModel):
    """OCR output for a single image; status must be preserved downstream."""
    index: int = 0
    text: str = ""
    confidence: float = 0.0
    regions: int = 0
    status: str = "FAILED"
    quality_score: float = 0.0
    candidates: List[dict] = Field(default_factory=list)
    selection_reason: str = ""


class OCROptions(BaseModel):
    """Optional per-request OCR tuning parameters."""
    lang: str = Field(default="en", description="Language code")
    use_angle_cls: bool = Field(default=False, description="Enable text orientation classification")
    det_db_unclip_ratio: float = Field(
        default=1.8,
        ge=0.5,
        le=3.0,
        description="Unclip ratio for DB detector.",
    )
    det_limit_side_len: int = Field(
        default=1536,
        ge=320,
        le=4096,
        description="Maximum side length for detection resize to preserve small font accuracy.",
    )
    det_db_thresh: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Binarization threshold for DB detector.",
    )
    det_db_box_thresh: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Box score threshold for DB detector.",
    )


class BatchOCRRequest(BaseModel):
    """Request body for batch OCR over file paths."""
    images: List[str] = Field(
        ...,
        description="List of absolute file paths to images on this machine.",
        min_length=1,
        max_length=500,
    )
    options: Optional[OCROptions] = Field(default=None, description="Optional OCR tuning overrides.")


class BatchOCRResponse(BaseModel):
    """Response for batch OCR."""
    results: List[OCRResult]
    model: str
    processing_time_ms: float


class Base64OCRRequest(BaseModel):
    """Request body for single base64-encoded image OCR."""
    image: str = Field(..., description="Base64-encoded image string (with or without data URI prefix).")
    options: Optional[OCROptions] = Field(default=None, description="Optional OCR tuning overrides.")


class Base64OCRResponse(BaseModel):
    """Response for single base64 image OCR."""
    text: str
    confidence: float
    regions: int
    status: str
    quality_score: float
    candidates: List[dict] = Field(default_factory=list)
    selection_reason: str = ""
    model: str
    processing_time_ms: float


class SingleOCRResponse(BaseModel):
    """Response for single file-path OCR (legacy /ocr endpoint)."""
    text: str
    confidence: float
    regions: int
    status: str
    quality_score: float
    candidates: List[dict] = Field(default_factory=list)
    selection_reason: str = ""
    model: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str
    model: str
    ready: bool
    state: str = Field(default=ServiceState.INITIALIZING, description="Service readiness state")
    error: Optional[str] = Field(default=None, description="Initialization error if any")


class ReadyResponse(BaseModel):
    """Readiness endpoint response."""
    status: str
    model: str
    ready: bool
    state: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TextRegion:
    """A single detected text region with position metadata."""
    __slots__ = ('text', 'confidence', 'x_min', 'y_min', 'y_max', 'x_max')

    def __init__(self, text, confidence, x_min, y_min, y_max, x_max):
        self.text = text
        self.confidence = confidence
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.x_max = x_max


CONFIDENCE_CUTOFF = 0.40
SYMBOL_RATIO_LIMIT = 0.35


def _is_slash_or_math_artifact(text: str) -> bool:
    """Check if text consists of repetitive slashes, backslashes, dashes, or isolated math symbols."""
    if not text:
        return True
    s = text.strip()
    if re.fullmatch(r'[/\-\\—_\s]+', s):
        return True
    if re.search(r'[/\\—\-]{2,}', s) and not re.search(r'[a-zA-Z0-9]', s):
        return True
    if re.fullmatch(r'[*+\\/\-—=]\s*[0-9A-Za-z]{0,3}', s):
        return True
    if re.fullmatch(r'[*+\\/\-—=]+', s):
        return True
    return False


def _symbol_ratio_exceeded(text: str, max_ratio: float = SYMBOL_RATIO_LIMIT) -> bool:
    """Check if ratio of non-alphanumeric to alphanumeric characters exceeds threshold."""
    if not text:
        return True
    alnum_count = len(re.findall(r'[a-zA-Z0-9]', text))
    if alnum_count == 0:
        return True
    non_alnum_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    ratio = non_alnum_count / float(alnum_count)
    return ratio > max_ratio


def _is_graphic_logo(region: '_TextRegion', img_h: int = 0, img_w: int = 0) -> bool:
    """Identify stylized main title cards / graphic logos (e.g. Solo Leveling logo misreads like 'Souls Lacing')."""
    box_w = region.x_max - region.x_min
    box_h = region.y_max - region.y_min
    if box_h <= 0 or box_w <= 0:
        return False

    text_lower = region.text.lower()
    if re.search(r'\bsouls?\s+lac(?:ing|e)\b', text_lower):
        return True

    if img_h > 0 and img_w > 0:
        aspect_ratio = box_w / float(box_h)
        area_ratio = (box_w * box_h) / float(img_w * img_h)
        if (area_ratio > 0.15 or aspect_ratio > 5.0 or box_h > img_h * 0.4) and region.confidence < 0.85:
            if not re.fullmatch(r'[\w\s.,!\'\"]+', region.text) or region.confidence < 0.75:
                return True

    return False


def _detect_ui_card_or_borders(img: np.ndarray) -> bool:
    """Check if panel image contains structured rectangular borders or high density UI/quest notification cards."""
    try:
        if img is None or img.size == 0:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = gray.shape[:2]
        img_area = img_h * img_w
        rect_count = 0
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                if 0.08 * img_area < area < 0.95 * img_area:
                    rect_count += 1
        return rect_count >= 1
    except Exception as e:
        logger.debug("UI card detection exception: %s", e)
        return False


def _clean_and_normalize_ocr_text(text: str) -> str:
    """Normalize ellipses, punctuation, character substitutions, and end cards."""
    if not text:
        return ""
    t = re.sub(r'\s*\b(minus|dash|underscore)\b\s*$', '...', text, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)

    t = re.sub(r'\bHO[0O]\b', 'HOO', t)
    t = re.sub(r'\bHO\s+O\b', 'HOO', t)
    t = re.sub(r'\bgood-curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)
    t = re.sub(r'\bgood\s+curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)

    t = re.sub(r'\bB\s+to\s+be\s+continued\.*', 'To Be Continued...', t, flags=re.IGNORECASE)
    t = re.sub(r'^\s*B\s+to\s+be\b(?!\s+continued)', 'To Be Continued', t, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)

    return t


def _sort_regions_reading_order(regions, is_ui_box=False):
    # type: (List[_TextRegion], bool) -> List[_TextRegion]
    """Sort detected text regions in natural reading order."""
    if not regions:
        return regions

    if is_ui_box:
        return sorted(regions, key=lambda r: (r.y_min, r.x_min))

    heights = [r.y_max - r.y_min for r in regions]
    mean_height = sum(heights) / len(heights) if heights else 20.0
    vertical_tolerance = max(mean_height * 0.4, 10.0)

    remaining = sorted(regions, key=lambda r: r.y_min)
    rows = []  # type: List[List[_TextRegion]]
    for r in remaining:
        placed = False
        for row in rows:
            row_y = sum(rr.y_min for rr in row) / len(row)
            if abs(r.y_min - row_y) < vertical_tolerance:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])

    rows.sort(key=lambda row: sum(rr.y_min for rr in row) / len(row))
    ordered = []  # type: List[_TextRegion]
    for row in rows:
        row.sort(key=lambda rr: rr.x_min)
        ordered.extend(row)
    return ordered


def _merge_regions(regions, is_ui_box=False):
    # type: (List[_TextRegion], bool) -> Tuple[str, float, int]
    """Merge sorted text regions into a single coherent string."""
    if not regions:
        return "", 0.0, 0

    sorted_regions = _sort_regions_reading_order(regions, is_ui_box=is_ui_box)

    lines = []  # type: List[List[_TextRegion]]
    current_line = [sorted_regions[0]]  # type: List[_TextRegion]

    for region in sorted_regions[1:]:
        prev = current_line[-1]
        vertical_gap = abs(region.y_min - prev.y_min)
        mean_h = (region.y_max - region.y_min + prev.y_max - prev.y_min) / 2
        threshold = max(mean_h * 0.2, 4.0) if is_ui_box else max(mean_h * 0.5, 10.0)

        if vertical_gap < threshold:
            current_line.append(region)
        else:
            lines.append(current_line)
            current_line = [region]

    lines.append(current_line)

    text_parts = []  # type: List[str]
    all_confidences = []  # type: List[float]

    for line in lines:
        line_sorted = sorted(line, key=lambda r: r.x_min)
        line_text = " ".join(_clean_and_normalize_ocr_text(r.text.strip()) for r in line_sorted if r.text.strip())
        if line_text:
            text_parts.append(line_text)
        for r in line_sorted:
            if r.confidence > 0:
                all_confidences.append(r.confidence)

    merged_text = _clean_and_normalize_ocr_text(" ".join(text_parts))
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return merged_text, round(avg_confidence, 4), len(sorted_regions)


def parse_ocr_results(result):
    extracted_lines = []
    if not result:
        return extracted_lines

    page_res = result[0] if isinstance(result, list) and len(result) > 0 else result

    if hasattr(page_res, 'rec_texts') or (isinstance(page_res, dict) and 'rec_texts' in page_res):
        rec_texts = getattr(page_res, 'rec_texts', None) or page_res.get('rec_texts', [])
        rec_scores = getattr(page_res, 'rec_scores', None) or page_res.get('rec_scores', [])
        rec_boxes = getattr(page_res, 'rec_boxes', None) or getattr(page_res, 'dt_polys', None) or page_res.get('rec_boxes', [])

        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
            extracted_lines.append({
                "text": str(text),
                "confidence": float(score),
                "box": box.tolist() if hasattr(box, 'tolist') else box
            })

    elif isinstance(page_res, (list, tuple)):
        for line in page_res:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                box, (text, score) = line[0], line[1]
                extracted_lines.append({
                    "text": str(text),
                    "confidence": float(score),
                    "box": box.tolist() if hasattr(box, 'tolist') else box
                })

    return extracted_lines


def _run_ocr_on_image(img, options=None):
    # type: (np.ndarray, Optional[OCROptions]) -> List[_TextRegion]
    """Run PaddleOCR on a numpy image array under _inference_lock."""
    global ocr
    if ocr is None:
        raise RuntimeError("OCR model is not initialized")

    opts = options or OCROptions()
    raw_result = None

    t_lock_start = time.perf_counter()
    with _inference_lock:
        lock_wait_ms = (time.perf_counter() - t_lock_start) * 1000.0
        if lock_wait_ms > 15.0:
            logger.info("Inference lock acquired after waiting %.2f ms", lock_wait_ms)

        if hasattr(ocr, "predict") and callable(getattr(ocr, "predict")):
            if not hasattr(_run_ocr_on_image, '_predict_kwargs_supported'):
                _run_ocr_on_image._predict_kwargs_supported = True  # type: ignore
            if _run_ocr_on_image._predict_kwargs_supported:  # type: ignore
                try:
                    raw_result = ocr.predict(
                        img,
                        text_det_unclip_ratio=opts.det_db_unclip_ratio,
                        text_det_limit_side_len=opts.det_limit_side_len,
                        text_det_thresh=opts.det_db_thresh,
                        text_det_box_thresh=opts.det_db_box_thresh,
                    )
                except TypeError as exc:
                    logger.warning("predict() rejected tuning kwargs (%s) — disabling for remaining calls", exc)
                    _run_ocr_on_image._predict_kwargs_supported = False  # type: ignore
                except Exception as exc:
                    logger.error("OCR inference predict() failed: %s", exc)
                    raise RuntimeError(f"OCR inference predict() failed: {exc}") from exc
            if raw_result is None and _run_ocr_on_image._predict_kwargs_supported is False:  # type: ignore
                try:
                    raw_result = ocr.predict(img)
                except Exception as exc:
                    logger.error("OCR inference predict() failed: %s", exc)
                    raise RuntimeError(f"OCR inference predict() failed: {exc}") from exc
        else:
            try:
                raw_result = ocr.ocr(img)
            except Exception as exc:
                logger.error("OCR inference ocr() failed: %s", exc)
                raise RuntimeError(f"OCR inference ocr() failed: {exc}") from exc

    regions = []  # type: List[_TextRegion]

    if not raw_result:
        return regions

    lines = parse_ocr_results(raw_result)
    img_h, img_w = img.shape[:2] if hasattr(img, 'shape') and len(img.shape) >= 2 else (0, 0)

    for line_data in lines:
        text = line_data.get("text", "")
        confidence = float(line_data.get("confidence", 0.0))
        box = line_data.get("box")

        if not box or not isinstance(box, (list, tuple)) or len(box) == 0:
            continue

        try:
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
        except (TypeError, ValueError, IndexError):
            continue

        reg = _TextRegion(
            text=text,
            confidence=confidence,
            x_min=min(xs), y_min=min(ys),
            y_max=max(ys), x_max=max(xs),
        )

        if confidence < CONFIDENCE_CUTOFF:
            continue
        if _is_slash_or_math_artifact(text):
            continue
        if _symbol_ratio_exceeded(text):
            continue
        if _is_graphic_logo(reg, img_h=img_h, img_w=img_w):
            continue

        regions.append(reg)

    return regions


def _load_image_from_path(file_path):
    # type: (str) -> Optional[np.ndarray]
    """Load an image from an absolute file path into a numpy array."""
    if not os.path.isfile(file_path):
        logger.warning("Image file not found: %s", file_path)
        return None
    try:
        img = Image.open(file_path).convert("RGB")
        return np.array(img)
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", file_path, exc)
        return None


def _decode_base64_image(b64_string):
    # type: (str) -> Optional[np.ndarray]
    """Decode a base64 string (with optional data-URI prefix) into a numpy array."""
    try:
        if "," in b64_string and ";base64," in b64_string:
            b64_string = b64_string.split(";base64,", 1)[1]

        img_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img)
    except Exception as exc:
        logger.warning("Failed to decode base64 image: %s", exc)
        return None


def _tesseract_available() -> bool:
    """Cached check for whether the `tesseract` CLI binary exists."""
    if not hasattr(_tesseract_available, "_cached"):
        _tesseract_available._cached = shutil.which("tesseract") is not None  # type: ignore
        if not _tesseract_available._cached:  # type: ignore
            logger.info("tesseract binary not found on PATH — Tesseract fallback tier disabled")
    return _tesseract_available._cached  # type: ignore


def _preprocess_upscale(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """Scale up image to help OCR recognize small text fonts in webtoon panels."""
    try:
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return img


def _preprocess_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to improve low-contrast panel text."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    except Exception:
        return img


def _preprocess_invert(img: np.ndarray) -> np.ndarray:
    """Invert image colors to help OCR detect dark-background or inverted speech bubbles."""
    try:
        return 255 - img
    except Exception:
        return img


def _run_tesseract_ocr(img):
    # type: (Any) -> Tuple[str, float]
    """Run Tesseract as an independent OCR candidate."""
    if callable(img):
        try:
            img = img()
        except Exception as call_exc:
            logger.error("[Tesseract] Failed to execute callable image argument: %s", call_exc)
            return "", 0.0

    if img is None:
        return "", 0.0

    try:
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
            img_array = img
        elif isinstance(img, Image.Image):
            pil_img = img
            img_array = np.array(img)
        else:
            img_array = np.array(img)
            pil_img = Image.fromarray(img_array)
    except Exception as conv_exc:
        logger.error("[Tesseract] Failed to convert image object to PIL Image / NumPy array: %s", conv_exc)
        return "", 0.0

    try:
        import pytesseract
        text = pytesseract.image_to_string(pil_img, lang="eng", config="--psm 6")
        if text and text.strip():
            return text.strip(), 0.80
    except ImportError:
        pass
    except Exception as pytess_exc:
        logger.warning("[Tesseract] pytesseract.image_to_string failed (%s); falling back to CLI", pytess_exc)

    if not _tesseract_available():
        return "", 0.0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        pil_img.save(tmp_path)
        result = subprocess.run(
            ["tesseract", tmp_path, "stdout", "--psm", "6", "tsv"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("tesseract exited %d: %s", result.returncode, (result.stderr or "")[:200])
            return "", 0.0

        words = []  # type: List[str]
        confs = []  # type: List[float]
        lines = result.stdout.splitlines()
        if len(lines) < 2:
            return "", 0.0
        header = lines[0].split("\t")
        try:
            text_idx = header.index("text")
            conf_idx = header.index("conf")
        except ValueError:
            return "", 0.0
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) <= max(text_idx, conf_idx):
                continue
            word = cols[text_idx].strip()
            try:
                conf = float(cols[conf_idx])
            except ValueError:
                continue
            if not word or conf < 0:
                continue
            words.append(word)
            confs.append(conf)

        if not words:
            return "", 0.0
        text = " ".join(words)
        avg_conf = (sum(confs) / len(confs)) / 100.0
        return text, max(0.0, min(1.0, avg_conf))
    except subprocess.TimeoutExpired:
        logger.warning("tesseract timed out after 5s")
        return "", 0.0
    except Exception as exc:
        logger.warning("tesseract OCR failed: %s", exc)
        return "", 0.0
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _ocr_with_cascade(img, options=None):
    # type: (np.ndarray, Optional[OCROptions]) -> Tuple[str, float, int, str, float, List[dict], str]
    """Run PaddleOCR pass (standard + preprocessing variants) and, if still
    UNCERTAIN/FAILED, fall back to Tesseract as an independent candidate.
    """
    try:
        opts = options or OCROptions()
        is_ui_box = _detect_ui_card_or_borders(img)
        if is_ui_box and opts.det_limit_side_len < 1216:
            opts = opts.copy(update={"det_limit_side_len": 1216})

        regions = _run_ocr_on_image(img, opts)
        merged_text, avg_conf, region_count = _merge_regions(regions, is_ui_box=is_ui_box)
        status, quality_score, reason = _quality_status(merged_text, avg_conf, region_count)

        candidates = [{
            "text": merged_text, "confidence": avg_conf, "regions": region_count,
            "provider": "paddleocr", "model": MODEL_NAME, "variant": "standard",
        }]

        best_tuple = (merged_text, avg_conf, region_count, status, quality_score, candidates, reason)

        if status == "SUCCESS":
            return best_tuple

        preprocessing_passes = [
            ("upscale_1.5x", lambda i: _preprocess_upscale(i, 1.5)),
            ("contrast_clahe", _preprocess_contrast),
            ("color_inverted", _preprocess_invert),
        ]

        for variant_name, prep_fn in preprocessing_passes:
            prep_img = prep_fn(img)
            var_regions = _run_ocr_on_image(prep_img, options)
            var_text, var_conf, var_count = _merge_regions(var_regions)
            var_status, var_quality, var_reason = _quality_status(var_text, var_conf, var_count)

            candidates.append({
                "text": var_text, "confidence": var_conf, "regions": var_count,
                "provider": "paddleocr", "model": MODEL_NAME, "variant": variant_name,
            })

            if var_status == "SUCCESS" and var_quality > best_tuple[4]:
                best_tuple = (
                    var_text, var_conf, var_count, var_status, var_quality,
                    candidates, f"paddleocr_variant_{variant_name}:{var_reason}",
                )
                return best_tuple
            elif var_quality > best_tuple[4]:
                best_tuple = (
                    var_text, var_conf, var_count, var_status, var_quality,
                    candidates, f"paddleocr_variant_{variant_name}:{var_reason}",
                )

        try:
            tess_passes = [
                ("standard", img),
                ("upscale_1.5x", _preprocess_upscale(img, 1.5)),
                ("contrast_clahe", _preprocess_contrast(img)),
            ]

            for tess_variant, t_img in tess_passes:
                tess_text, tess_conf = _run_tesseract_ocr(t_img)
                if not tess_text:
                    continue

                tess_regions = len(tess_text.split())
                candidates.append({
                    "text": tess_text, "confidence": tess_conf, "regions": tess_regions,
                    "provider": "tesseract", "model": "tesseract", "variant": tess_variant,
                })

                tess_status, tess_quality, tess_reason = _quality_status(tess_text, tess_conf, tess_regions)
                if tess_status == "SUCCESS" and tess_quality > best_tuple[4]:
                    return (
                        tess_text, tess_conf, tess_regions, tess_status, tess_quality,
                        candidates, f"tesseract_{tess_variant}_fallback_beat_paddleocr:{tess_reason}",
                    )
        except Exception as tess_err:
            logger.warning("Tesseract fallback cascade failed gracefully: %s", tess_err)

        return best_tuple
    except Exception as exc:
        logger.error("Unexpected error in _ocr_with_cascade: %s", exc, exc_info=True)
        return "", 0.0, 0, "FAILED", 0.0, [], f"ocr_cascade_exception:{exc}"


def _quality_status(text, confidence, regions):
    quality = round(max(0.0, min(1.0, (confidence or 0.0))) * (1.0 if regions > 0 else 0.0), 4)
    if regions > 0 and text.strip() and confidence >= 0.55:
        return "SUCCESS", quality, "accepted_confident_candidate"
    if regions > 0 or text.strip():
        return "UNCERTAIN", quality, "low_confidence_or_incomplete_candidate"
    return "UNCERTAIN", 0.0, "no_regions_detected_blank_or_failed"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health and model readiness status."""
    return HealthResponse(
        status="ok" if SERVICE_STATE == ServiceState.READY else ("initializing" if SERVICE_STATE == ServiceState.INITIALIZING else "degraded"),
        model=MODEL_NAME,
        ready=MODEL_READY,
        state=SERVICE_STATE,
        error=INIT_ERROR,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready_check():
    """Explicit readiness endpoint. Returns HTTP 200 if ready, HTTP 503 if not ready."""
    if SERVICE_STATE == ServiceState.READY:
        return ReadyResponse(
            status="ready",
            model=MODEL_NAME,
            ready=True,
            state=SERVICE_STATE,
        )
    raise HTTPException(
        status_code=503,
        detail={
            "status": "not_ready",
            "model": MODEL_NAME,
            "ready": False,
            "state": SERVICE_STATE,
            "error": INIT_ERROR,
        },
    )


@app.post("/reload")
async def reload_model():
    """Force a synchronous re-attempt at model initialisation."""
    logger.info("Manual reload triggered on /reload endpoint")
    _init_ocr()
    if SERVICE_STATE != ServiceState.READY:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "model": MODEL_NAME, "state": SERVICE_STATE, "error": INIT_ERROR},
        )
    return {"ready": MODEL_READY, "model": MODEL_NAME, "state": SERVICE_STATE}


OCR_CONCURRENCY = max(1, int(os.environ.get("OCR_CONCURRENCY", "1")))
_ocr_semaphore = None  # type: Any


def _get_ocr_semaphore():
    global _ocr_semaphore
    if _ocr_semaphore is None:
        import asyncio
        _ocr_semaphore = asyncio.Semaphore(OCR_CONCURRENCY)
    return _ocr_semaphore


async def _ocr_one(idx, total_count, img_path, options):
    # type: (int, int, str, Optional[OCROptions]) -> OCRResult
    import asyncio
    t_one_start = time.perf_counter()
    try:
        img_array = _load_image_from_path(img_path)
        if img_array is None:
            logger.warning("[OCR] Skipping unreadable image %d/%d: %s", idx + 1, total_count, img_path)
            return OCRResult(index=idx, text="", confidence=0.0, regions=0, status="FAILED", quality_score=0.0, selection_reason="unreadable_image")

        async with _get_ocr_semaphore():
            text, avg_conf, region_count, status, quality_score, candidates, reason = await asyncio.to_thread(
                _ocr_with_cascade, img_array, options
            )

        elapsed_ms = (time.perf_counter() - t_one_start) * 1000.0
        logger.info(
            "[OCR] Completed image %d/%d (%s) in %.1f ms — status=%s, regions=%d, conf=%.2f",
            idx + 1, total_count, os.path.basename(img_path), elapsed_ms, status, region_count, avg_conf
        )

        return OCRResult(
            index=idx, text=text, confidence=avg_conf, regions=region_count,
            status=status, quality_score=quality_score, candidates=candidates, selection_reason=reason,
        )
    except Exception as exc:
        logger.error("[OCR] Exception during OCR for image %s at index %d: %s", img_path, idx, exc, exc_info=True)
        return OCRResult(
            index=idx, text="", confidence=0.0, regions=0, status="FAILED",
            quality_score=0.0, candidates=[], selection_reason=f"ocr_exception:{exc}",
        )


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(request: BatchOCRRequest):
    """
    Accepts a list of absolute file paths to images.
    Returns an array of {index, text, confidence, regions} objects.
    """
    if SERVICE_STATE != ServiceState.READY or ocr is None:
        logger.warning("[Batch OCR] Request rejected: service state is %s", SERVICE_STATE)
        raise HTTPException(
            status_code=503,
            detail=f"OCR service is not ready for inference (state: {SERVICE_STATE})",
        )

    import asyncio
    t_start = time.perf_counter()
    total = len(request.images)
    logger.info("[Batch OCR] Received request for %d images (concurrency limit: %d)", total, OCR_CONCURRENCY)

    results = list(await asyncio.gather(*[
        _ocr_one(idx, total, img_path, request.options) for idx, img_path in enumerate(request.images)
    ]))  # type: List[OCRResult]

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "[Batch OCR] Batch completed: %d images in %.1f ms (%s)",
        total,
        elapsed_ms,
        MODEL_NAME,
    )

    return BatchOCRResponse(
        results=results,
        model=MODEL_NAME,
        processing_time_ms=elapsed_ms,
    )


@app.post("/ocr/base64", response_model=Base64OCRResponse)
async def ocr_base64(request: Base64OCRRequest):
    """
    Accepts a single base64-encoded image and returns OCR transcription.
    """
    if SERVICE_STATE != ServiceState.READY or ocr is None:
        logger.warning("[Base64 OCR] Request rejected: service state is %s", SERVICE_STATE)
        raise HTTPException(
            status_code=503,
            detail=f"OCR service is not ready for inference (state: {SERVICE_STATE})",
        )

    import asyncio
    t_start = time.perf_counter()

    img_array = _decode_base64_image(request.image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        text, avg_conf, region_count, status, quality_score, candidates, reason = await asyncio.to_thread(
            _ocr_with_cascade, img_array, request.options
        )
    except Exception as exc:
        logger.error("[Base64 OCR] Exception during OCR: %s", exc, exc_info=True)
        text, avg_conf, region_count, status, quality_score, candidates, reason = "", 0.0, 0, "FAILED", 0.0, [], f"ocr_exception:{exc}"

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "[Base64 OCR] Completed: %d regions in %.1f ms (status=%s, %s)",
        region_count,
        elapsed_ms,
        status,
        MODEL_NAME,
    )

    return Base64OCRResponse(
        text=text,
        confidence=avg_conf,
        regions=region_count,
        status=status,
        quality_score=quality_score,
        candidates=candidates,
        selection_reason=reason,
        model=MODEL_NAME,
        processing_time_ms=elapsed_ms,
    )


@app.post("/ocr", response_model=SingleOCRResponse)
async def ocr_single(request: Base64OCRRequest):
    """
    Legacy single-image OCR endpoint.
    """
    if SERVICE_STATE != ServiceState.READY or ocr is None:
        logger.warning("[Single OCR] Request rejected: service state is %s", SERVICE_STATE)
        raise HTTPException(
            status_code=503,
            detail=f"OCR service is not ready for inference (state: {SERVICE_STATE})",
        )

    import asyncio
    t_start = time.perf_counter()

    img_array = _decode_base64_image(request.image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        text, avg_conf, region_count, status, quality_score, candidates, reason = await asyncio.to_thread(
            _ocr_with_cascade, img_array, request.options
        )
    except Exception as exc:
        logger.error("[Single OCR] Exception during OCR: %s", exc, exc_info=True)
        text, avg_conf, region_count, status, quality_score, candidates, reason = "", 0.0, 0, "FAILED", 0.0, [], f"ocr_exception:{exc}"

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "[Single OCR] Completed: %d regions in %.1f ms (status=%s, %s)",
        region_count,
        elapsed_ms,
        status,
        MODEL_NAME,
    )

    return SingleOCRResponse(
        text=text,
        confidence=avg_conf,
        regions=region_count,
        status=status,
        quality_score=quality_score,
        candidates=candidates,
        selection_reason=reason,
        model=MODEL_NAME,
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting PaddleOCR service on port 3002 ...")
    uvicorn.run(app, host="0.0.0.0", port=3002, workers=1)
