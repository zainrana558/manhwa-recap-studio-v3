"""
PaddleOCR PP-OCRv5 Mini-Service

A FastAPI service providing high-accuracy OCR for manhwa/manga recap pipelines.
Uses PaddleOCR with PP-OCRv5 (falls back to PP-OCRv4) for text extraction
from speech bubbles and captions.

Port: 3002
"""

import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paddleocr-service")

# ---------------------------------------------------------------------------
# PaddleOCR initialisation (warm-start — loaded once at import time)
# ---------------------------------------------------------------------------

ocr = None  # type: Any
MODEL_NAME = "unknown"  # type: str
MODEL_READY = False  # type: bool


def _init_ocr():
    # type: () -> None
    """Attempt to initialise PaddleOCR PP-OCRv5, falling back to PP-OCRv4.

    Retries each version a few times with a short delay before giving up —
    first-run model downloads (from HuggingFace/ModelScope/AIStudio/BOS) can
    hit transient network blips on cloud boxes with flaky egress, and a
    single failed attempt used to permanently mark the service unavailable
    for the rest of the process lifetime (every job would then silently fall
    through to VLM until the service was manually restarted).
    """
    global ocr, MODEL_NAME, MODEL_READY

    # Skip the AIStudio connectivity pre-check (adds a slow round-trip and
    # can itself fail on restrictive egress even when the actual download
    # host is reachable) — let the real download attempt be the test.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    MODEL_INIT_RETRIES = 5

    def _try_init(ocr_version):
        # type: (str) -> Any
        from paddleocr import PaddleOCR
        last_exc = None  # type: Optional[Exception]
        delay = 10
        for attempt in range(1, MODEL_INIT_RETRIES + 1):
            try:
                return PaddleOCR(
                    ocr_version=ocr_version,
                    lang="en",
                    use_angle_cls=True,
                    det_db_thresh=0.3,
                    det_limit_side_len=1216,
                    cpu_threads=4,
                    enable_mkldnn=True,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < MODEL_INIT_RETRIES:
                    logger.warning(
                        "%s init attempt %d/%d failed (%s) — retrying in %ds",
                        ocr_version, attempt, MODEL_INIT_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
        raise last_exc  # type: ignore

    try:
        ocr = _try_init("PP-OCRv5")
        MODEL_NAME = "PP-OCRv5"
        MODEL_READY = True
        logger.info("PaddleOCR PP-OCRv5 initialised successfully")
    except Exception as exc_v5:
        logger.warning("PP-OCRv5 init failed after retries (%s), falling back to PP-OCRv4", exc_v5)
        try:
            ocr = _try_init("PP-OCRv4")
            MODEL_NAME = "PP-OCRv4"
            MODEL_READY = True
            logger.info("PaddleOCR PP-OCRv4 initialised successfully (fallback)")
        except Exception as exc_v4:
            logger.error("Both PP-OCRv5 and PP-OCRv4 failed to initialise after retries: %s", exc_v4)
            MODEL_READY = False


# Run initialisation at module load so the model is ready before the first request.
_init_ocr()

# If startup init failed outright (not just slow — genuinely exhausted its
# retries), keep trying in the background instead of staying broken until
# someone notices and restarts the process. A transient network blip during
# boot shouldn't take down transcription for the rest of the box's uptime.
if not MODEL_READY:
    import threading

    def _background_retry_loop():
        # type: () -> None
        backoff_sec = 60
        max_backoff_sec = 600
        while not MODEL_READY:
            time.sleep(backoff_sec)
            logger.info("Retrying PaddleOCR initialisation in background...")
            _init_ocr()
            if not MODEL_READY:
                backoff_sec = min(backoff_sec * 2, max_backoff_sec)

    threading.Thread(target=_background_retry_loop, daemon=True).start()
    logger.warning("PaddleOCR not ready at startup — background retry loop started")

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


import re

CONFIDENCE_CUTOFF = 0.70
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
    # Strip erroneous spoken symbol conversions at sentence boundaries
    t = re.sub(r'\s*\b(minus|dash|underscore)\b\s*$', '...', text, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)

    # OCR character substitutions
    t = re.sub(r'\bHO[0O]\b', 'HOO', t)
    t = re.sub(r'\bHO\s+O\b', 'HOO', t)
    t = re.sub(r'\bgood-curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)
    t = re.sub(r'\bgood\s+curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)

    # End card first-letter / misread correction logic
    t = re.sub(r'\bB\s+to\s+be\s+continued\.*', 'To Be Continued...', t, flags=re.IGNORECASE)
    t = re.sub(r'^\s*B\s+to\s+be\b(?!\s+continued)', 'To Be Continued', t, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)

    return t


def _sort_regions_reading_order(regions, is_ui_box=False):
    # type: (List[_TextRegion], bool) -> List[_TextRegion]
    """Sort detected text regions in natural reading order.
    For UI boxes / quest windows, strictly sort top-to-bottom by y_min coordinates.
    """
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
        # Reduced line height threshold for quest / UI boxes so itemized multi-line lists segment into individual lines
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


def _ensure_list(x):
    # type: (Any) -> list
    """Safely coerce a value to a list WITHOUT ever char-splitting a string.

    Python's bare list("hello") produces ['h','e','l','l','o'] — a classic
    footgun when a value that's supposed to be "a list of N items" turns out
    to actually be a single already-joined string. This happens with
    PaddleOCR result parsing because different versions/formats of the
    result object don't always agree on whether a field is per-region
    (list) or whole-image (string). A single mis-typed field here silently
    turns real words into individual space-separated letters after
    _merge_regions' downstream space-join — e.g. "Ohh no a dungeon"
    becomes "O H H N O A D U N G E O N".
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, 'tolist'):  # numpy array
        try:
            return x.tolist()
        except Exception:
            pass
    try:
        return list(x)
    except TypeError:
        return [x]


def parse_ocr_results(result):
    """Refactored universal OCRResult parser.

    Seamlessly handles both PaddleOCR 3.x dataclasses/objects/dicts (OCRResult)
    and legacy tuple/list structures.
    """
    extracted_lines = []
    if not result:
        return extracted_lines

    # Convert generator to list if necessary
    if not isinstance(result, (list, tuple)):
        try:
            result = list(result)
        except Exception:
            pass

    for item in result:
        if item is None:
            continue

        # Handle PaddleOCR 3.x OCRResult object attributes or dict-like objects
        has_attr_rec = hasattr(item, 'rec_texts') and hasattr(item, 'rec_scores')
        has_key_rec = hasattr(item, 'keys') and 'rec_texts' in item and 'rec_scores' in item

        if has_attr_rec or has_key_rec:
            if has_attr_rec:
                texts = getattr(item, 'rec_texts', [])
                scores = getattr(item, 'rec_scores', [])
                boxes = getattr(item, 'rec_boxes', getattr(item, 'rec_polys', []))
            else:
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])
                boxes = item.get('rec_boxes', item.get('rec_polys', []))

            texts = _ensure_list(texts)
            scores = _ensure_list(scores)
            boxes = _ensure_list(boxes)

            for text, score, box in zip(texts, scores, boxes):
                box_val = box.tolist() if hasattr(box, 'tolist') else box
                extracted_lines.append({
                    "text": str(text),
                    "confidence": float(score),
                    "box": box_val
                })

        # Legacy tuple/list fallback: [[[box], (text, score)]]
        elif isinstance(item, (list, tuple)):
            for line in item:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    box = line[0]
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, score = text_info[0], text_info[1]
                    else:
                        text, score = text_info, 0.0
                    box_val = box.tolist() if hasattr(box, 'tolist') else box
                    extracted_lines.append({
                        "text": str(text),
                        "confidence": float(score),
                        "box": box_val
                    })

    return extracted_lines


def _run_ocr_on_image(img, options=None):
    # type: (np.ndarray, Optional[OCROptions]) -> List[_TextRegion]
    """Run PaddleOCR on a numpy image array and return structured regions."""
    global ocr
    if ocr is None:
        return []

    # One-time debug flag to dump result structure
    if not hasattr(_run_ocr_on_image, '_debugged'):
        _run_ocr_on_image._debugged = False  # type: ignore

    opts = options or OCROptions()

    try:
        # Call predict() or ocr() without invalid keyword arguments to prevent kwarg mismatch warnings
        if hasattr(ocr, "predict") and callable(getattr(ocr, "predict")):
            raw_result = ocr.predict(img)
        else:
            raw_result = ocr.ocr(img)
    except Exception as exc:
        logger.warning("OCR inference failed: %s", exc)
        return []

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

        # Scrubbing & filtering rules
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
    # type: () -> bool
    """Cached check for whether the `tesseract` CLI binary exists.

    Tesseract is the plan's independent last-resort OCR candidate — it
    should never be a hard dependency (Oracle free-tier images may not
    have it installed), so every call site must be able to skip it
    cleanly rather than crash. Cached because shutil.which() does a PATH
    scan and this gets checked per-panel.
    """
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    """Run Tesseract as an independent OCR candidate.

    Ensures the image is explicitly converted to a valid NumPy array and PIL Image,
    preventing `'function' object has no attribute '__array_interface__'` errors if
    an un-executed function reference or PIL Image/other object is passed.

    Supports pytesseract if available, falling back to the tesseract CLI if not.
    """
    # 1. Guard against un-executed function reference or non-array/image types
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

    # 2. Attempt pytesseract if installed
    try:
        import pytesseract
        text = pytesseract.image_to_string(pil_img, lang="eng", config="--psm 6")
        if text and text.strip():
            return text.strip(), 0.80
    except ImportError:
        pass
    except Exception as pytess_exc:
        logger.warning("[Tesseract] pytesseract.image_to_string failed (%s); falling back to CLI", pytess_exc)

    # 3. Fallback to CLI
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
            # Tesseract uses conf == -1 for non-text layout rows (blocks/
            # paragraphs/lines themselves, not actual words).
            if not word or conf < 0:
                continue
            words.append(word)
            confs.append(conf)

        if not words:
            return "", 0.0
        text = " ".join(words)
        avg_conf = (sum(confs) / len(confs)) / 100.0  # Tesseract reports 0-100
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
    Automatically detects high-density UI/quest cards to set det_limit_side_len >= 1216.
    """
    opts = options or OCROptions()
    is_ui_box = _detect_ui_card_or_borders(img)
    if is_ui_box and opts.det_limit_side_len < 1216:
        opts = opts.copy(update={"det_limit_side_len": 1216})

    # Pass 1: Standard PaddleOCR
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

    # Pass 2: PaddleOCR with preprocessing variants (upscale, contrast, invert)
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

    # Pass 3: Tesseract fallback with original & preprocessed images (guarded to prevent exception spikes)
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


def _quality_status(text, confidence, regions):
    # Empty/no-region output is explicitly UNCERTAIN, not successful text.
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
        status="ok" if MODEL_READY else "degraded",
        model=MODEL_NAME,
        ready=MODEL_READY,
    )


@app.post("/reload")
async def reload_model():
    """Force a synchronous re-attempt at model initialisation.

    Useful after fixing a network/firewall issue without needing to restart
    the whole process (systemd unit / pm2 process / docker container).
    """
    _init_ocr()
    return {"ready": MODEL_READY, "model": MODEL_NAME}


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(request: BatchOCRRequest):
    """
    Accepts a list of absolute file paths to images.
    Returns an array of {index, text, confidence, regions} objects.
    """
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="OCR model not initialised")

    t_start = time.perf_counter()
    results = []  # type: List[OCRResult]

    for idx, img_path in enumerate(request.images):
        img_array = _load_image_from_path(img_path)
        if img_array is None:
            logger.warning("Skipping unreadable image at index %d: %s", idx, img_path)
            results.append(OCRResult(index=idx, text="", confidence=0.0, regions=0, status="FAILED", quality_score=0.0, selection_reason="unreadable_image"))
            continue

        text, avg_conf, region_count, status, quality_score, candidates, reason = _ocr_with_cascade(img_array, request.options)

        if region_count == 0:
            logger.warning("[OCR] Zero regions detected for %s (index %d) — panel may be blank, or detection failed", img_path, idx)

        results.append(
            OCRResult(
                index=idx,
                text=text,
                confidence=avg_conf,
                regions=region_count,
                status=status,
                quality_score=quality_score,
                candidates=candidates,
                selection_reason=reason,
            )
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "Batch OCR: %d images processed in %.1f ms (%s)",
        len(request.images),
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
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="OCR model not initialised")

    t_start = time.perf_counter()

    img_array = _decode_base64_image(request.image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    text, avg_conf, region_count, status, quality_score, candidates, reason = _ocr_with_cascade(img_array, request.options)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "Base64 OCR: %d regions in %.1f ms (%s)",
        region_count,
        elapsed_ms,
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
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="OCR model not initialised")

    t_start = time.perf_counter()

    img_array = _decode_base64_image(request.image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    text, avg_conf, region_count, status, quality_score, candidates, reason = _ocr_with_cascade(img_array, request.options)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "Single OCR: %d regions in %.1f ms (%s)",
        region_count,
        elapsed_ms,
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
