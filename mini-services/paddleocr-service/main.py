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
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

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
    """Attempt to initialise PaddleOCR PP-OCRv5, falling back to PP-OCRv4."""
    global ocr, MODEL_NAME, MODEL_READY

    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            ocr_version="PP-OCRv5",
            lang="en",
        )
        MODEL_NAME = "PP-OCRv5"
        MODEL_READY = True
        logger.info("PaddleOCR PP-OCRv5 initialised successfully")
    except Exception as exc_v5:
        logger.warning("PP-OCRv5 init failed (%s), falling back to PP-OCRv4", exc_v5)
        try:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(
                ocr_version="PP-OCRv4",
                lang="en",
            )
            MODEL_NAME = "PP-OCRv4"
            MODEL_READY = True
            logger.info("PaddleOCR PP-OCRv4 initialised successfully (fallback)")
        except Exception as exc_v4:
            logger.error("Both PP-OCRv5 and PP-OCRv4 failed to initialise: %s", exc_v4)
            MODEL_READY = False


# Run initialisation at module load so the model is ready before the first request.
_init_ocr()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PaddleOCR Service",
    description="OCR engine for manhwa/manga recap pipeline using PaddleOCR PP-OCRv5",
    version="1.0.0",
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
    """OCR output for a single image."""
    index: int = 0
    text: str = ""
    confidence: float = 0.0
    regions: int = 0


class OCROptions(BaseModel):
    """Optional per-request OCR tuning parameters."""
    lang: str = Field(default="en", description="Language code")
    use_angle_cls: bool = Field(default=True, description="Enable text orientation classification")
    det_db_unclip_ratio: float = Field(
        default=1.8,
        ge=0.5,
        le=3.0,
        description="Unclip ratio for DB detector.",
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
    model: str
    processing_time_ms: float


class SingleOCRResponse(BaseModel):
    """Response for single file-path OCR (legacy /ocr endpoint)."""
    text: str
    confidence: float
    regions: int
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


def _sort_regions_reading_order(regions):
    # type: (List[_TextRegion]) -> List[_TextRegion]
    """Sort detected text regions in natural reading order."""
    if not regions:
        return regions

    heights = [r.y_max - r.y_min for r in regions]
    mean_height = sum(heights) / len(heights) if heights else 20.0
    vertical_tolerance = max(mean_height * 0.4, 10.0)

    def _sort_key(r):
        row = r.y_min // vertical_tolerance
        return (row, r.x_min)

    return sorted(regions, key=_sort_key)


def _merge_regions(regions):
    # type: (List[_TextRegion]) -> Tuple[str, float, int]
    """Merge sorted text regions into a single coherent string."""
    if not regions:
        return "", 0.0, 0

    sorted_regions = _sort_regions_reading_order(regions)

    lines = []  # type: List[List[_TextRegion]]
    current_line = [sorted_regions[0]]  # type: List[_TextRegion]

    for region in sorted_regions[1:]:
        prev = current_line[-1]
        vertical_gap = abs(region.y_min - prev.y_min)
        mean_h = (region.y_max - region.y_min + prev.y_max - prev.y_min) / 2
        threshold = max(mean_h * 0.5, 10.0)

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
        line_text = " ".join(r.text.strip() for r in line_sorted if r.text.strip())
        if line_text:
            text_parts.append(line_text)
        for r in line_sorted:
            if r.confidence > 0:
                all_confidences.append(r.confidence)

    merged_text = "\n".join(text_parts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return merged_text, round(avg_confidence, 4), len(sorted_regions)


def _run_ocr_on_image(img):
    # type: (np.ndarray) -> List[_TextRegion]
    """Run PaddleOCR on a numpy image array and return structured regions."""
    global ocr
    if ocr is None:
        return []

    # One-time debug flag to dump result structure
    if not hasattr(_run_ocr_on_image, '_debugged'):
        _run_ocr_on_image._debugged = False  # type: ignore

    try:
        # PaddleOCR v3.7+ uses predict(); older versions use ocr()
        raw_result = ocr.predict(img)
    except TypeError:
        try:
            raw_result = ocr.ocr(img)
        except Exception as exc:
            logger.warning("OCR inference failed: %s", exc)
            return []
    except Exception as exc:
        logger.warning("OCR inference failed: %s", exc)
        return []

    regions = []  # type: List[_TextRegion]

    if not raw_result:
        return regions

    # Convert generator to list
    if not isinstance(raw_result, list):
        try:
            raw_result = list(raw_result)
        except Exception:
            pass

    for item in raw_result:
        if item is None:
            continue

        # --- Debug: dump first result structure ---
        if not _run_ocr_on_image._debugged:  # type: ignore
            _run_ocr_on_image._debugged = True  # type: ignore
            logger.info("[DEBUG] result item type: %s", type(item).__name__)
            if hasattr(item, 'keys'):
                try:
                    logger.info("[DEBUG] result item keys: %s", list(item.keys()))  # type: ignore
                except Exception:
                    pass
            if hasattr(item, '__dict__'):
                logger.info("[DEBUG] result item attrs: %s", list(item.__dict__.keys()))

        # --- PaddleX PipelineResult: has .keys() or dict-like access ---
        # Try to extract data from dict-like objects (PipelineResult, dict, etc.)
        extracted = None  # type: Optional[dict]

        if hasattr(item, 'keys') or hasattr(item, 'get'):
            try:
                extracted = dict(item) if not isinstance(item, dict) else item
            except Exception:
                try:
                    extracted = {k: item[k] for k in item.keys()}  # type: ignore
                except Exception:
                    pass

        if extracted:
            # PaddleX wraps results in 'output' key
            data = extracted.get('output', extracted)

            # Try various key naming conventions
            texts = None  # type: Optional[list]
            scores = None  # type: Optional[list]
            polys = None  # type: Optional[list]

            for tk in ('rec_texts', 'rec_text', 'texts', 'text'):
                if tk in data:
                    texts = data[tk]
                    break
            for sk in ('rec_scores', 'rec_score', 'scores', 'score', 'confs'):
                if sk in data:
                    scores = data[sk]
                    break
            for pk in ('rec_polys', 'dt_polys', 'dt_poly', 'polys', 'poly', 'boxes', 'bboxes'):
                if pk in data:
                    polys = data[pk]
                    break

            if texts is None:
                # Debug: log available keys so we can fix
                logger.debug("OCR result keys: %s", list(data.keys()) if hasattr(data, 'keys') else type(data))
                continue

            texts = list(texts) if not isinstance(texts, list) else texts
            scores = list(scores) if scores and not isinstance(scores, list) else (scores or [])
            polys = list(polys) if polys and not isinstance(polys, list) else (polys or [])

            for k in range(len(texts)):
                text = str(texts[k]) if k < len(texts) else ''
                confidence = float(scores[k]) if k < len(scores) else 0.0
                poly = polys[k] if k < len(polys) else None
                if poly is None:
                    continue
                # Handle numpy arrays for polys
                if hasattr(poly, 'tolist'):
                    poly = poly.tolist()
                if not isinstance(poly, (list, tuple)) or len(poly) == 0:
                    continue
                xs = [float(pt[0]) for pt in poly]
                ys = [float(pt[1]) for pt in poly]
                regions.append(_TextRegion(
                    text=text,
                    confidence=confidence,
                    x_min=min(xs), y_min=min(ys),
                    y_max=max(ys), x_max=max(xs),
                ))
            continue

        # --- Old format: list of pages, each page is list of (bbox, (text, conf)) ---
        if isinstance(item, list):
            for line in item:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                bbox = line[0]
                text_info = line[1]
                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                confidence = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                regions.append(_TextRegion(
                    text=text,
                    confidence=confidence,
                    x_min=min(xs), y_min=min(ys),
                    y_max=max(ys), x_max=max(xs),
                ))

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
            results.append(OCRResult(index=idx, text="", confidence=0.0, regions=0))
            continue

        regions = _run_ocr_on_image(img_array)
        merged_text, avg_conf, region_count = _merge_regions(regions)

        results.append(
            OCRResult(
                index=idx,
                text=merged_text,
                confidence=avg_conf,
                regions=region_count,
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

    regions = _run_ocr_on_image(img_array)
    merged_text, avg_conf, region_count = _merge_regions(regions)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "Base64 OCR: %d regions in %.1f ms (%s)",
        region_count,
        elapsed_ms,
        MODEL_NAME,
    )

    return Base64OCRResponse(
        text=merged_text,
        confidence=avg_conf,
        regions=region_count,
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

    regions = _run_ocr_on_image(img_array)
    merged_text, avg_conf, region_count = _merge_regions(regions)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        "Single OCR: %d regions in %.1f ms (%s)",
        region_count,
        elapsed_ms,
        MODEL_NAME,
    )

    return SingleOCRResponse(
        text=merged_text,
        confidence=avg_conf,
        regions=region_count,
        model=MODEL_NAME,
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting PaddleOCR service on port 3002 ...")
    uvicorn.run(app, host="0.0.0.0", port=3002, workers=1)
