"""
PaddleOCR PP-OCRv5 Mini-Service

A FastAPI service providing high-accuracy OCR for manhwa/manga recap pipelines.
Uses PaddleOCR with PP-OCRv5 (falls back to PP-OCRv4) for text extraction
from speech bubbles and captions.

Port: 3002
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paddleocr-service")

# ---------------------------------------------------------------------------
# PaddleOCR initialisation (warm-start — loaded once at import time)
# ---------------------------------------------------------------------------

ocr: Any = None
MODEL_NAME: str = "unknown"
MODEL_READY: bool = False


def _init_ocr() -> None:
    """Attempt to initialise PaddleOCR PP-OCRv5, falling back to PP-OCRv4."""
    global ocr, MODEL_NAME, MODEL_READY

    # Try PP-OCRv5 first (+13% accuracy over v4)
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
    index: int
    path: str
    error: str


class OCRResult(BaseModel):
    """OCR output for a single image."""
    index: int
    text: str
    confidence: float
    regions: int


class OCROptions(BaseModel):
    """Optional per-request OCR tuning parameters."""
    lang: str = Field(default="en", description="Language code (e.g. 'en', 'ch', 'japan', 'korean')")
    use_angle_cls: bool = Field(default=True, description="Enable text orientation classification")
    det_db_unclip_ratio: float = Field(
        default=1.8,
        ge=0.5,
        le=3.0,
        description="Unclip ratio for DB detector. Higher values better capture rounded speech bubbles.",
    )


class BatchOCRRequest(BaseModel):
    """Request body for batch OCR over file paths."""
    images: list[str] = Field(
        ..., description="List of absolute file paths to images on this machine.",
        min_length=1,
        max_length=500,
    )
    options: Optional[OCROptions] = Field(default=None, description="Optional OCR tuning overrides.")


class BatchOCRResponse(BaseModel):
    """Response for batch OCR."""
    results: list[OCRResult]
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


@dataclass
class _TextRegion:
    """A single detected text region with position metadata."""
    text: str
    confidence: float
    # Bounding-box top-left coordinates for reading-order sorting
    x_min: float
    y_min: float
    y_max: float
    x_max: float


def _sort_regions_reading_order(regions: list[_TextRegion]) -> list[_TextRegion]:
    """Sort detected text regions in natural reading order.

    Strategy:
    1. Sort primarily by vertical position (y_min — top-to-bottom).
    2. Within the same vertical band (±20% of mean line height),
       sort left-to-right by x_min.
    """
    if not regions:
        return regions

    # Compute a reasonable "line height" threshold from the data itself.
    heights = [r.y_max - r.y_min for r in regions]
    mean_height = sum(heights) / len(heights) if heights else 20.0
    vertical_tolerance = max(mean_height * 0.4, 10.0)  # 40% of mean height, min 10px

    def _sort_key(r: _TextRegion) -> tuple[float, float]:
        # Bucket y into discrete rows
        row = r.y_min // vertical_tolerance
        return (row, r.x_min)

    return sorted(regions, key=_sort_key)


def _merge_regions(regions: list[_TextRegion]) -> tuple[str, float, int]:
    """Merge sorted text regions into a single coherent string.

    Regions on different vertical lines are separated by newlines.
    Regions on the same line are joined with spaces.
    """
    if not regions:
        return "", 0.0, 0

    sorted_regions = _sort_regions_reading_order(regions)

    lines: list[list[_TextRegion]] = []
    current_line: list[_TextRegion] = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        # If this region starts at roughly the same vertical position as the
        # last region in the current line, treat it as part of the same line.
        prev = current_line[-1]
        vertical_gap = abs(region.y_min - prev.y_min)
        mean_h = (region.y_max - region.y_min + prev.y_max - prev.y_min) / 2
        threshold = max(mean_h * 0.5, 10.0)

        if vertical_gap < threshold:
            current_line.append(region)
        else:
            lines.append(current_line)
            current_line = [region]

    lines.append(current_line)  # don't forget the last line

    # Build the final text
    text_parts: list[str] = []
    all_confidences: list[float] = []

    for line in lines:
        # Sort each line left-to-right
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


def _run_ocr_on_image(
    img: np.ndarray,
) -> list[_TextRegion]:
    """Run PaddleOCR on a numpy image array and return structured regions."""
    global ocr
    if ocr is None:
        return []

    try:
        raw_result = ocr.ocr(img, cls=True)
    except Exception as exc:
        logger.warning("OCR inference failed: %s", exc)
        return []

    regions: list[_TextRegion] = []

    # PaddleOCR returns: list[page_results] where each page_result is
    # a list of (bbox, (text, confidence)) tuples.
    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    if not raw_result:
        return regions

    for page in raw_result:
        if page is None:
            continue
        for line in page:
            if len(line) < 2:
                continue
            bbox = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text_info = line[1]  # (text, confidence)

            text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
            confidence = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0

            # Extract bounding box coordinates
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]

            regions.append(
                _TextRegion(
                    text=text,
                    confidence=confidence,
                    x_min=min(xs),
                    y_min=min(ys),
                    y_max=max(ys),
                    x_max=max(xs),
                )
            )

    return regions


def _load_image_from_path(file_path: str) -> Optional[np.ndarray]:
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


def _decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decode a base64 string (with optional data-URI prefix) into a numpy array."""
    try:
        # Strip data-URI prefix if present
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

    Uses PaddleOCR to extract text from each image.
    For manhwa/manga, text is extracted from speech bubbles and captions.
    The results are ordered by reading position (top-to-bottom, left-to-right).
    """
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="OCR model not initialised")

    t_start = time.perf_counter()
    results: list[OCRResult] = []

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

    The image string may include a data-URI prefix (e.g. "data:image/png;base64,...")
    which will be stripped automatically.
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
    Accepts a single base64-encoded image and returns transcription text.
    Alias for /ocr/base64 with a slightly different response shape.
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
    logger.info("Starting PaddleOCR service on port 3002 …")
    uvicorn.run(app, host="0.0.0.0", port=3002, workers=1)
