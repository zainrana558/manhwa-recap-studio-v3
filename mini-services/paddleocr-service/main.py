"""
PaddleOCR PP-OCRv4 Mini-Service

A FastAPI service providing high-accuracy OCR for manhwa/manga recap pipelines.
Uses PaddleOCR PP-OCRv4 (see requirements.txt for the paddleocr==2.9.1 /
paddlepaddle==2.6.2 pin and why) for text extraction from speech bubbles and
captions.

Port: 3002
"""

import os
import signal
import sys

# ---------------------------------------------------------------------------
# Prevent OpenMP and C++ thread collisions & PIR interpreter SIGSEGV
#
# PaddlePaddle's PIR (Paddle Intermediate Representation) interpreter is
# known to crash with SIGSEGV during garbage collection on certain CPU-only
# configurations. Setting a single flag (FLAGS_enable_pir_api=0) is NOT
# sufficient — the interpreter can still be instantiated by the inference
# engine. We disable every PIR-related flag to force the legacy executor.
# ---------------------------------------------------------------------------
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_pir_apply_inplace_pass"] = "0"
os.environ["FLAGS_pir_apply_general_fuse_pass"] = "0"
os.environ["FLAGS_enable_pir_compatible"] = "0"
os.environ["FLAGS_enable_pir_debug"] = "0"
os.environ["FLAGS_pir_print_group_ops"] = "0"
os.environ["FLAGS_pir_onednn_use_execution_pool"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["GLOG_minloglevel"] = "2"  # Suppress noisy Paddle warnings

# Thread count limits — prevents OpenMP/MKL thread explosion on small instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "passive"  # threads sleep instead of spin-waiting between requests
os.environ["PADDLE_CPP_LOG_LEVEL"] = "3"   # suppress noisy Paddle C++ init logging
# No-op on the CPU-only `paddlepaddle` wheel pinned in requirements.txt (no
# GPU code is even compiled in) — kept as a harmless safeguard in case this
# ever runs against a `paddlepaddle-gpu` build instead.
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0"


def _sigsegv_handler(signum, frame):
    """Log a helpful crash message instead of silently dying.

    The stack trace from a PIR interpreter SIGSEGV is not useful for
    debugging OCR issues — this handler prints a clear diagnostic and
    exits with a non-zero code so the process manager can restart it.
    """
    # Use stderr directly — logger may not be initialized yet if the
    # crash happens during early PaddlePaddle import.
    import traceback
    sys.stderr.write(
        "\n=== SIGSEGV (segmentation fault) caught ===\n"
        "requirements.txt already pins paddlepaddle==2.6.2 / paddleocr==2.9.1\n"
        "(pre-PIR legacy executor) specifically to avoid this crash class.\n"
        "If you are seeing this, check that the running environment actually\n"
        "has those versions installed (pip show paddlepaddle paddleocr) rather\n"
        "than a stale 3.x install.\n"
        f"PID={os.getpid()}, signal={signum}\n"
    )
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)


# Install SIGSEGV handler AFTER setting env vars but BEFORE importing PaddlePaddle,
# so any early-init segfault is caught with a useful message.
signal.signal(signal.SIGSEGV, _sigsegv_handler)

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

# RapidOCR (PP-OCRv5 mobile det + PP-OCRv5-EN mobile rec, via ONNXRuntime)
# — PRIMARY engine.
# Runs through ONNXRuntime with no PaddlePaddle framework involved at all,
# which sidesteps the CPU-only inference crash history (SIGSEGV, and a
# documented ~43GB OOM regression as recent as April 2026) that keeps the
# PaddleOCR engine below pinned to paddleocr==2.9.1/paddlepaddle==2.6.2.
# Verified against the actual production failure mode: a wide-tracked bold
# word ("HUNTER") that PP-OCRv4's detector split into six single-letter
# boxes (narrated "H U N T E R") stays one box here.
# The PP-OCRv5-mobile det+rec pair was picked by a hand-transcribed 10-panel
# bake-off (see scratchpad/bench/run_bench.py) over the stock PP-OCRv6
# det+rec, PP-OCRv4, and every v5/v6 mix — best word recall (0.99), best
# precision, best char-sim, and it recovers whole bubbles the v6 detector
# dropped. `RAPIDOCR_STOCK=1` reverts to the untuned RapidOCR() default.
# Post-OCR spelling repair + garbage removal is in _repair_and_denoise.
# PaddleOCR PP-OCRv4 (below) is kept as a fallback tier.
RAPIDOCR_MODEL_NAME = "RapidOCR-PPOCRv5mobile-EN"
rapidocr_engine = None  # type: Any
RAPIDOCR_READY = False  # type: bool
RAPIDOCR_ERROR = None  # type: Optional[str]


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
                # No tuning kwargs are passed here (unlike _run_ocr_on_image),
                # so a retry with the identical call/args can never behave
                # differently — any TypeError is a real signature mismatch
                # and belongs to the outer `except Exception` below.
                _ = ocr_obj.predict(dummy_img)
            elif hasattr(ocr_obj, "ocr") and callable(getattr(ocr_obj, "ocr")):
                _ = ocr_obj.ocr(dummy_img)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info("Real inference warmup succeeded in %.2f ms", elapsed_ms)
        return True
    except Exception as exc:
        logger.error("Warmup inference failed: %s", exc, exc_info=True)
        return False


def _init_rapidocr() -> None:
    """Initialize RapidOCR (PP-OCRv5 mobile det + PP-OCRv5-EN mobile rec, ONNXRuntime
    backend) — the PRIMARY OCR engine.
    Mirrors _init_ocr()'s structure (init lock, real-inference
    warmup before trusting the engine) but has no retry/backoff loop of
    its own here: it's covered by the same _background_retry_loop as
    PaddleOCR, at module scope below.
    """
    global rapidocr_engine, RAPIDOCR_READY, RAPIDOCR_ERROR

    with _init_lock:
        RAPIDOCR_READY = False
        RAPIDOCR_ERROR = None
        try:
            from rapidocr import RapidOCR
            # Detection + recognition head: PP-OCRv5 mobile (det) + the
            # dedicated English PP-OCRv5 mobile rec (`en_PP-OCRv5_rec_mobile`).
            # Chosen by a 10-panel hand-transcribed bake-off (see
            # scratchpad/bench): word RECALL 0.99 vs 0.97 for the stock
            # PP-OCRv6 det+rec, precision 0.97 vs 0.96, char-sim 0.98 vs 0.97
            # — it wins every metric AND recovers whole speech bubbles the v6
            # detector missed (the "missing text" problem). It must be the
            # matched v5 *mobile* family: v5 SERVER detection over-segments and
            # scrambles reading order, and a v5-en rec bolted onto a v6
            # detector (tried earlier) underperforms. `RAPIDOCR_STOCK=1`
            # reverts to the untuned RapidOCR() default.
            engine = None
            if not os.environ.get("RAPIDOCR_STOCK"):
                try:
                    from rapidocr import LangRec, ModelType, OCRVersion
                    engine = RapidOCR(params={
                        "Det.ocr_version": OCRVersion.PPOCRV5,
                        "Det.model_type": ModelType.MOBILE,
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                        "Rec.model_type": ModelType.MOBILE,
                        "Rec.lang_type": LangRec.EN,
                    })
                except Exception as _e:
                    logger.warning("RapidOCR PP-OCRv5 mobile unavailable (%s) — using stock", _e)
            if engine is None:
                engine = RapidOCR()
            # Real inference warmup, not just "the constructor didn't
            # raise" — same reasoning as _run_warmup below.
            dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
            with _inference_lock:
                engine(dummy_img)
            rapidocr_engine = engine
            RAPIDOCR_READY = True
            logger.info(
                "RapidOCR initialized and warmed up successfully (%s, ONNXRuntime)",
                "PP-OCRv6 stock det+rec" if os.environ.get("RAPIDOCR_STOCK")
                else "PP-OCRv5 mobile det + PP-OCRv5-EN mobile rec",
            )
        except Exception as exc:
            rapidocr_engine = None
            RAPIDOCR_READY = False
            RAPIDOCR_ERROR = str(exc)
            logger.error("RapidOCR initialization failed: %s", exc, exc_info=True)


def _init_ocr() -> None:
    """Attempt to initialise PaddleOCR PP-OCRv4 — now the FALLBACK tier
    (see RapidOCR/_init_rapidocr() above for the primary engine), kept on
    the pinned paddleocr==2.9.1 / paddlepaddle==2.6.2 line — see
    requirements.txt for why; PP-OCRv5/v6 do not exist on this line
    natively, so they are not attempted here.

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
                    # Only takes effect as a fallback if _run_ocr_on_image's
                    # per-call text_det_unclip_ratio kwarg is ever rejected
                    # (predict() TypeError -> falls back to a bare
                    # ocr.predict(img) with no kwargs at all) — the normal
                    # path is controlled by OCROptions.det_db_unclip_ratio
                    # (see that field) and by pipeline-service/lib.ts, which
                    # already deliberately sends 2.4 for manhwa/manhua's
                    # bold, wide-tracked hand-lettered text (see
                    # OCR_TUNING_VERSION comment there). Matching that same
                    # value here instead of guessing an independent number
                    # keeps the rare fallback path consistent with the
                    # already-validated primary path rather than silently
                    # reverting to PaddleOCR's generic document-tuned
                    # default (~1.5) if the kwargs path ever breaks.
                    "det_db_unclip_ratio": 2.4,
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
            cand_ocr = _try_init("PP-OCRv4")
            cand_name = "PP-OCRv4"
            logger.info("PP-OCRv4 loaded constructor successfully")
        except Exception as exc_v4:
            logger.error("PP-OCRv4 failed to initialise: %s", exc_v4)
            ocr = None
            MODEL_NAME = "unknown"
            MODEL_READY = False
            SERVICE_STATE = ServiceState.FAILED
            INIT_ERROR = f"PP-OCRv4 error: {exc_v4}"
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


# Run initialisation at module load so a model is ready before requests
# arrive. Both engines are attempted independently — the service overall
# is READY as long as AT LEAST ONE works, since either alone can serve
# requests (see _ocr_with_cascade, which tries RapidOCR first and falls
# through to PaddleOCR PP-OCRv4 only if RapidOCR is unavailable or
# uncertain).
def _active_model_name() -> str:
    """Name of whichever engine would actually serve the next request:
    RapidOCR (primary) if ready, else PaddleOCR (fallback) if ready, else
    'unknown'. Used everywhere a response reports which model is active,
    so it never falls back to reporting the PaddleOCR-only MODEL_NAME
    even when RapidOCR is the one actually serving requests.
    """
    if RAPIDOCR_READY:
        return RAPIDOCR_MODEL_NAME
    if MODEL_READY:
        return MODEL_NAME
    return "unknown"


def _recompute_service_state() -> None:
    global SERVICE_STATE, INIT_ERROR
    if RAPIDOCR_READY or MODEL_READY:
        SERVICE_STATE = ServiceState.READY
        INIT_ERROR = None
    else:
        SERVICE_STATE = ServiceState.FAILED
        INIT_ERROR = f"rapidocr: {RAPIDOCR_ERROR}; paddleocr: {INIT_ERROR}"


_SKIP_OCR_INIT = os.environ.get("SKIP_OCR_INIT") == "1"
if not _SKIP_OCR_INIT:
    _init_rapidocr()
    _init_ocr()  # _init_ocr() sets SERVICE_STATE itself; reconcile below
    _recompute_service_state()

# If startup init failed outright for BOTH engines, run background retry
# loop until at least one comes up. Never under SKIP_OCR_INIT (test mode):
# the retry thread would wake after 60s and run the REAL model init anyway,
# silently loading heavy engines into the module and polluting any test
# session that outlasts the backoff.
if not _SKIP_OCR_INIT and SERVICE_STATE != ServiceState.READY:
    def _background_retry_loop():
        # type: () -> None
        backoff_sec = 60
        max_backoff_sec = 600
        while SERVICE_STATE != ServiceState.READY:
            time.sleep(backoff_sec)
            logger.info("Retrying OCR engine initialisation in background (current state: %s)...", SERVICE_STATE)
            if not RAPIDOCR_READY:
                _init_rapidocr()
            if not MODEL_READY:
                _init_ocr()
            _recompute_service_state()
            if SERVICE_STATE != ServiceState.READY:
                backoff_sec = min(backoff_sec * 2, max_backoff_sec)

    threading.Thread(target=_background_retry_loop, daemon=True).start()
    logger.warning("No OCR engine ready at startup (state: %s) — background retry loop started", SERVICE_STATE)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OCR Service",
    description="OCR engine for manhwa/manga recap pipeline. Primary: RapidOCR (PP-OCRv5 mobile det+rec, ONNXRuntime) + post-OCR spelling repair. Fallback: PaddleOCR PP-OCRv4.",
    version="2.0.0",
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
        default=0.4,
        ge=0.1,
        le=0.9,
        description="Box score threshold for DB detector. 0.4 (was 0.5) after "
                    "a param sweep on PP-OCRv5 — lifts word recall ~2pts with "
                    "no precision cost; unclip 1.8 stays optimal (2.2 merges lines).",
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
    rapidocr_ready: bool = Field(default=False, description="Whether the primary RapidOCR (PP-OCRv5) engine is ready")
    paddleocr_ready: bool = Field(default=False, description="Whether the fallback PaddleOCR (PP-OCRv4) engine is ready")


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


try:
    import wordninja as _wordninja
    _WORDCOST = getattr(_wordninja.DEFAULT_LANGUAGE_MODEL, "_wordcost", {})
except Exception:  # pragma: no cover
    _wordninja = None
    _WORDCOST = {}


def _is_dict_word(w):
    # type: (str) -> bool
    wl = w.lower()
    return wl in ("a", "i") or (len(wl) >= 2 and wl in _WORDCOST)


# --- OCR spelling repair + garbage denoise -------------------------------
# Fix genuine mis-recognitions ("dunngoeoon" -> "dungeon", "absollte" ->
# "absolute") WITHOUT paraphrasing, drop pure garbage the detector
# hallucinated out of texture/hatching ("OO NN N WN T T R", "^^^"), and
# KEEP sound effects verbatim ("FWOOSH", "BOOOM", "KRA-KOOM").
try:
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import Indel as _rf_Indel, JaroWinkler as _rf_JW
except Exception:  # pragma: no cover
    _rf_process = None

# Onomatopoeia / vocal noises that show up in these comics — never
# "corrected" to a dictionary word and never dropped as garbage.
_SFX_WORDS = {
    "boom", "booom", "kaboom", "kraboom", "krakoom", "bang", "crash", "krash",
    "crack", "krak", "crackle", "smash", "slam", "wham", "bam", "pow", "thud",
    "thump", "thunk", "clang", "clank", "clink", "clunk", "ding", "dong", "buzz",
    "bzzt", "hiss", "sizzle", "fizz", "hum", "rumble", "roar", "growl", "snarl",
    "screech", "shriek", "splash", "sploosh", "drip", "plop", "splat", "swish",
    "swoosh", "fwoosh", "vwoosh", "whoosh", "hwoosh", "woosh", "swoop", "flash",
    "poof", "puff", "tap", "rap", "knock", "click", "clack", "clatter", "rustle",
    "crunch", "stomp", "thwack", "whack", "smack", "slap", "grr", "grrr", "argh",
    "gah", "ugh", "gasp", "pant", "huff", "sigh", "groan", "moan", "gulp", "slurp",
    "beep", "boop", "ring", "brring", "tick", "tock", "vroom", "zoom", "zap", "zip",
    "shing", "clash", "twang", "boing", "sproing", "rattle", "shatter", "whir",
    "whirr", "fwip", "fwsh", "shff", "shf", "ksss", "fwoom", "vwoom", "krsh",
    "tmp", "thmp", "step", "steps", "screee", "skrrt", "nyoom", "fwm", "vwm",
    # laughter + vocal reactions (2 distinct letters, so the garbage filter
    # would otherwise eat them)
    "haha", "hahaha", "hahahaha", "hehe", "hehehe", "heehee", "hoho", "hohoho",
    "muahaha", "mwahaha", "bwahaha", "kekeke", "heh", "hah", "huh", "hmph",
    "hmm", "hmmm", "mmm", "tch", "tsk", "pfft", "psst", "shh", "shhh", "aha",
    "aah", "ahh", "ooh", "ohh", "eek", "whew", "phew", "uwah", "waah", "wah",
    "gwah", "kya", "kyaa", "nng", "nngh", "hnng", "urgh", "blegh", "ack", "gack",
    "humph", "hmph", "hmp", "harrumph", "pff", "pfft", "meh", "bah", "psh", "feh",
}
_SFX_TAIL_RE = re.compile(r"^[A-Z]*(?:SH|OSH|OOSH|OM|OOM|NG|ANG|ONG|CK|MP|ZZ|RR)$")
_SFX_SYLLABLE_RE = re.compile(r"(?:ha|he|hi|ho|hu|ja|ka|ke|na|la|da|ba|wa|nya|mwa|bwa)+$", re.I)


def _is_probable_sfx(tok):
    # type: (str) -> bool
    core = re.sub(r"[^A-Za-z]", "", tok)
    if len(core) < 2:
        return False
    if core.lower() in _SFX_WORDS:
        return True
    # a stretched letter is the signature of a shout / crash ("BOOOM",
    # "AAARGH", "GRRR", "NOOO", "HMMM")
    if re.search(r"(.)\1{2,}", core):
        return True
    # repeated CV syllable = laughter / chant ("HAHA", "NANANA", "KEKEKE")
    if len(core) >= 4 and _SFX_SYLLABLE_RE.fullmatch(core.lower()):
        return True
    # short all-caps blob ending like an impact sound, not a real word
    if core.isupper() and 3 <= len(core) <= 9 and _SFX_TAIL_RE.match(core) \
            and not _is_dict_word(core):
        return True
    return False


_REAL_SHORT_WORDS = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if",
    "in", "is", "it", "me", "my", "no", "of", "oh", "ok", "on", "or", "ox", "so",
    "to", "up", "us", "we", "ye", "ah", "ha", "um", "ow", "eh", "yo", "aw", "mr",
    "ms", "dr",
}
_DICT_BUCKETS = {}  # len -> [candidate words], built lazily
_SPELL_CACHE = {}


def _spell_candidates(n):
    # type: (int) -> list
    b = _DICT_BUCKETS.get(n)
    if b is None:
        # OCR mis-recognitions add stray characters far more often than they
        # drop them ("dunngoeoon" is 3 longer than "dungeon"), so the window
        # reaches further BELOW the observed length than above.
        lo, hi = n - 4, n + 2
        b = [w for w, c in _WORDCOST.items()
             if lo <= len(w) <= hi and c <= 13.6 and w.isalpha()]
        _DICT_BUCKETS[n] = b
    return b


def _correct_token(tok):
    # type: (str) -> str
    """Return a corrected spelling for a single OCR token, or the token
    unchanged. Only fires on a clearly non-word of length >= 5 that has a
    very close real-word neighbour — so character names and sound effects
    (no close dictionary neighbour) pass straight through. Hyphenated /
    apostrophe'd tokens (stutters "M-MOVE", compounds "LOW-TIER",
    contractions) are left alone."""
    if _rf_process is None or not _WORDCOST:
        return tok
    if "-" in tok or "'" in tok or "’" in tok:
        return tok
    core = re.sub(r"[^A-Za-z]", "", tok)
    if len(core) < 5 or any(ch.isdigit() for ch in tok):
        return tok
    if _is_dict_word(core) or _is_probable_sfx(tok):
        return tok
    key = core.lower()
    if key in _SPELL_CACHE:
        cand = _SPELL_CACHE[key]
    else:
        cand = None
        best = _rf_process.extractOne(
            key, _spell_candidates(len(key)),
            scorer=_rf_Indel.normalized_similarity, score_cutoff=0.80)
        if best:
            w = best[0]
            if abs(len(w) - len(key)) <= max(3, len(key) // 2):
                indel = best[1]
                jw = _rf_JW.normalized_similarity(key, w)
                if indel >= 0.88 or (indel >= 0.80 and jw >= 0.88):
                    cand = w
        _SPELL_CACHE[key] = cand
    if not cand:
        return tok
    if core.isupper():
        repl = cand.upper()
    elif core[:1].isupper():
        repl = cand.capitalize()
    else:
        repl = cand
    return tok.replace(core, repl, 1)


_SINGLE_LETTER_RUN_RE = re.compile(
    r"(?:(?<![\w'’\-])[B-HJ-Zb-hj-z](?![\w'’\-])(?:\s+|,\s*)?){2,}")
_SYMBOL_RUN_RE = re.compile(r"(?<![.!?])([^\w\s.!?'\"()\-’–—])\1{1,}")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")


def _repair_and_denoise(text):
    # type: (str) -> str
    """Final pass over a merged panel transcription: strip hallucinated
    garbage the detector read out of texture/hatching, fix the spelling of
    genuine mis-recognitions, and keep sound effects + dialogue verbatim.
    No rephrasing — word order and wording are never changed."""
    if not text or not text.strip():
        return text or ""
    t = text
    # Is this panel's lettering essentially ALL-CAPS (the norm for these
    # comics)? If so, a stray all-lowercase blob ("xgex", "ina") is texture
    # the detector mis-read, not dialogue — real lowercase words would be
    # rare and are still protected by the dict-word check below.
    _upper = sum(c.isupper() for c in t)
    _lower = sum(c.islower() for c in t)
    allcaps_panel = _upper >= 6 and _upper >= _lower * 4
    # raw-aggregator site watermark stamped into the panel ("www.baozimh.con",
    # often with the TLD mis-read). Strip just the URL token.
    t = re.sub(r"\b(?:https?://)?www\.\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b[a-z][a-z0-9-]{2,}\.(?:com|net|org|con|c0m|xyz|top|io)\b(?=$|\s|[.,!?])",
               " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:baozimh|mangabuddy|manhuaplus|manhuafast|asuracomic|flamescans?)\b",
               " ", t, flags=re.IGNORECASE)
    # a caret / backtick / lone star BETWEEN two letters is a mangled
    # apostrophe ("CAN^T" -> "CAN'T"), not a symbol run
    t = re.sub(r"([A-Za-z])[\^`*]([A-Za-z])", r"\1'\2", t)
    # runs of isolated single consonants ("O N N W N", "T T R") = texture noise
    t = _SINGLE_LETTER_RUN_RE.sub(" ", t)
    # runs of repeated punctuation / symbols ("^^^", "~~", "***") — but leave
    # "...", "!!", "?!" alone
    t = _SYMBOL_RUN_RE.sub(" ", t)
    t = re.sub(r"[|_~^`<>{}\[\]\\]+", " ", t)

    out = []
    for tok in t.split():
        core = re.sub(r"[^A-Za-z0-9]", "", tok)
        if not core:
            if re.fullmatch(r"(?:\.{2,}|!+|\?+|[!?]{2,}|[-–—]+|,)", tok):
                out.append("..." if tok.startswith("..") else tok)
            continue
        # a real dictionary word is always kept as-is (protects "TOO",
        # "SEE", "ALL", "OFF" from the low-distinct-letter garbage rule).
        # wordninja's list is polluted with 2-letter corpus cruft ("oo",
        # "nn", "mm"), so require length >= 3 unless it's a genuine short
        # word / interjection.
        if (len(core) >= 3 and _is_dict_word(core)) or core.lower() in _REAL_SHORT_WORDS:
            out.append(tok)
            continue
        if _is_probable_sfx(tok):
            out.append(tok)
            continue
        letters = re.sub(r"[^A-Za-z]", "", core)
        nset = len(set(letters.lower()))
        # --- pure garbage the detector hallucinated ---
        if len(letters) >= 2 and nset == 1:                       # "OO", "NNN"
            continue
        if len(letters) >= 4 and nset == 2 and not _VOWEL_RE.search(letters):  # "WNWN"
            continue
        if 0 < len(letters) <= 4 and not _VOWEL_RE.search(letters) \
                and letters.upper() not in ("MR", "MRS", "DR", "ST", "TV", "HP", "MP"):
            continue                                              # "WN", "TTR"
        if len(core) == 1 and core not in ("I", "A", "a") and not core.isdigit():
            continue
        # lowercase shard in an otherwise all-caps panel = mis-read texture
        if allcaps_panel and core.isalpha() and core.islower() and len(core) <= 6 \
                and not _is_dict_word(core):
            continue
        out.append(_correct_token(tok))

    # a trailing junk token whose case clashes with an otherwise all-caps
    # line ("...THAT IS?  Djinni", "...HAIL!  inen") is the watermark strip /
    # texture the detector tacked onto the end — drop it
    if len(out) >= 3:
        body_upper = sum(c.isupper() for c in " ".join(out[:-1]))
        body_lower = sum(c.islower() for c in " ".join(out[:-1]))
        last = re.sub(r"[^A-Za-z]", "", out[-1])
        if (body_upper >= 6 and body_upper >= body_lower * 3 and 3 <= len(last) <= 8
                and not last.isupper() and not _is_dict_word(last)
                and not _is_probable_sfx(out[-1])):
            out = out[:-1]

    t = " ".join(out)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    # a panel whose ENTIRE transcription is one short lower-case non-word
    # ("winz", "inen", "ina") is the detector reading hatching/texture in an
    # art panel — no dialogue, drop it
    lone = re.sub(r"[^A-Za-z]", "", t)
    if t and lone == t.strip(".,!?:;'\"") and len(lone) <= 6 and lone.islower() \
            and not _is_dict_word(lone) and lone not in _REAL_SHORT_WORDS:
        return ""
    return t


def _desegment_runon(text):
    # type: (str) -> str
    """Split words the OCR glued together ("BURNEDWHOLE" -> "BURNED WHOLE",
    "THEINTO" -> "THE INTO"). Conservative: a token is only split when
    wordninja's segmentation is ENTIRELY real dictionary words — so proper
    names ("SHENYE", "DINGZHOU"), whose syllables aren't dictionary words,
    are left intact. Original casing/punctuation preserved by slicing the
    source token at the split lengths. A single junk consonant stuck to a
    real word ("RTHIS" -> "THIS") is dropped."""
    if _wordninja is None or not text or not _WORDCOST:
        return text

    def _cost(w):
        return _WORDCOST.get(w.lower(), 99.0)

    def _fix(m):
        tok = m.group(0)
        if len(tok) < 5 or _is_dict_word(tok):
            return tok
        parts = _wordninja.split(tok)
        if len(parts) < 2 or sum(len(p) for p in parts) != len(tok):
            return tok
        # drop a leading 1-char junk shard glued to a real word ("R"+"THIS")
        if len(parts[0]) == 1 and parts[0].lower() not in ("a", "i") and _is_dict_word(parts[1]):
            tok, parts = tok[1:], parts[1:]
            if len(parts) == 1:
                return tok
        if not all(_is_dict_word(p) for p in parts):
            return m.group(0)
        # Guard against splitting a romanised NAME whose syllables happen to
        # be rare dictionary words ("SHENYE" -> "SHE NYE", "DINGZHOU"): only
        # split a short token when EVERY part is a genuinely common word.
        if len(tok) < 10 and any(_cost(p) > 9.5 for p in parts):
            return m.group(0)
        out, i = [], 0
        for p in parts:
            out.append(tok[i:i + len(p)])
            i += len(p)
        return " ".join(out)

    return re.sub(r"[A-Za-z]{5,}", _fix, text)


def _trim_leading_noise(text):
    # type: (str) -> str
    """Drop a run of short vowel-less letter shards at the very start of a
    line ("WN T T R THIS FLAME'S..." -> "THIS FLAME'S...") — leftover flame/
    texture the detector read as letters and glued onto the real sentence."""
    toks = text.split()
    i = 0
    while i < len(toks) - 1:
        core = re.sub(r"[^A-Za-z]", "", toks[i])
        if core in ("I", "A", "a"):
            break
        if len(core) <= 3 and not re.search(r"[aeiouyAEIOUY]", core):
            i += 1
            continue
        break
    # A single leading shard is far more likely a real (mis-OCR'd) word than
    # texture noise — only trim a RUN of them ("WN T T R" -> 4 shards).
    return " ".join(toks[i:]) if i >= 2 else text


def _clean_and_normalize_ocr_text(text: str) -> str:
    """Normalize ellipses, punctuation, character substitutions, and end cards."""
    if not text:
        return ""
    t = _trim_leading_noise(_desegment_runon(text))
    # OCR frequently drops the space after mid-sentence punctuation when two
    # bubbles are read in one pass ("SPELL,WANG" -> "SPELL, WANG"). Safe: an
    # apostrophe/decimal is a letter-adjacent case we exclude.
    t = re.sub(r'([A-Za-z]{2}),([A-Za-z]{2})', r'\1, \2', t)
    t = re.sub(r'([A-Za-z]{3})([!?])([A-Za-z]{2})', r'\1\2 \3', t)
    t = re.sub(r'\s*\b(minus|dash|underscore)\b\s*$', '...', t, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)

    t = re.sub(r'\bHO[0O]\b', 'HOO', t)
    t = re.sub(r'\bHO\s+O\b', 'HOO', t)
    t = re.sub(r'\bgood-curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)
    t = re.sub(r'\bgood\s+curdling\b', 'blood-curdling', t, flags=re.IGNORECASE)

    t = re.sub(r'\bB\s+to\s+be\s+continued\.*', 'To Be Continued...', t, flags=re.IGNORECASE)
    t = re.sub(r'^\s*B\s+to\s+be\b(?!\s+continued)', 'To Be Continued', t, flags=re.IGNORECASE)
    t = re.sub(r'\.{2,}', '...', t)

    # Split contractions the OCR broke on the apostrophe ("ISN T" -> "ISN'T",
    # "WE RE" -> "WE'RE", "I LL" -> "I'LL"). Without this the orphaned
    # "T"/"RE"/"LL" gets swept away as a stray shard downstream, turning
    # "isn't" into "isn". Casing of the join follows the surrounding text.
    def _rejoin(m):
        joined = (m.group(1) + m.group(2)).replace(" ", "")
        out = m.group(1).rstrip() + "'" + m.group(2).lstrip()
        if joined.isupper():
            return out.upper()
        if joined.islower():
            return out.lower()
        return out

    t = re.sub(r"\b((?:is|was|wer|were|are|has|have|had|does|did|do|would|should|"
               r"could|ca|wo|ai|might|must|need|dare)n)(\s+t)\b", _rejoin, t, flags=re.IGNORECASE)
    t = re.sub(r"\b(we|you|they)(\s+re)\b", _rejoin, t, flags=re.IGNORECASE)
    t = re.sub(r"\b(i|we|you|they|he|she|it|that|there|who|what)(\s+(?:ll|ve|d))\b",
               _rejoin, t, flags=re.IGNORECASE)
    t = re.sub(r"\b(he|she|it|that|there|what|who|here|one|thing)(\s+s)\b",
               _rejoin, t, flags=re.IGNORECASE)
    t = re.sub(r"\b(i)(\s+m)\b", _rejoin, t, flags=re.IGNORECASE)
    # "I L TAKE" / "YOU L SEE" — OCR dropped one L of "'LL"
    t = re.sub(r"\bI\s+L\b(?=\s+[A-Z])", "I'LL", t)
    t = re.sub(r"\b(You|We|They|He|She)\s+l\b(?=\s+[a-z])", r"\1'll", t)

    return t


def _sort_regions_reading_order(regions, is_ui_box=False):
    # type: (List[_TextRegion], bool) -> List[_TextRegion]
    """Order detected text regions in natural LTR reading order.

    1. Recursively split on a clean VERTICAL gutter that no region crosses —
       separates side-by-side speech bubbles / columns, so each bubble is
       read fully before the next (was being interleaved line-by-line).
    2. Inside a column, group regions into visual LINES by vertical OVERLAP
       (not y_min proximity — an ascender/descender or a tall glyph next to
       a short one used to throw two words of ONE line into different rows,
       which then got emitted out of order, e.g. "2nd word ... 1st word").
    3. Lines top-to-bottom, words within a line left-to-right.
    """
    if not regions:
        return regions
    if is_ui_box or len(regions) == 1:
        return sorted(regions, key=lambda r: (r.y_min, r.x_min))

    heights = sorted(max(1.0, r.y_max - r.y_min) for r in regions)
    med_h = heights[len(heights) // 2]

    def _group_lines(regs):
        # Group regions into printed lines. Two regions are on the SAME line
        # when they overlap vertically AND sit side-by-side horizontally
        # (little/no x-overlap). Two STACKED lines of one bubble also overlap
        # vertically — big webtoon lettering has tight leading, so lines
        # routinely overlap 40-50% — but they overlap horizontally too, which
        # is the discriminator that stops them being merged and their words
        # x-sorted out of order.
        rem = sorted(regs, key=lambda r: (r.y_min, r.x_min))
        lines = []  # list of dicts {members, y1, y2}
        for r in rem:
            rh = max(1.0, r.y_max - r.y_min)
            rw = max(1.0, r.x_max - r.x_min)
            placed = False
            for ln in lines:
                v_ov = min(r.y_max, ln["y2"]) - max(r.y_min, ln["y1"])
                if v_ov / min(rh, ln["y2"] - ln["y1"]) < 0.45:
                    continue
                h_ov = min(r.x_max, ln["x2"]) - max(r.x_min, ln["x1"])
                if h_ov / min(rw, ln["x2"] - ln["x1"]) > 0.25:
                    continue  # stacked, not same line
                ln["members"].append(r)
                ln["y1"] = min(ln["y1"], r.y_min); ln["y2"] = max(ln["y2"], r.y_max)
                ln["x1"] = min(ln["x1"], r.x_min); ln["x2"] = max(ln["x2"], r.x_max)
                placed = True
                break
            if not placed:
                lines.append({"members": [r], "y1": r.y_min, "y2": r.y_max,
                              "x1": r.x_min, "x2": r.x_max})
        lines.sort(key=lambda ln: (ln["y1"] + ln["y2"]) / 2.0)
        out = []
        for ln in lines:
            out.extend(sorted(ln["members"], key=lambda r: r.x_min))
        return out

    def _cluster_bubbles(regs):
        # When no clean guillotine cut exists (diagonally-placed speech
        # bubbles overlap in BOTH x and y), group regions into bubbles by
        # spatial proximity, order the bubbles top-to-bottom / left-to-right,
        # then line-group within each bubble.
        gap = med_h * 1.6
        parent = list(range(len(regs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                dx = max(a.x_min - b.x_max, b.x_min - a.x_max, 0.0)
                dy = max(a.y_min - b.y_max, b.y_min - a.y_max, 0.0)
                if dx <= gap and dy <= gap:
                    parent[find(i)] = find(j)

        groups = {}
        for i in range(len(regs)):
            groups.setdefault(find(i), []).append(regs[i])
        clusters = list(groups.values())
        if len(clusters) <= 1:
            return _group_lines(regs)

        def key(cl):
            y0 = min(r.y_min for r in cl)
            x0 = min(r.x_min for r in cl)
            return (round(y0 / max(1.0, med_h * 1.5)), x0)  # rows of bubbles, L->R

        clusters.sort(key=key)
        out = []
        for cl in clusters:
            out.extend(_group_lines(cl))
        return out

    def _split(regs, depth=0):
        if len(regs) <= 1 or depth > 40:
            return _group_lines(regs)
        span_w = max(r.x_max for r in regs) - min(r.x_min for r in regs)
        xs = sorted(regs, key=lambda r: r.x_min)
        cur_x2 = xs[0].x_max
        v_gap, v_at = 0.0, None
        for r in xs[1:]:
            g = r.x_min - cur_x2
            if g > v_gap:
                v_gap, v_at = g, (cur_x2 + r.x_min) / 2.0
            cur_x2 = max(cur_x2, r.x_max)
        if v_at is not None and v_gap >= max(med_h * 1.4, span_w * 0.09):
            left = [r for r in regs if (r.x_min + r.x_max) / 2.0 < v_at]
            right = [r for r in regs if (r.x_min + r.x_max) / 2.0 >= v_at]
            if left and right:
                return _split(left, depth + 1) + _split(right, depth + 1)
        return _cluster_bubbles(regs)

    return _split(list(regions))


def _looks_like_ocr_noise(text):
    # type: (str) -> bool
    """A detected 'region' that is really flame/speed-line/texture the
    detector hallucinated letters out of: single stray chars, all-caps
    consonant clusters, or a run of 1-2 char fragments ("OO NN N WN T T R").
    Deliberately narrow — real 1-2 letter words (I, a, ok, no) never trip it
    because they are single tokens, not runs, and they contain vowels."""
    s = (text or "").strip()
    if not s:
        return True
    if len(s) <= 1:
        return s not in ("I", "A", "a")
    if re.fullmatch(r"(.)\1{2,}", s):                       # "OOOO", "!!!!"
        return True
    toks = [t for t in re.split(r"\s+", s) if t]
    letters_only = re.sub(r"[^A-Za-z]", "", s)
    if not letters_only:
        return True
    # a run of short fragments, none of which is a real little word
    if len(toks) >= 2 and all(len(re.sub(r"[^A-Za-z]", "", t)) <= 2 for t in toks):
        if not any(t.lower() in ("i", "a", "an", "as", "at", "be", "is", "it", "no",
                                 "of", "oh", "ok", "on", "or", "so", "to", "up", "us",
                                 "we", "ah", "hi", "ha", "ho", "uh", "um", "my")
                   for t in toks):
            return True
    # a single short vowel-less alpha blob ("NNW", "TTR", "WN")
    if len(letters_only) <= 4 and not re.search(r"[aeiouyAEIOUY]", letters_only):
        return True
    return False


def _merge_regions(regions, is_ui_box=False, engine="paddleocr"):
    # type: (List[_TextRegion], bool, str) -> Tuple[str, float, int]
    """Merge sorted text regions into a single coherent string.

    `engine` controls the no-space join heuristic: PP-OCRv4 over-segments
    bold lettering into per-glyph boxes, RapidOCR's PP-OCRv5 detector does
    not — so for RapidOCR any positive gap between boxes is a word boundary."""
    if not regions:
        return "", 0.0, 0

    # Filter detector noise, but never down to nothing.
    filtered = [r for r in regions if not _looks_like_ocr_noise(r.text)]
    if filtered:
        regions = filtered

    sorted_regions = _sort_regions_reading_order(regions, is_ui_box=is_ui_box)

    # Walk the ALREADY-ORDERED regions and join them. Default: one space
    # between regions. NO space only for a genuine same-line glyph split —
    # two boxes on the same printed line that physically touch/overlap, or
    # (PP-OCRv4 only) two 1-2 char fragments abutting each other ("H" "U"
    # "N" "T" "E" "R"). A large NEGATIVE x-gap means the next region is a new
    # line that starts further left — that is a word boundary and MUST get a
    # space ("...FOREST" / "HAS BEEN..." was becoming "FORESTHAS").
    parts = []  # type: List[str]
    all_confidences = []  # type: List[float]
    prev = None
    for r in sorted_regions:
        t = _clean_and_normalize_ocr_text(r.text.strip())
        if not t:
            continue
        if prev is not None:
            gap = r.x_min - prev.x_max
            char_h = max(1.0, ((r.y_max - r.y_min) + (prev.y_max - prev.y_min)) / 2.0)
            v_ov = min(r.y_max, prev.y_max) - max(r.y_min, prev.y_min)
            same_line = v_ov / max(1.0, min(r.y_max - r.y_min, prev.y_max - prev.y_min)) > 0.45
            prev_t = parts[-1].strip()
            touching = same_line and (-char_h * 0.30 < gap < char_h * 0.08)
            glyph_split = (
                engine != "rapidocr" and same_line and gap < char_h * 0.6
                and len(re.sub(r"[^A-Za-z]", "", prev_t)) <= 2
                and len(re.sub(r"[^A-Za-z]", "", t)) <= 2
            )
            parts.append(t if (touching or glyph_split) else " " + t)
        else:
            parts.append(t)
        prev = r
        if r.confidence > 0:
            all_confidences.append(r.confidence)

    merged_text = _repair_and_denoise(_clean_and_normalize_ocr_text("".join(parts)))
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return merged_text, round(avg_confidence, 4), len(sorted_regions)


def _get_field(obj, name):
    """Safely fetch `name` from a PaddleX/PaddleOCR result object via
    attribute or dict-key access, without ever evaluating the truthiness
    of the returned value.

    The obvious `getattr(obj, name, None) or obj.get(name, default)`
    fallback chain forces Python's `or` to evaluate `bool()` on whatever
    getattr() returns first — but real PaddleOCR/PaddleX results
    routinely hand back rec_scores and rec_boxes as multi-element numpy
    arrays (confirmed directly against PaddleOCR's own documented output
    samples, e.g. `'rec_scores': array([0.984..., 0.980...])` and
    `'rec_boxes': array([[3, 10, 82, 33], ...])`), and `bool()` on any
    numpy array with more than one element raises "The truth value of an
    array with more than one element is ambiguous" — turning a perfectly
    good detection into an uncaught exception that _ocr_with_cascade's
    outer try/except quietly converts to a FAILED result with empty text.
    Reproduced directly with a synthetic Result object carrying real
    numpy-array fields. Checking `is not None` instead never touches the
    array's contents, so it works identically whether the underlying
    value is a numpy array, a plain list, empty, or absent.
    """
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def parse_ocr_results(result):
    extracted_lines = []
    if not result:
        return extracted_lines

    page_res = result[0] if isinstance(result, list) and len(result) > 0 else result

    if hasattr(page_res, 'rec_texts') or (isinstance(page_res, dict) and 'rec_texts' in page_res):
        rec_texts = _get_field(page_res, 'rec_texts')
        rec_texts = [] if rec_texts is None else rec_texts
        rec_scores = _get_field(page_res, 'rec_scores')
        rec_scores = [] if rec_scores is None else rec_scores
        rec_boxes = _get_field(page_res, 'rec_boxes')
        if rec_boxes is None:
            rec_boxes = _get_field(page_res, 'dt_polys')
        rec_boxes = [] if rec_boxes is None else rec_boxes

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


def _run_rapidocr_on_image(img, options=None):
    # type: (np.ndarray, Optional[OCROptions]) -> List[_TextRegion]
    """Run RapidOCR (PP-OCRv5 mobile, ONNXRuntime) on a numpy image array.

    Mirrors _run_ocr_on_image's contract (same _TextRegion return shape)
    so the existing _merge_regions/_quality_status pipeline — including
    the gap-aware word-join fix for over-segmented bold lettering — works
    unchanged regardless of which engine produced the regions.
    """
    global rapidocr_engine
    if rapidocr_engine is None:
        raise RuntimeError("RapidOCR engine is not initialized")

    opts = options or OCROptions()
    with _inference_lock:
        result = rapidocr_engine(
            img,
            box_thresh=opts.det_db_box_thresh,
            unclip_ratio=opts.det_db_unclip_ratio,
        )

    regions = []  # type: List[_TextRegion]
    # RapidOCR returns boxes/txts/scores as None (not empty sequences) when
    # nothing is detected — verified directly against a blank test image.
    if result is None or result.boxes is None or result.txts is None:
        return regions

    scores = result.scores if result.scores is not None else [0.0] * len(result.boxes)
    for box, text, score in zip(result.boxes, result.txts, scores):
        if not text or not str(text).strip():
            continue
        try:
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
        except (TypeError, ValueError, IndexError):
            continue
        regions.append(_TextRegion(
            text=str(text), confidence=float(score),
            x_min=min(xs), y_min=min(ys), y_max=max(ys), x_max=max(xs),
        ))
    return regions


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
            if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                # Flat [xmin, ymin, xmax, ymax] format — this is exactly
                # what real PaddleOCR/PaddleX results use for 'rec_boxes'
                # (confirmed against PaddleOCR's own documented output,
                # e.g. 'rec_boxes': array([[3, 10, 82, 33], ...])), as
                # opposed to the 4-corner-point format 'dt_polys'/
                # 'rec_polys' use. The old code assumed every box was a
                # list of (x, y) points and did `pt[0] for pt in box`,
                # which on a flat box iterates over 4 bare numbers and
                # raises TypeError on the very first one ('int' object is
                # not subscriptable) — silently caught below and dropping
                # the region entirely. Reproduced directly: a real-shaped
                # rec_boxes detection came back with regions=0 despite
                # valid text and confidence.
                xs = [float(box[0]), float(box[2])]
                ys = [float(box[1]), float(box[3])]
            else:
                # Corner-point format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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
    """Run OCR through an engine cascade: RapidOCR (PP-OCRv5, PRIMARY) ->
    PaddleOCR PP-OCRv4 standard + preprocessing variants (FALLBACK) ->
    Tesseract (LAST RESORT). Every stage's non-empty candidate is tracked
    so the single best-quality result across ALL engines tried wins, even
    if none individually reached the SUCCESS confidence threshold — the
    same "keep the best candidate seen so far" contract the PaddleOCR
    preprocessing-variant loop already used, extended across engines
    rather than only across one engine's variants.

    Skips all further stages entirely the moment a stage finds ZERO text
    regions — no amount of a different engine or pixel-level
    preprocessing manufactures text regions that were never localized as
    a candidate bounding box in the first place, and this is the common
    case (manhwa chapters are frequently 80%+ silent/action panels with
    no dialogue at all).
    """
    try:
        opts = options or OCROptions()
        is_ui_box = _detect_ui_card_or_borders(img)
        if is_ui_box and opts.det_limit_side_len < 1216:
            opts = opts.copy(update={"det_limit_side_len": 1216})

        candidates = []  # type: List[dict]
        best_tuple = None  # type: Optional[Tuple[str, float, int, str, float, List[dict], str]]

        # --- PRIMARY: RapidOCR (PP-OCRv5 mobile, ONNXRuntime) ---
        if rapidocr_engine is not None:
            try:
                rapid_regions = _run_rapidocr_on_image(img, opts)
                rapid_text, rapid_conf, rapid_count = _merge_regions(rapid_regions, is_ui_box=is_ui_box, engine="rapidocr")
                rapid_status, rapid_quality, rapid_reason = _quality_status(rapid_text, rapid_conf, rapid_count)

                candidates.append({
                    "text": rapid_text, "confidence": rapid_conf, "regions": rapid_count,
                    "provider": "rapidocr", "model": RAPIDOCR_MODEL_NAME, "variant": "standard",
                })
                best_tuple = (rapid_text, rapid_conf, rapid_count, rapid_status, rapid_quality,
                              list(candidates), rapid_reason)

                if rapid_status == "SUCCESS":
                    return best_tuple
                if rapid_count == 0:
                    return (rapid_text, rapid_conf, rapid_count, rapid_status, rapid_quality,
                            list(candidates), f"skipped_fallback_zero_regions_detected:{rapid_reason}")
            except Exception as exc:
                logger.warning("RapidOCR pass failed (%s) — falling back to PaddleOCR PP-OCRv4", exc)

        # --- FALLBACK: PaddleOCR PP-OCRv4 (reached only if RapidOCR was
        # unavailable, errored, or came back UNCERTAIN with some text
        # found but not confidently). Wrapped in try/except (mirroring the
        # RapidOCR block above) rather than pre-checking `ocr is None`, so
        # a real "not initialized" failure here can't wipe out an
        # already-gathered RapidOCR candidate — it just falls through to
        # Tesseract with whatever best_tuple exists so far.
        try:
            regions = _run_ocr_on_image(img, opts)
            merged_text, avg_conf, region_count = _merge_regions(regions, is_ui_box=is_ui_box)
            status, quality_score, reason = _quality_status(merged_text, avg_conf, region_count)

            candidates.append({
                "text": merged_text, "confidence": avg_conf, "regions": region_count,
                "provider": "paddleocr", "model": MODEL_NAME, "variant": "standard",
            })

            fallback_tuple = (merged_text, avg_conf, region_count, status, quality_score, list(candidates), reason)
            if best_tuple is None or quality_score > best_tuple[4]:
                best_tuple = fallback_tuple

            if status == "SUCCESS":
                return best_tuple

            # Detector found ZERO candidate text regions in the original image —
            # this is the common case for action/establishing panels (manhwa
            # chapters are frequently 80%+ silent panels with no bubbles at
            # all). Upscaling/contrast/inversion tweak pixel values; they don't
            # manufacture text regions the detector never localized a bounding
            # box for in the first place, so running 3 more full inference
            # passes plus a 3-pass Tesseract fallback here is pure wasted
            # compute — it was making every quiet chapter (the majority of most
            # chapters) several times slower for no quality benefit.
            if region_count == 0:
                return (merged_text, avg_conf, region_count, status, quality_score,
                        list(candidates), f"skipped_cascade_zero_regions_detected:{reason}")

            preprocessing_passes = [
                ("upscale_1.5x", lambda i: _preprocess_upscale(i, 1.5)),
                ("contrast_clahe", _preprocess_contrast),
                ("color_inverted", _preprocess_invert),
            ]

            for variant_name, prep_fn in preprocessing_passes:
                prep_img = prep_fn(img)
                # Use `opts` (carries the UI-box det_limit_side_len bump) and
                # pass is_ui_box through to _merge_regions, same as the standard
                # pass above — using the original `options` here silently
                # reverted every fallback variant to non-UI-box thresholds,
                # which defeats the UI-box handling exactly when it's needed
                # most (the standard pass already failed to reach SUCCESS).
                var_regions = _run_ocr_on_image(prep_img, opts)
                var_text, var_conf, var_count = _merge_regions(var_regions, is_ui_box=is_ui_box)
                var_status, var_quality, var_reason = _quality_status(var_text, var_conf, var_count)

                candidates.append({
                    "text": var_text, "confidence": var_conf, "regions": var_count,
                    "provider": "paddleocr", "model": MODEL_NAME, "variant": variant_name,
                })

                if var_status == "SUCCESS" and var_quality > best_tuple[4]:
                    best_tuple = (
                        var_text, var_conf, var_count, var_status, var_quality,
                        list(candidates), f"paddleocr_variant_{variant_name}:{var_reason}",
                    )
                    return best_tuple
                elif var_quality > best_tuple[4]:
                    best_tuple = (
                        var_text, var_conf, var_count, var_status, var_quality,
                        list(candidates), f"paddleocr_variant_{variant_name}:{var_reason}",
                    )
        except Exception as exc:
            logger.warning("PaddleOCR fallback pass failed (%s)", exc)

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
                        list(candidates), f"tesseract_{tess_variant}_fallback_beat_prior_engines:{tess_reason}",
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
    """Return service health and model readiness status.

    `model` reports the active PRIMARY engine (RapidOCR) when it's ready,
    falling back to reporting the PaddleOCR fallback tier's name if only
    that one is up — `rapidocr_ready`/`paddleocr_ready` give the caller
    the precise per-engine picture either way.
    """
    active_model = _active_model_name()
    return HealthResponse(
        status="ok" if SERVICE_STATE == ServiceState.READY else ("initializing" if SERVICE_STATE == ServiceState.INITIALIZING else "degraded"),
        model=active_model,
        ready=(RAPIDOCR_READY or MODEL_READY),
        state=SERVICE_STATE,
        error=INIT_ERROR,
        rapidocr_ready=RAPIDOCR_READY,
        paddleocr_ready=MODEL_READY,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready_check():
    """Explicit readiness endpoint. Returns HTTP 200 if ready, HTTP 503 if not ready."""
    if SERVICE_STATE == ServiceState.READY:
        return ReadyResponse(
            status="ready",
            model=_active_model_name(),
            ready=True,
            state=SERVICE_STATE,
        )
    raise HTTPException(
        status_code=503,
        detail={
            "status": "not_ready",
            "model": _active_model_name(),
            "ready": False,
            "state": SERVICE_STATE,
            "error": INIT_ERROR,
        },
    )


@app.post("/reload")
async def reload_model():
    """Force a synchronous re-attempt at initialising both engines."""
    logger.info("Manual reload triggered on /reload endpoint")
    if not RAPIDOCR_READY:
        _init_rapidocr()
    if not MODEL_READY:
        _init_ocr()
    _recompute_service_state()
    active_model = _active_model_name()
    if SERVICE_STATE != ServiceState.READY:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "model": active_model, "state": SERVICE_STATE, "error": INIT_ERROR},
        )
    return {"ready": True, "model": active_model, "state": SERVICE_STATE,
            "rapidocr_ready": RAPIDOCR_READY, "paddleocr_ready": MODEL_READY}


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
    if SERVICE_STATE != ServiceState.READY:
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
        _active_model_name(),
    )

    return BatchOCRResponse(
        results=results,
        model=_active_model_name(),
        processing_time_ms=elapsed_ms,
    )


@app.post("/ocr/base64", response_model=Base64OCRResponse)
async def ocr_base64(request: Base64OCRRequest):
    """
    Accepts a single base64-encoded image and returns OCR transcription.
    """
    if SERVICE_STATE != ServiceState.READY:
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
        _active_model_name(),
    )

    return Base64OCRResponse(
        text=text,
        confidence=avg_conf,
        regions=region_count,
        status=status,
        quality_score=quality_score,
        candidates=candidates,
        selection_reason=reason,
        model=_active_model_name(),
        processing_time_ms=elapsed_ms,
    )


@app.post("/ocr", response_model=SingleOCRResponse)
async def ocr_single(request: Base64OCRRequest):
    """
    Legacy single-image OCR endpoint.
    """
    if SERVICE_STATE != ServiceState.READY:
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
        _active_model_name(),
    )

    return SingleOCRResponse(
        text=text,
        confidence=avg_conf,
        regions=region_count,
        status=status,
        quality_score=quality_score,
        candidates=candidates,
        selection_reason=reason,
        model=_active_model_name(),
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting PaddleOCR service on port 3002 ...")
    uvicorn.run(app, host="0.0.0.0", port=3002, workers=1)
