"""
PaddleOCR PP-OCRv4 Mini-Service

A FastAPI service providing high-accuracy OCR for manhwa/manga recap pipelines.
Uses PaddleOCR PP-OCRv4 (see requirements.txt for the paddleocr==2.9.1 /
paddlepaddle==2.6.2 pin and why) for text extraction from speech bubbles and
captions.

Port: 3002
"""

import json
import os
import signal
import sys
from pathlib import Path

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
import gc
import io
import logging
import queue
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
# F2: memory reclamation between batches.
#
# Over a long job (e.g. 328 chapters = 328 /ocr/batch calls) this service's RSS
# crept from ~100 MB to ~7 GB. It is NOT a true leak — Python frees the objects,
# but glibc's allocator holds the freed arenas instead of returning them to the
# OS, and torch/onnxruntime scratch buffers fragment the heap. `malloc_trim(0)`
# forces glibc to give the pages back; a `gc.collect()` first breaks any cycles
# holding large ndarrays/tensors. Called at the end of every batch — cheap
# (single-digit ms) next to a multi-second OCR batch.
# ---------------------------------------------------------------------------
import ctypes
import ctypes.util as _ctypes_util

try:
    _libc = ctypes.CDLL(_ctypes_util.find_library("c") or "libc.so.6", use_errno=True)
    _HAVE_MALLOC_TRIM = hasattr(_libc, "malloc_trim")
except Exception:  # pragma: no cover - non-glibc / exotic libc
    _libc = None
    _HAVE_MALLOC_TRIM = False


def _reclaim_memory(tag=""):
    """gc.collect() + glibc malloc_trim so RSS doesn't ratchet up over a long job."""
    try:
        collected = gc.collect()
        trimmed = False
        if _HAVE_MALLOC_TRIM:
            try:
                trimmed = bool(_libc.malloc_trim(0))
            except Exception:
                trimmed = False
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            logger.info("[mem] reclaim%s: gc=%d trim=%s peakRSS=%.0fMB",
                        f" {tag}" if tag else "", collected, trimmed, rss_mb)
        except Exception:
            pass
    except Exception as exc:
        logger.debug("[mem] reclaim failed: %s", exc)

# ---------------------------------------------------------------------------
# Service Readiness State & Synchronization Locks
# ---------------------------------------------------------------------------

class ServiceState:
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


_init_lock = threading.Lock()
# PaddleOCR ONLY — paddlepaddle 2.6.2's legacy-executor predictor is not
# thread-safe, so concurrent _run_ocr_on_image() calls must serialize here.
# RapidOCR (the primary engine) does NOT use this lock: it runs through a
# pool of independent engine instances (_rapidocr_pool) so OCR_CONCURRENCY
# calls execute in parallel.
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
rapidocr_engine = None  # type: Any   # first pool engine; kept for readiness checks
RAPIDOCR_READY = False  # type: bool
RAPIDOCR_ERROR = None  # type: Optional[str]

# RapidOCR engine pool. RapidOCR/ONNXRuntime inference is thread-safe across
# SEPARATE engine instances, so rather than serialize every call under one
# lock we keep a small pool and let OCR_CONCURRENCY calls run truly parallel.
# Bench, this 4-core box, 90 synthetic bubble frames:
#   1 engine / intra_op=1 ...... 375 ms/frame   (the old serialized path)
#   pool=3   / intra_op=1 ...... 203 ms/frame   (~1.85x)
# pool>3 and per-engine intra_op>1 both flatten out — the PP-OCRv5 *mobile*
# models are memory-bound, not compute-bound. Pool size tracks OCR_CONCURRENCY
# (override with RAPIDOCR_POOL_SIZE); ~50 MB of ONNX weights per engine.
RAPIDOCR_POOL_SIZE = max(1, int(os.environ.get(
    "RAPIDOCR_POOL_SIZE", os.environ.get("OCR_CONCURRENCY", "3"))))
_rapidocr_pool = None  # type: Any   # queue.Queue of RapidOCR instances


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
    global rapidocr_engine, RAPIDOCR_READY, RAPIDOCR_ERROR, _rapidocr_pool

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
            # ONNXRuntime intra-op threads PER engine. With the engine pool
            # below providing the cross-image parallelism, 1 thread/engine is
            # the sweet spot on this 4-core box (bench: pool=3/intra_op=1 at
            # 203 ms/frame beats pool=2/intra_op=2 at 221). Override with
            # RAPIDOCR_INTRA_OP_THREADS (0/-1 = let onnxruntime decide).
            _rapid_intra_op = int(os.environ.get("RAPIDOCR_INTRA_OP_THREADS", "1"))

            def _build_rapidocr():
                eng = None
                if not os.environ.get("RAPIDOCR_STOCK"):
                    try:
                        from rapidocr import LangRec, ModelType, OCRVersion
                        eng = RapidOCR(params={
                            "Det.ocr_version": OCRVersion.PPOCRV5,
                            "Det.model_type": ModelType.MOBILE,
                            "Rec.ocr_version": OCRVersion.PPOCRV5,
                            "Rec.model_type": ModelType.MOBILE,
                            "Rec.lang_type": LangRec.EN,
                            "EngineConfig.onnxruntime.intra_op_num_threads": _rapid_intra_op,
                        })
                    except Exception as _e:
                        logger.warning("RapidOCR PP-OCRv5 mobile unavailable (%s) — using stock", _e)
                if eng is None:
                    eng = RapidOCR()
                # Real inference warmup, not just "the constructor didn't raise"
                # — same reasoning as _run_warmup below.
                eng(np.zeros((32, 32, 3), dtype=np.uint8))
                return eng

            pool = queue.Queue()  # type: Any
            first = _build_rapidocr()
            pool.put(first)
            for _ in range(max(0, RAPIDOCR_POOL_SIZE - 1)):
                pool.put(_build_rapidocr())
            rapidocr_engine = first
            _rapidocr_pool = pool
            RAPIDOCR_READY = True
            logger.info(
                "RapidOCR initialized: pool of %d (%s, ONNXRuntime, intra_op=%d)",
                RAPIDOCR_POOL_SIZE,
                "PP-OCRv6 stock det+rec" if os.environ.get("RAPIDOCR_STOCK")
                else "PP-OCRv5 mobile det + PP-OCRv5-EN mobile rec",
                _rapid_intra_op,
            )
        except Exception as exc:
            rapidocr_engine = None
            _rapidocr_pool = None
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
        default=1.6,
        ge=0.5,
        le=3.0,
        description="Unclip ratio for DB detector. 1.6 (RapidOCR's own default; "
                    "was 1.8). The prior 1.8 over-dilated boxes so adjacent "
                    "lines/glyphs bled together and PP-OCRv5 mis-read or "
                    "truncated them — verified on the reconstruction-path "
                    "display-window crops (2026-09-09): 1.8 gave 'HE'S BEHIND "
                    "iiisn' / 'SIHI NANO MACHINE' / dropped whole clauses where "
                    "1.6 reads 'HE'S BEHIND US!!!' / 'THIS NANO MACHINE' "
                    "correctly. Env override: RECAP_OCR_UNCLIP.",
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
        description="Box score threshold for DB detector. 0.5 (RapidOCR's own "
                    "default; was 0.4). The old 0.4/1.8 pair came from a sweep "
                    "on the pre-reconstruction tile crops; on the current "
                    "display-window crops 0.5/1.6 fixed ~9 real dialogue errors "
                    "per chapter on the Nano Machine test set for ~4 trivial "
                    "junk shards. Env override: RECAP_OCR_BOX_THRESH.",
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
    series: Optional[str] = Field(
        default=None,
        description="Series name/slug — enables the per-series known-mistakes "
                    "correction dictionary (data/ocr-corrections/<slug>.json).",
    )


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
    vlm_fallback: Optional[str] = Field(default=None, description="Cloud VLM OCR fallback model if a key is configured, else null")
    vlm_stats: Optional[dict] = Field(default=None, description="VLM fallback call counters this process")
    local_vlm: Optional[str] = Field(default=None, description="Local GOT-OCR2 fallback model if OCR_LOCAL_VLM is enabled, else null")
    local_vlm_stats: Optional[dict] = Field(default=None, description="Local VLM OCR call counters this process")


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


def _load_clean_dict():
    # type: () -> set
    """The system spellcheck word list — clean, unlike wordninja's frequency
    list which is polluted with glued tokens ('atleast', 'bethe') and rare
    surnames. Used to gate word-splitting so a real merged pair is split but
    a romanised name is not."""
    for p in ("/usr/share/dict/american-english", "/usr/share/dict/words"):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                d = {ln.strip().lower() for ln in f
                     if ln.strip().isalpha() and len(ln.strip()) >= 2}
            if len(d) > 20000:
                return d
        except OSError:
            continue
    return set()


_CLEAN_DICT = _load_clean_dict()


def _load_extra_words():
    # type: () -> set
    """Real English + romanised wuxia/murim vocabulary the bundled wordninja
    list is missing (pipeline/ocr-corrections/_words.txt). Cross-series."""
    out = set()
    _h = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.join(_h, "..", "..", "pipeline", "ocr-corrections"),
                 os.path.join(_h, "..", "..", "data", "ocr-corrections")):
        p = os.path.join(base, "_words.txt")
        try:
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.split("#", 1)[0].strip().lower()
                    if ln and re.fullmatch(r"[a-z']+", ln):
                        out.add(ln)
        except OSError:
            continue
    return out


_EXTRA_WORDS = _load_extra_words()
# Short words the clean dict may lack but that are valid split parts.
_SPLIT_FUNCTION_WORDS = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
}


def _is_common_word(w):
    # type: (str) -> bool
    wl = w.lower()
    if wl in _SPLIT_FUNCTION_WORDS:
        return True
    if _CLEAN_DICT:
        return wl in _CLEAN_DICT
    return len(wl) >= 2 and _WORDCOST.get(wl, 99.0) <= 11.0


# Words the OCR (and its training corpus) frequently glues together, where
# wordninja either won't split them (it has the glued form in its own list)
# or a syllable collides with a surname. Whole-token, case-insensitive.
_GLUED_WORDS = {
    "atleast": "at least", "alot": "a lot", "aswell": "as well",
    "infront": "in front", "incase": "in case", "ofcourse": "of course",
    "eachother": "each other", "nomatter": "no matter", "thankyou": "thank you",
    "everytime": "every time", "infact": "in fact", "aslong": "as long",
    "atall": "at all", "bethe": "be the", "asif": "as if",
}


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

# short grammatical words that are never validly said twice in a row — an
# adjacent repeat is an OCR double-read, not emphasis
_DEDUPE_FUNCTION_WORDS = {
    "the", "to", "of", "and", "a", "is", "in", "it", "that", "at", "on",
    "for", "was", "with", "his", "her", "your", "my", "be", "he", "she",
}


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


# Small (4.4M param) masked-LM, domain-fine-tuned on ~55k verified
# narration sentences (comic-dialogue register), used ONLY to break a
# genuine tie between two spell-correction candidates that score
# identically by edit distance (e.g. "likes" vs "lakes" for corrupted
# "lkes") -- something no dictionary size can resolve, since the ambiguity
# is inherent to the edit-distance metric itself, not a coverage gap.
# Lazily loaded so a missing/not-yet-trained checkpoint just disables this
# one feature (falls back to the pre-existing deterministic top-1 pick)
# instead of failing service startup.
_MLM_MODEL_PATH = Path(__file__).parent / "models" / "ocr_correction_mlm"
_mlm_tokenizer = None
_mlm_model = None
_MLM_LOAD_ATTEMPTED = False


def _get_mlm():
    global _mlm_tokenizer, _mlm_model, _MLM_LOAD_ATTEMPTED
    if _MLM_LOAD_ATTEMPTED:
        return _mlm_tokenizer, _mlm_model
    _MLM_LOAD_ATTEMPTED = True
    if not _MLM_MODEL_PATH.exists():
        logger.info("OCR-correction disambiguation MLM not found at %s -- disambiguation disabled", _MLM_MODEL_PATH)
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tok = AutoTokenizer.from_pretrained(str(_MLM_MODEL_PATH))
        model = AutoModelForMaskedLM.from_pretrained(str(_MLM_MODEL_PATH))
        model.eval()
        _mlm_tokenizer, _mlm_model = tok, model
        logger.info("OCR-correction disambiguation MLM loaded from %s", _MLM_MODEL_PATH)
    except Exception as e:
        logger.warning("OCR-correction MLM failed to load (%s) -- disambiguation disabled", e)
    return _mlm_tokenizer, _mlm_model


def _disambiguate_tie(context_tokens, idx, candidates):
    # type: (List[str], int, List[str]) -> Optional[str]
    """Pick between 2 tied spell-correction candidates using the sentence
    they actually appear in, via a masked-LM cloze prediction, instead of
    an arbitrary (effectively random) tie-break. Returns the chosen
    candidate's exact string (lowercase), or None if the model isn't
    available or gives no usable signal -- callers should fall back to the
    existing deterministic behavior in that case, never guess further."""
    tok, model = _get_mlm()
    if tok is None:
        return None
    try:
        import torch
        masked = list(context_tokens)
        masked[idx] = tok.mask_token
        sentence = " ".join(masked)
        inputs = tok(sentence, return_tensors="pt", truncation=True, max_length=64)
        mask_positions = (inputs.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            return None
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits[0, mask_positions[0]], dim=-1)
        scored = []
        for c in candidates:
            cid = tok.convert_tokens_to_ids(c.lower())
            if cid is None or cid == tok.unk_token_id:
                continue
            scored.append((c.lower(), probs[cid].item()))
        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]
    except Exception as e:
        logger.debug("MLM disambiguation failed (%s) -- falling back", e)
        return None


def _correct_token(tok, context_tokens=None, idx=None):
    # type: (str, Optional[List[str]], Optional[int]) -> str
    """Return a corrected spelling for a single OCR token, or the token
    unchanged. Only fires on a clearly non-word of length >= 5 that has a
    very close real-word neighbour — so character names and sound effects
    (no close dictionary neighbour) pass straight through. Hyphenated /
    apostrophe'd tokens (stutters "M-MOVE", compounds "LOW-TIER",
    contractions) are left alone.

    context_tokens/idx (the full token list this token came from, and its
    position in it) are used ONLY to break a genuine tie between two
    similarly-scored candidates via a small masked-LM (see
    _disambiguate_tie) — e.g. "lkes" is equidistant from both "likes" and
    "lakes" by edit distance alone (proven: identical indel/JW scores), and
    picking one is a coin flip without the surrounding sentence. Callers
    that don't have context (or during tests) can omit these and get the
    exact pre-existing single-best-candidate behavior."""
    if _rf_process is None or not _WORDCOST:
        return tok
    if "-" in tok or "'" in tok or "’" in tok:
        return tok
    core = re.sub(r"[^A-Za-z]", "", tok)
    # Floor was 5 -- verified (real 4-letter substitution pairs: than/then,
    # chat/that, ever/even, were/here, wave/gave, warm/worm, best/rest,
    # fast/last all score indel=0.75, safely below the 0.80 gate below) that
    # dropping to 4 doesn't let same-length substitutions merge distinct
    # real words. It's specifically needed for a DELETION dropping a
    # 5-letter word to 4 chars ("likes" -> "lkes", indel=0.89/jw=0.94 --
    # comfortably clears both gates and is nothing like the substitution
    # pairs above). Below 4, single-letter words get too ambiguous either
    # way to risk it.
    if len(core) < 4 or any(ch.isdigit() for ch in tok):
        return tok
    if _is_dict_word(core) or _is_probable_sfx(tok):
        return tok
    key = core.lower()
    if key in _SPELL_CACHE:
        cand = _SPELL_CACHE[key]
    else:
        cand = None
        tie_candidate = None
        # top-2, not top-1: need the runner-up to detect a genuine tie.
        results = _rf_process.extract(
            key, _spell_candidates(len(key)),
            scorer=_rf_Indel.normalized_similarity, score_cutoff=0.80, limit=2)
        if results:
            w, indel, _score_idx = results[0]
            if abs(len(w) - len(key)) <= max(3, len(key) // 2):
                jw = _rf_JW.normalized_similarity(key, w)

                def _gate(cw, ind):
                    # Only ever correct TO a clean-dictionary word (never to
                    # wordninja-list junk like "muri"), and:
                    #  - near-identical  -> typo, fix it
                    #  - close + strong prefix match  -> OCR letter swap
                    #  - a pure adjacent transposition ("escpae"->"escape")
                    #  - looser, but only if the OCR token itself looks
                    #    mangled ("dunngoeoon" - 4 vowels in a row)
                    # A romanised name ('SUNBAE','GONGJA','JAIHUAN') sits one
                    # substitution (indel ~0.83) from a real word but has
                    # clean phonotactics and a non-word target -> never fires.
                    if not _is_common_word(cw):
                        return False
                    jwx = _rf_JW.normalized_similarity(key, cw)
                    transposed = (len(key) == len(cw) and sorted(key) == sorted(cw)
                                  and 0 < sum(a != b for a, b in zip(key, cw)) <= 2)
                    return (ind >= 0.90
                            or (ind >= 0.86 and jwx >= 0.90)
                            or transposed
                            or (ind >= 0.80 and jwx >= 0.88 and _looks_corrupted(key)))

                if _gate(w, indel):
                    cand = w
                    if len(results) > 1:
                        w2, indel2, _ = results[1]
                        jw2 = _rf_JW.normalized_similarity(key, w2)
                        gate2 = _gate(w2, indel2)
                        # "tie" = both candidates clear the safety gate AND
                        # score within 0.02 of each other (indel is 0-1) --
                        # e.g. likes/lakes score IDENTICALLY here.
                        if gate2 and w2.lower() != w.lower() and abs(indel - indel2) < 0.02:
                            tie_candidate = w2
        # A tied result is context-dependent (the same corrupted token can
        # resolve differently in different sentences), so it must NOT go
        # into _SPELL_CACHE the way a clean single-winner does -- caching it
        # would freeze whichever sentence asked first as the answer for
        # every future occurrence of this token regardless of its own
        # context. Only cache the non-tied path.
        if tie_candidate is not None and context_tokens is not None and idx is not None:
            resolved = _disambiguate_tie(context_tokens, idx, [cand, tie_candidate])
            if resolved:
                return _apply_case(tok, core, resolved)
            # model unavailable/inconclusive -- fall through to the
            # deterministic top-1 below, same as if there'd been no tie
        _SPELL_CACHE[key] = cand
    if not cand:
        return tok
    return _apply_case(tok, core, cand)


def _apply_case(tok, core, cand):
    # type: (str, str, str) -> str
    """Re-apply tok's original casing (all-caps / title-case / lower) to
    the chosen replacement word `cand`, and substitute it into `tok`."""
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

# current series slug for the in-flight OCR batch (set by ocr_batch); read by
# the garble detector and the correction layer.
_ocr_series_ctx = {"slug": "_none"}


def _looks_corrupted(s):
    # type: (str) -> bool
    """Heuristic: does this look like an OCR mangling rather than a clean
    (if unfamiliar) name? Used to gate fuzzy spell-correction in the risky
    mid-similarity band — a romanised name ('SUNBAE', 'GONGJA', 'JAIHUAN')
    has clean phonotactics and must never be 'corrected' to a real word,
    whereas a genuine garble ('dunngoeoon', 'absollte', 'lkes') carries a
    visible signature."""
    sl = re.sub(r"[^a-z]", "", s.lower())
    if len(sl) < 4:
        return False
    if re.search(r"(.)\1\1", sl):                       # 3+ same char in a row
        return True
    if re.search(r"[aeiouy]{3,}", sl):                  # 3+ vowels in a row
        return True
    v = sum(c in "aeiouy" for c in sl)
    if v / len(sl) < 0.30:                              # very consonant-heavy
        return True
    return False


def _is_known_wordform(w):
    # type: (str) -> bool
    """Membership-only check against the bundled wordninja list (NOT gated on
    frequency cost like _is_common_word) plus the clean system dict. Used by
    the garble detector to tell a real-but-uncommon word or a known romanised
    name ("SHREWD", "DEATHBED", "CHEON", "JANG") from an OCR mangling
    ("CTNOHS", "IILLHS") — the strict cost threshold wrongly rejects the
    former on a box with no /usr/share/dict installed."""
    raw = w.lower().strip()
    # a contraction / possessive is "known" when the stem + clitic are both
    # known ("they're", "would've", "jang's", "ma'am") — wordninja's bare-alpha
    # list has neither the glued nor the apostrophe form.
    if "'" in raw or "’" in raw:
        parts = [re.sub(r"[^a-z]", "", p) for p in re.split(r"['’]", raw)]
        parts = [p for p in parts if p]
        _CLITIC = {"s", "re", "ve", "ll", "d", "m", "t", "am", "o", "clock", "all", "cause", "em", "n"}
        if parts and all(
            len(p) <= 1 or p in _CLITIC or p in _SPLIT_FUNCTION_WORDS
            or p in _WORDCOST or (_CLEAN_DICT and p in _CLEAN_DICT)
            for p in parts
        ):
            return True
    wl = re.sub(r"[^a-z]", "", raw)
    if len(wl) < 2:
        return True
    if wl in _SPLIT_FUNCTION_WORDS or wl in _EXTRA_WORDS:
        return True
    if _CLEAN_DICT and wl in _CLEAN_DICT:
        return True
    return wl in _WORDCOST


def _token_looks_mangled(tok):
    # type: (str) -> bool
    """One alphabetic token: does its letter structure look like an OCR
    mangling rather than a real (if unfamiliar) word or romanised name?
    Signals that survive a high OCR confidence score: a run of 3+ consonants
    that isn't a legal English cluster ("CTNOHS", "IILLHS"), no vowel at all
    ("BHND"), or a 3+ same-letter run in a lowercase token ("iiisn")."""
    # check each apostrophe-separated part ("MUST'VE" -> "MUST", "VE")
    if _is_known_wordform(tok):
        return False
    for part in re.split(r"['’]", tok):
        low = re.sub(r"[^a-z]", "", part.lower())
        if len(low) < 3 or _is_known_wordform(part):
            continue
        if part.islower() and re.search(r"(.)\1\1", low):
            return True
        if len(low) >= 4 and not _VOWEL_RE.search(low):
            return True
        cons = sum(c not in "aeiouy" for c in low)
        # very consonant-dense for its length ("CTNOHS" 5/6) — real English
        # words and romanised names top out well below this.
        if len(low) >= 5 and cons / len(low) >= 0.78:
            return True
        # a 4+ consonant run is a garble regardless of any legal 3-cluster
        # it happens to contain ("IILLHS" -> "llhs").
        if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", low):
            return True
    return False


def _ocr_text_quality(text):
    # type: (str) -> float
    """Lower = cleaner. Cost = mangled tokens (weight 3) + other non-dict
    tokens >=3 chars (weight 1), normalised against a small completeness
    bonus for capturing more real words. Used to compare two OCR reads of
    the same panel (e.g. standard vs tight-unclip)."""
    toks = re.findall(r"[A-Za-z][A-Za-z'’]*", text or "")
    real = [t for t in toks if len(re.sub(r"[^a-z]", "", t.lower())) >= 2]
    if not real:
        return 99.0
    cost = 0.0
    dict_hits = 0
    for t in real:
        low = re.sub(r"[^a-z]", "", t.lower())
        if _is_known_wordform(t):
            dict_hits += 1
        elif len(low) >= 3:
            cost += 3.0 if _token_looks_mangled(t) else 1.0
    # a couple more real words is worth a small non-dict token
    return cost - 0.25 * dict_hits


def _looks_like_ocr_garble(text):
    # type: (str) -> bool
    """Does a *confident* classical OCR result contain a mangled token — the
    failure a low-confidence gate misses? Real Nano Machine cases at conf
    0.84-0.99: "A HE'S BEHIND iiisn" (US!!!), "WHAT I CTNOHS DO" (SHOULD),
    "IILLHS GUESS THERE IS NO CHOICE" (I).

    Only second-guesses a result that is *mostly* real words — so a lone SFX
    token ("THRRRK", "KAJIK") or a genuine name line isn't sent to the slow
    VLM tier. A flagged panel is re-read by PaddleOCR variants + the VLM
    tier(s); if none does better the original text is still returned.
    """
    if not text:
        return False
    try:
        _nm = _load_corrections(_ocr_series_ctx.get("slug", "_none")).get("names") or frozenset()
    except Exception:
        _nm = frozenset()

    def _known(t):
        return _is_known_wordform(t) or re.sub(r"[^a-z]", "", t.lower()) in _nm

    toks = re.findall(r"[A-Za-z][A-Za-z'’]*", text)
    real = [t for t in toks if len(t) >= 2]
    if len(real) < 3:
        return False
    non_dict = [t for t in real
                if not _known(t)
                and len(re.sub(r"[^a-z]", "", t.lower())) >= 3]
    dict_hits = len(real) - len(non_dict)
    # A structurally-mangled token ("CTNOHS", "iiisn", "IILLHS") is a garble
    # as long as SOME clean words are present — regardless of the overall
    # dict ratio.
    if dict_hits >= 1 and any(_token_looks_mangled(t) for t in non_dict):
        return True
    # A RUN of 2+ consecutive non-dictionary tokens inside an otherwise
    # dict-heavy line ("IT SEEOH SS SWIIS HE CAN'T", "...HAPPY MOOD EVEN
    # DEATHBED") is a mid-sentence mis-read — a real name is normally a lone
    # token next to real words, not a cluster.
    if dict_hits >= 3:
        run = 0
        for t in real:
            is_nd = (not _known(t)
                     and len(re.sub(r"[^a-z]", "", t.lower())) >= 3)
            run = run + 1 if is_nd else 0
            if run >= 2:
                return True
    # The weaker casing / short-non-word heuristics need a mostly-clean panel
    # so a name- or SFX-heavy line isn't second-guessed.
    if dict_hits / len(real) < 0.55:
        return False
    caps = [t for t in real if t.isupper()]
    allcaps_panel = len(caps) >= max(2, len(real) - 1)
    for t in non_dict:
        low = re.sub(r"[^a-z]", "", t.lower())
        # a non-word blob whose casing doesn't match ALL-CAPS lettering:
        # "Mol" / "aX" dropped into an otherwise all-caps panel.
        if allcaps_panel and not t.isupper() and len(low) <= 4:
            return True
    return False


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
    _tokens = t.split()
    for _idx, tok in enumerate(_tokens):
        core = re.sub(r"[^A-Za-z0-9]", "", tok)
        if not core:
            if re.fullmatch(r"(?:\.{2,}|!+|\?+|[!?]{2,}|[-–—]+|,)", tok):
                out.append("..." if tok.startswith("..") else tok)
            continue
        # ordinal ("3RD", "2ND", "1ST", "4TH") — keep intact; the "RD"/"ND"/
        # "TH" tail would otherwise be dropped by the vowel-less-shard rule
        if re.fullmatch(r"\d+(?:st|nd|rd|th)", core, re.IGNORECASE):
            out.append(tok)
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
        out.append(_correct_token(tok, _tokens, _idx))

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

    # a stray article the detector invented before a pronoun ("A HE'S BEHIND
    # US" -> "HE'S BEHIND US", "...NO. A HE DISAPPEARED" -> "...NO. HE
    # DISAPPEARED") — an extra word being removed, not a real one lost.
    _PRON_RE = re.compile(r"(?i)(he|she|they|it|we|you)('|s|$)")
    if len(out) >= 3 and re.fullmatch(r"[Aa]n?", out[0]) and \
            _PRON_RE.match(re.sub(r"[^A-Za-z']", "", out[1])):
        out = out[1:]
    for _i in range(1, len(out) - 1):
        if (re.fullmatch(r"[Aa]n?", out[_i])
                and out[_i - 1].rstrip().endswith((".", "!", "?", "...", '."', '?"'))
                and _PRON_RE.match(re.sub(r"[^A-Za-z']", "", out[_i + 1]))):
            out[_i] = ""
    out = [w for w in out if w]

    # mid-word case flip in an all-caps panel ("WEll" -> "WELL", "RiGHT",
    # "iNDiViDUAL", "So" -> "SO", "oF" -> "OF"): the detector wobbled on a
    # glyph's case, not the letter. Only when uppercase already dominates the
    # token, so a genuine lower-case word is left alone.
    if allcaps_panel:
        for _k, _w in enumerate(out):
            _wl = re.sub(r"[^A-Za-z]", "", _w)
            if (len(_wl) >= 2 and not _w.isupper() and not _w.islower()
                    and sum(c.isupper() for c in _wl) >= sum(c.islower() for c in _wl)
                    and not _is_probable_sfx(_w)):
                out[_k] = _w.upper()

    # adjacent duplicate of a short function word ("TO TO THE", "OF OF THE") —
    # an OCR double-read across a line wrap, never real emphasis for these.
    if len(out) >= 2:
        _deduped = [out[0]]
        for _w in out[1:]:
            _c = re.sub(r"[^a-z]", "", _w.lower())
            if (_c in _DEDUPE_FUNCTION_WORDS
                    and re.sub(r"[^a-z]", "", _deduped[-1].lower()) == _c):
                continue
            _deduped.append(_w)
        out = _deduped

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

    def _glued(m):
        tok = m.group(0)
        repl = _GLUED_WORDS.get(tok.lower())
        if not repl or _is_probable_sfx(tok):
            return tok
        return repl.upper() if tok.isupper() else repl

    text = re.sub(r"\b[A-Za-z]{4,}\b", _glued, text)

    def _splittable_part(p, short_token):
        # every part must be a clean-dictionary word; for a short source token
        # (< 8 chars) also require each part to be >= 3 chars OR a function
        # word, so a romanised name ("GARAM" -> "GAR AM", "SEOLAH" -> "SEOL
        # AH") is never split even though its syllables are dictionary words.
        if not _is_common_word(p):
            return False
        if short_token and len(p) < 3 and p.lower() not in _SPLIT_FUNCTION_WORDS:
            return False
        return True

    def _fix(m):
        tok = m.group(0)
        if len(tok) < 5 or _is_common_word(tok) or _is_probable_sfx(tok):
            return tok
        parts = _wordninja.split(tok)
        if len(parts) < 2 or sum(len(p) for p in parts) != len(tok):
            return tok
        short_token = len(tok) < 8
        if not all(_splittable_part(p, short_token) for p in parts):
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

    # possessive / contraction 's split off the word by a stray space around
    # the apostrophe ("DEMON' S" / "DRAGON 'S" -> "DEMON'S" / "DRAGON'S").
    # Without this the orphaned "S" is swept away as a 1-char shard later.
    t = re.sub(r"\b([A-Za-z]{2,})'\s+([Ss])\b", r"\1'\2", t)
    t = re.sub(r"\b([A-Za-z]{2,})\s+'([Ss])\b", r"\1'\2", t)
    t = re.sub(r"\b([A-Za-z]{2,})\s+'\s+([Ss])\b", r"\1'\2", t)

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


# ---------------------------------------------------------------------------
# KNOWN-MISTAKES CORRECTION DICTIONARY
# A deterministic, zero-cost layer: OCR errors that recur in a given
# scanlation (a stylised glyph the model always mis-reads the same way —
# "EHOH"->"HOW", "SIHI"->"THIS") get corrected from a per-series JSON, plus a
# global list of universal glued-word / spacing fixes. Files live in
# data/ocr-corrections/ : _global.json + <series-slug>.json (+ <slug>.learned.jsonl,
# auto-appended candidates promoted after N confirmations). Applied last, after
# _repair_and_denoise, so it operates on clean-ish text.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# Curated, version-controlled dictionaries ship in pipeline/ocr-corrections/;
# runtime data/ocr-corrections/ holds the auto-learned candidate files and any
# local per-box overrides (both merged, data/ winning on conflict).
_CORR_DIRS = [os.path.join(_HERE, "..", "..", "pipeline", "ocr-corrections"),
              os.path.join(_HERE, "..", "..", "data", "ocr-corrections")]
_CORR_LEARN_DIR = _CORR_DIRS[1]
_corr_cache = {}   # slug -> (mtime_sig, compiled)
_corr_lock = threading.Lock()


def _slugify_series(name):
    # type: (str) -> str
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "_none"


# Scanlation credit / aggregator-watermark / staff-list line: OCR reads it off
# the title & end cards of almost every chapter and it must never be spoken or
# memorised as an exemplar. Matched case-insensitively anywhere in the line.
_CREDIT_LINE_RE = re.compile(
    r"(?i)(?:asura ?scan?s?|asura ?scams?|asura\.?gg|discord\.?gg|/asuran\b|"
    r"\bzo?doc?c\b|\brodocc\b|\bpedoce\b|korean ?translators?|recruiting now|"
    r"for the fastest releases?|read (?:it |them )?at for the|"
    r"re?[dl] ?ice s[tu][tu][dl][il][oc]|"                       # Red ice Studio / Studic / Stutlo
    r"\bartista\b|han ?joong ?w[u]?eol ?ya|w[u]?eol ?ya\b|guem ?gang ?bul ?gae|"
    r"\bartist ?a?\b.{0,40}\bstudi[oc]\b|author ?[-:·]? ?han ?joong|"
    r"novel ?chapters? ?:|published by river|read at\b.{0,20}fastest)"
)
# a run of scanlation staff-list tokens ("NANO MACHINE OAKS SCARPET SCARPET KIRO
# RUSH REGIS", "DOTORI BRAHIMM DAN") that OCR lifts off the credits banner —
# ALL-CAPS short tokens, no lowercase, no sentence punctuation.
_CREDIT_STAFF_RE = re.compile(
    r"(?i)\b(?:nano ?machine|oaks|scarpet|kiro|dotori|brahimm|regis|kishi|rush|"
    r"lance|peri|victor|joemama|euwen|dan|mado|med|me|reg is)\b")


def _line_has_real_dialogue(text):
    # type: (str) -> bool
    """After removing any credit signature, is there still a run of ordinary
    words left (i.e. the line is dialogue that merely picked up a trailing
    'Red ice Studio', not a pure credits line)?"""
    stripped = _CREDIT_LINE_RE.sub(" ", text)
    stripped = _CREDIT_STAFF_RE.sub(" ", stripped)
    words = re.findall(r"[A-Za-z][A-Za-z']{2,}", stripped)
    real = [w for w in words
            if _is_known_wordform(w) or w.lower() in _REAL_SHORT_WORDS]
    return len(real) >= 4


# Source → confidence weight for a learned correction. A GOT-OCR2 / cloud-VLM
# re-read of the pixels is strong evidence; a tighter-params retry of the same
# classical model is weaker.
_LEARN_WEIGHTS = {"got-ocr2": 1.0, "cloud-vlm": 1.0, "tight-unclip": 0.55,
                  "paddle-variant": 0.45, "cross-panel": 0.7}
_PROMOTE_MIN_WEIGHT = 3.0      # weighted confirmations to auto-apply a token pair
_PROMOTE_MIN_SHARE = 0.62      # this `to` must own >=62% of `from`'s total weight
_EXEMPLAR_MIN_WEIGHT = 2.0     # whole-line exemplar reuse


def _norm_line(s):
    # type: (str) -> str
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9'!?. ]", "", (s or "").upper())).strip()


def _read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except Exception as exc:
        logger.debug("jsonl read %s (%s)", path, exc)
    return out


def _learned_paths(slug):
    return (os.path.join(_CORR_LEARN_DIR, f"{slug}.learned.jsonl"),
            os.path.join(_CORR_LEARN_DIR, f"{slug}.exemplars.jsonl"),
            os.path.join(_CORR_LEARN_DIR, f"{slug}.review.md"))


# Cross-series learned pool. A structurally-mangled token ("CTNOHS", "IIHWOS")
# that a pixel-level re-read resolves to a common English word is almost always
# a FONT-level mis-read (the same spiky scanlation lettering is reused across
# many titles), not something specific to one series — so it also accrues here
# and is offered to every series' dict. Romanised names never reach this pool
# (they aren't `_token_looks_mangled`), so it can't corrupt a different title's
# proper nouns.
_GLOBAL_LEARN_PATH = os.path.join(_CORR_LEARN_DIR, "_global.learned.jsonl")
_GLOBAL_PROMOTE_MIN_WEIGHT = 4.0   # higher bar than per-series: needs real repetition


def _globally_learnable(cb, ca, source):
    # type: (str, str, str) -> bool
    cl = re.sub(r"[^a-z]", "", ca.lower())
    return (source in ("got-ocr2", "cloud-vlm", "cloud-llm")
            and _token_looks_mangled(cb) and not _is_probable_sfx(cb)
            and len(cl) >= 3 and _is_known_wordform(cl)
            and _is_common_word(cl))          # a genuinely common target, not a rare name-ish word


def _aggregate_pairs(path, pairs, weight, source, ts):
    # type: (str, list, float, str, int) -> None
    """Merge (FROM, TO) token pairs into a learned.jsonl-style file, bumping
    weight/count/src for existing rows."""
    rows = _read_jsonl(path)
    agg = {}
    for e in rows:
        agg[(e.get("from", ""), e.get("to", ""), e.get("kind", "tok"))] = e
    for fr, to in pairs:
        k = (fr, to, "tok")
        e = agg.get(k) or {"from": fr, "to": to, "kind": "tok",
                           "weight": 0.0, "count": 0, "first": ts}
        e["weight"] = round(float(e.get("weight", 0)) + weight, 2)
        e["count"] = int(e.get("count", 0)) + 1
        e["last"] = ts
        e.setdefault("src", {})
        e["src"][source] = e["src"].get(source, 0) + 1
        agg[k] = e
    with open(path, "w", encoding="utf-8") as f:
        for e in sorted(agg.values(), key=lambda x: -float(x.get("weight", 0))):
            f.write(json.dumps(e) + "\n")


def _compile_learned(slug):
    # type: (str) -> dict
    """Turn the raw <slug>.learned.jsonl / .exemplars.jsonl into an applied set:
    promoted token map (conflict-safe), a char-confusion table for fuzzy repair
    of NOVEL tokens, learned SFX, and whole-line exemplars. Also (re)writes the
    human review file."""
    lp, xp, rp = _learned_paths(slug)
    raw = _read_jsonl(lp)
    # group candidate `to`s per `from`, with weighted counts
    by_from = {}   # FROM -> {TO -> weight}
    contra = {}    # FROM -> weight of "seen but resolved to something ELSE / kept"
    sfx_votes = {}
    vlm_w = {}      # FROM -> weight that came specifically from a pixel-level VLM re-read
    global_from = set()   # FROM keys that came (also) from the cross-series pool
    for e in raw:
        fr = (e.get("from") or "").upper()
        to = e.get("to")
        if not fr or to is None:
            continue
        w = float(e.get("weight", e.get("count", 1)))
        if e.get("kind") == "sfx":
            sfx_votes[fr] = sfx_votes.get(fr, 0.0) + w
            continue
        by_from.setdefault(fr, {})
        by_from[fr][to.upper()] = by_from[fr].get(to.upper(), 0.0) + w
        _srcs = e.get("src", {})
        if isinstance(_srcs, dict) and any(s in _srcs for s in ("got-ocr2", "cloud-vlm", "cloud-llm")):
            vlm_w[fr] = max(vlm_w.get(fr, 0.0), w)
    # merge the cross-series pool: a font-level mis-read confirmed on OTHER
    # titles is offered here too, but only counts once it clears a higher bar.
    for e in _read_jsonl(_GLOBAL_LEARN_PATH):
        fr = (e.get("from") or "").upper()
        to = (e.get("to") or "").upper()
        if not fr or not to or e.get("kind") == "sfx":
            continue
        gw = float(e.get("weight", 0))
        if gw < _GLOBAL_PROMOTE_MIN_WEIGHT:
            continue
        by_from.setdefault(fr, {})
        by_from[fr][to] = by_from[fr].get(to, 0.0) + gw
        vlm_w[fr] = max(vlm_w.get(fr, 0.0), gw)
        global_from.add(fr)
    promoted = {}
    near = []       # for review
    conflicts = []
    for fr, tos in by_from.items():
        total = sum(tos.values())
        best_to, best_w = max(tos.items(), key=lambda kv: kv[1])
        share = best_w / total if total else 0
        _bt_low = re.sub(r"[^a-z]", "", best_to.lower())
        _fr_low = re.sub(r"[^a-z]", "", fr.lower())
        real = (_is_known_wordform(_bt_low) or _bt_low in _REAL_SHORT_WORDS)
        # a `to` that is just a truncation of `from` ("MENTS"->"MENT") is a
        # tokenisation artefact, never a real fix
        trunc = bool(_bt_low) and (_fr_low.startswith(_bt_low) or _fr_low.endswith(_bt_low))
        # FAST TRACK: `from` is STRUCTURALLY garbled (4+ consonant run / vowelless
        # / consonant-dense) so it cannot be a character name — a single
        # pixel-level VLM read to a real word, with no competing target, is
        # trusted immediately. This is what stops GOT-OCR2 being called again
        # for the same mis-read. Never fast-track a shout/onomatopoeia `from`
        # (GOT-OCR2 returns a junk short word for a scream panel) or a 1-2 letter
        # `to`.
        fast = (_token_looks_mangled(fr) and real and not trunc and len(tos) == 1
                and not _is_probable_sfx(fr) and len(_bt_low) >= 3
                and vlm_w.get(fr, 0) >= 1.0 and best_to != fr)
        if not trunc and (fast or (best_w >= _PROMOTE_MIN_WEIGHT
                                   and share >= _PROMOTE_MIN_SHARE and real)):
            promoted[fr] = best_to
        elif best_w >= 1.5:
            (conflicts if len(tos) > 1 and share < _PROMOTE_MIN_SHARE else near).append(
                (fr, tos, best_to, best_w, share))
    # character-confusion table — ONLY from pairs whose letters mostly already
    # line up (a genuine per-glyph mis-read: "SHREWD"->"SHREVD"), never a total
    # scramble ("CTNOHS"->"SHOULD"), so the fuzzy-repair rules stay trustworthy.
    confus = {}   # (wrong_char, right_char) -> [weight, {distinct from-tokens}]
    for fr, tos in by_from.items():
        for to, w in tos.items():
            a, b = re.sub(r"[^A-Z]", "", fr), re.sub(r"[^A-Z]", "", to)
            if not (3 <= len(a) <= 12 and len(a) == len(b)):
                continue
            same = sum(1 for ca, cb in zip(a, b) if ca == cb)
            if same < 0.6 * len(a) or (len(a) - same) > 2:
                continue          # not a clean 1-2 glyph substitution
            for ca, cb in zip(a, b):
                if ca != cb:
                    slot = confus.setdefault((ca, cb), [0.0, set()])
                    slot[0] += w
                    slot[1].add(fr)
    char_subs = {}
    for (ca, cb), (w, froms) in confus.items():
        # a glyph-confusion rule feeds fuzzy-repair of NOVEL tokens, so demand
        # real corroboration: weight >= 3 AND the same swap seen in >= 2
        # different mangled tokens (one pair at weight 2.2 is not a pattern).
        if w >= 3.0 and len(froms) >= 2:
            char_subs.setdefault(ca, []).append((cb, w))
    for ca in char_subs:
        char_subs[ca].sort(key=lambda t: -t[1])
    learned_sfx = {k for k, v in sfx_votes.items() if v >= 2.0}
    # exemplars: whole normalised garbled line -> fixed line, weighted
    ex_raw = _read_jsonl(xp)
    ex_by = {}
    for e in ex_raw:
        k = _norm_line(e.get("from", ""))
        if not k:
            continue
        ex_by.setdefault(k, {})
        ex_by[k][e.get("to", "")] = ex_by[k].get(e.get("to", ""), 0.0) + float(e.get("weight", 1))
    exemplars = {}
    for k, tos in ex_by.items():
        bt, bw = max(tos.items(), key=lambda kv: kv[1])
        # a whole-line exemplar only ever fires on an EXACT normalised match of
        # the garbled input, so a single pixel-level VLM read is enough — that
        # panel won't be sent to GOT-OCR2 a second time.
        if bw >= 1.0 and bt and not _looks_like_ocr_garble(bt):
            exemplars[k] = bt
    # write review file (best-effort)
    try:
        if near or conflicts:
            with open(rp, "w", encoding="utf-8") as f:
                f.write(f"# OCR correction review — {slug}\n\n")
                f.write(f"Auto-generated. {len(promoted)} promoted, {len(near)} near, "
                        f"{len(conflicts)} conflicting.\n\n")
                if conflicts:
                    f.write("## Conflicting (no `to` dominates — add to "
                            f"pipeline/ocr-corrections/{slug}.json by hand if you know it)\n\n")
                    for fr, tos, bt, bw, sh in conflicts:
                        f.write(f"- `{fr}` → {dict((k, round(v,1)) for k,v in tos.items())}\n")
                if near:
                    f.write("\n## Near promotion (need more confirmations)\n\n")
                    for fr, tos, bt, bw, sh in sorted(near, key=lambda t: -t[3]):
                        f.write(f"- `{fr}` → `{bt}`  (weight {bw:.1f}, share {sh:.0%})\n")
    except Exception:
        pass
    return {"promoted": promoted, "char_subs": char_subs,
            "sfx": learned_sfx, "exemplars": exemplars}


def _load_corrections(slug):
    # type: (str) -> dict
    """Compiled correction set: curated _global.json + <slug>.json, plus the
    self-learned layer (promoted tokens, char-confusion fuzzy repair, learned
    SFX, whole-line exemplars). Cached; invalidated on any source-file mtime
    change."""
    json_paths = []
    for d in _CORR_DIRS:
        json_paths.append(os.path.join(d, "_global.json"))
        json_paths.append(os.path.join(d, f"{slug}.json"))
    lp, xp, _rp = _learned_paths(slug)
    all_paths = json_paths + [lp, xp, _GLOBAL_LEARN_PATH]
    sig = tuple((p, os.path.getmtime(p)) for p in all_paths if os.path.exists(p))
    cached = _corr_cache.get(slug)
    if cached and cached[0] == sig:
        return cached[1]
    with _corr_lock:
        phrases = []
        tokens = {}
        sfx = set()
        names = set()
        for p in json_paths:
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception as exc:
                logger.warning("bad correction file %s (%s)", p, exc)
                continue
            for a, b in (d.get("phrases") or []):
                phrases.append((a.lower(), b))
            for k, v in (d.get("tokens") or {}).items():
                tokens[k.upper()] = v
            for s in (d.get("sfx") or []):
                sfx.add(s.upper())
            # `names`: series-specific proper nouns / romanised terms that OCR
            # reads correctly but a generic dict flags as garble. Protected —
            # never token-corrected, never dropped as noise, never learned as a
            # mistake.
            for nm in (d.get("names") or []):
                nc = re.sub(r"[^a-z]", "", str(nm).lower())
                if nc:
                    names.add(nc)
        learned = _compile_learned(slug)
        for k, v in learned["promoted"].items():
            if re.sub(r"[^a-z]", "", k.lower()) in names:
                continue                     # never "correct" a known name
            tokens.setdefault(k, v)          # curated .json wins on conflict
        sfx |= learned["sfx"]
        phrases.sort(key=lambda t: -len(t[0]))
        compiled = {"phrases": phrases, "tokens": tokens, "sfx": sfx, "names": names,
                    "char_subs": learned["char_subs"], "exemplars": learned["exemplars"]}
        _corr_cache[slug] = (sig, compiled)
        return compiled


def _series_name_set(slug):
    # type: (str) -> set
    """Protected proper-noun allow-list for a series (from its .json `names`)."""
    try:
        return _load_corrections(slug).get("names") or set()
    except Exception:
        return set()


def _match_case(src, repl):
    # type: (str, str) -> str
    if src.isupper():
        return repl.upper()
    if src.islower():
        return repl.lower()
    if src[:1].isupper() and src[1:].islower():
        return repl[:1].upper() + repl[1:].lower()
    return repl


def _fuzzy_repair_token(word, char_subs):
    # type: (str, dict) -> Optional[str]
    """For a NOVEL non-dictionary token, try applying this series' learned
    character confusions ("this font's U reads as H") — accept ONLY if exactly
    one resulting candidate is a real English word (no guessing)."""
    core = re.sub(r"[^A-Za-z]", "", word)
    if len(core) < 3 or len(core) > 12 or _is_known_wordform(core) or not char_subs:
        return None
    up = core.upper()
    positions = [i for i, c in enumerate(up) if c in char_subs]
    if not positions or len(positions) > 4:
        return None
    hits = set()
    import itertools
    # try changing 1..2 of the confusable positions
    for r in (1, 2):
        for combo in itertools.combinations(positions, r):
            opts = [char_subs[up[i]][:2] for i in combo]
            for choice in itertools.product(*opts):
                cand = list(up)
                for i, (cb, _w) in zip(combo, choice):
                    cand[i] = cb
                cs = "".join(cand)
                if cs != up and _is_known_wordform(cs.lower()):
                    hits.add(cs)
    if len(hits) == 1:
        return _match_case(core, next(iter(hits)))
    return None


def _apply_known_corrections(text, slug):
    # type: (str, str) -> str
    if not text:
        return text
    corr = _load_corrections(slug)
    if not (corr["phrases"] or corr["tokens"] or corr["sfx"]
            or corr["char_subs"] or corr["exemplars"]):
        # credit strip is unconditional even with no per-series dict
        return "" if _CREDIT_LINE_RE.search(text) and not _line_has_real_dialogue(text) else text
    # 0a. a scanlation credit / staff / aggregator-watermark line is never spoken
    if _CREDIT_LINE_RE.search(text) and not _line_has_real_dialogue(text):
        return ""
    # 0b. dialogue line that merely picked up a trailing credit tag
    #     ("...MASTER. Red ice Studio") — excise just the credit span
    if _CREDIT_LINE_RE.search(text):
        text = _CREDIT_LINE_RE.sub(" ", text)
        text = re.sub(r"\s*[/·|]+\s*$", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" /·|-")
    # 0c. whole-line exemplar — exact repeat of a garbled line we've already resolved
    ex = corr["exemplars"].get(_norm_line(text))
    if ex:
        return ex
    _names = corr.get("names") or set()
    out = text
    for search_lc, repl in corr["phrases"]:
        def _r(m, _repl=repl):
            return _match_case(m.group(0), _repl)
        out = re.sub(re.escape(search_lc), _r, out, flags=re.IGNORECASE)
    _tokmap, _subs = corr["tokens"], corr["char_subs"]
    if _tokmap or _subs:
        def _tok(m):
            w = m.group(0)
            key = re.sub(r"[^A-Za-z]", "", w).upper()
            if key.lower() in _names:           # protected proper noun — leave it
                return w
            repl = _tokmap.get(key)
            if repl is not None:
                return "" if repl == "" else _match_case(re.sub(r"[^A-Za-z]", "", w), repl)
            fz = _fuzzy_repair_token(w, _subs)
            return fz if fz is not None else w
        out = re.sub(r"\b[A-Za-z][A-Za-z']*\b", _tok, out)
    if corr["sfx"]:
        toks = out.split()
        real = [t for t in toks
                if _is_known_wordform(re.sub(r"[^A-Za-z]", "", t.lower()))]
        if len(real) >= 2:
            toks = [t for t in toks
                    if re.sub(r"[^A-Za-z]", "", t).upper() not in corr["sfx"]]
            out = " ".join(toks)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    # A panel whose ENTIRE transcription is one short non-word token ("JEHNH",
    # "Mah", "XXXX", "$fa") is the OCR hallucinating letters out of an
    # untranslated Korean SFX / texture — there is no English dialogue to
    # "correct", so the panel is correctly silent.
    _wtoks = re.findall(r"[A-Za-z][A-Za-z']*", out)
    _lone_junk = (len(_wtoks) <= 1 and len(re.sub(r"[^A-Za-z]", "", out)) <= 6
                  and not any(_is_known_wordform(w) or w.lower() in _REAL_SHORT_WORDS
                              for w in _wtoks)
                  and not re.search(r"[.!?]{2,}|\.\.\.|[-–—]$", out))
    if _wtoks and (_lone_junk or (len(_wtoks) <= 1 and _looks_like_ocr_noise(out))):
        return ""
    return out


def _learn_ocr_correction(before, after, slug, source="got-ocr2"):
    # type: (str, str, str, str) -> None
    """Record how a VLM / retry re-read of a panel differs from the classical
    OCR. Token-aligned single-word swaps feed the promotion + char-confusion
    machinery; whole-line differences feed the exemplar cache. `source` sets
    the evidence weight."""
    if not before or not after or slug == "_none":
        return
    w = _LEARN_WEIGHTS.get(source, 0.5)
    bw, aw = before.split(), after.split()
    token_pairs = []
    if len(bw) == len(aw):
        for b, a in zip(bw, aw):
            cb = re.sub(r"[^A-Za-z']", "", b)
            ca = re.sub(r"[^A-Za-z']", "", a)
            if not (cb and ca) or cb.lower() == ca.lower() or len(cb) < 3:
                continue
            bl, al = cb.lower(), ca.lower()
            # never learn a "fix" for a word that is already correct, a known
            # series proper noun, or a stretched shout / onomatopoeia (GOT-OCR2
            # reads a scream panel and returns a random short word:
            # "ARRRGHH"->"AR", "UWAAA"->"DON'T").
            if _is_known_wordform(cb) or _is_probable_sfx(cb):
                continue
            if bl in _series_name_set(slug):
                continue
            # `after` has to be a real word (drop the old len<=2 escape hatch
            # that let "->AR" / "->FO" / "->LO" through).
            if not (_is_known_wordform(ca) or al in _REAL_SHORT_WORDS):
                continue
            # truncation guard: GOT-OCR2 split the word differently and the
            # "correction" is just a fragment ("MENTS"->"MENT", "CIAL"->"CIA",
            # "TARDS"->"TARD", "SEOB"->"SEO"). Reject when one is a prefix/suffix
            # of the other.
            if bl.startswith(al) or bl.endswith(al) or al.startswith(bl) or al.endswith(bl):
                continue
            # length has to stay in the same ballpark
            ratio = len(al) / len(bl)
            if ratio < 0.5 or ratio > 1.8:
                continue
            token_pairs.append((cb.upper(), ca.upper()))
    lp, xp, _rp = _learned_paths(slug)
    try:
        os.makedirs(_CORR_LEARN_DIR, exist_ok=True)
        ts = int(time.time())
        if token_pairs:
            _aggregate_pairs(lp, token_pairs, w, source, ts)
            # font-level mis-reads also feed the cross-series pool
            gpairs = [(fr, to) for fr, to in token_pairs
                      if _globally_learnable(fr, to, source)]
            if gpairs:
                _aggregate_pairs(_GLOBAL_LEARN_PATH, gpairs, w, source, ts)
                _corr_cache.clear()          # global pool affects every slug
        # whole-line exemplar when the structure changed a lot. Store the
        # CLEANED `after` (VLM spacing tics like "ID ON'T" / "RUNAWAY" run
        # through the repair chain) so a verbatim replay isn't itself garbled.
        if (len(bw) != len(aw) or abs(len(before) - len(after)) > 4) \
                and len(re.findall(r"[A-Za-z]{2,}", after)) >= 3:
            clean_after = _repair_and_denoise(_clean_and_normalize_ocr_text(after))
            clean_after = _apply_known_corrections(clean_after, slug) or clean_after
            _ex_toks = re.findall(r"[A-Za-z][A-Za-z']{2,}", clean_after)
            _ex_clean = sum(1 for t in _ex_toks
                            if _is_known_wordform(t) or t.lower() in _REAL_SHORT_WORDS)
            if (not _looks_like_ocr_garble(clean_after)
                    and not _CREDIT_LINE_RE.search(clean_after)
                    # the fixed line has to be almost entirely real words, or a
                    # garble→garble pair (or a credit strip) gets memorised
                    and _ex_toks and _ex_clean / len(_ex_toks) >= 0.85):
                with open(xp, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"from": before, "to": clean_after,
                                        "weight": w, "src": source, "ts": ts}) + "\n")
        _corr_cache.pop(slug, None)
    except Exception as exc:
        logger.debug("learn correction write (%s)", exc)


def _learn_sfx(token, slug, weight=1.0):
    # type: (str, str, float) -> None
    """A lone short non-dict token on a panel with CJK present that even the VLM
    can't read as English → an untranslated sound effect."""
    if slug == "_none":
        return
    key = re.sub(r"[^A-Za-z]", "", token).upper()
    if len(key) < 2:
        return
    lp, _xp, _rp = _learned_paths(slug)
    try:
        os.makedirs(_CORR_LEARN_DIR, exist_ok=True)
        rows = _read_jsonl(lp)
        hit = None
        for e in rows:
            if e.get("kind") == "sfx" and e.get("from") == key:
                hit = e
                break
        if hit:
            hit["weight"] = round(float(hit.get("weight", 0)) + weight, 2)
        else:
            rows.append({"from": key, "to": "", "kind": "sfx", "weight": weight,
                         "first": int(time.time())})
        with open(lp, "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e) + "\n")
        _corr_cache.pop(slug, None)
    except Exception as exc:
        logger.debug("learn sfx (%s)", exc)


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
    merged_text = _apply_known_corrections(merged_text, _ocr_series_ctx["slug"])
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


def _run_rapidocr_on_image(img, options=None, unclip_override=None, box_override=None):
    # type: (np.ndarray, Optional[OCROptions], Optional[float], Optional[float]) -> List[_TextRegion]
    """Run RapidOCR (PP-OCRv5 mobile, ONNXRuntime) on a numpy image array.

    Mirrors _run_ocr_on_image's contract (same _TextRegion return shape)
    so the existing _merge_regions/_quality_status pipeline — including
    the gap-aware word-join fix for over-segmented bold lettering — works
    unchanged regardless of which engine produced the regions.
    """
    pool = _rapidocr_pool
    if pool is None:
        raise RuntimeError("RapidOCR engine is not initialized")

    opts = options or OCROptions()
    # Borrow one engine from the pool for the duration of this call. Separate
    # RapidOCR/ONNXRuntime instances are independent, so up to RAPIDOCR_POOL_SIZE
    # of these run concurrently (no _inference_lock — that's Paddle-only now).
    # Env overrides for quick A/B tuning without a code change / redeploy.
    try:
        _box = float(box_override if box_override is not None
                     else os.environ.get("RECAP_OCR_BOX_THRESH") or opts.det_db_box_thresh)
        _uc = float(unclip_override if unclip_override is not None
                    else os.environ.get("RECAP_OCR_UNCLIP") or opts.det_db_unclip_ratio)
    except (TypeError, ValueError):
        _box, _uc = opts.det_db_box_thresh, opts.det_db_unclip_ratio
    eng = pool.get()
    try:
        result = eng(img, box_thresh=_box, unclip_ratio=_uc)
    finally:
        pool.put(eng)

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


# ---------------------------------------------------------------------------
# VLM fallback tier (OPT-IN — only fires when a text region was localized but
# no classical engine could read it, and only when a cloud key is present)
#
# The classical cascade (RapidOCR -> PaddleOCR -> Tesseract) handles clean
# bubble lettering well but loses stylised SFX, warped/curved text, and
# low-contrast text-on-art. A multimodal model reads those. This tier is
# gated hard: it needs GEMINI_API_KEY (or OPENROUTER_API_KEY) in the
# service env (start.sh now sources ../../.env), it only runs after every
# classical engine has failed to reach SUCCESS on a panel that DID have a
# detected text region, and a sliding-window rate limiter keeps it inside
# the provider's free tier. With no key it is a no-op — zero behaviour
# change, zero new dependency (the `openai` client is already installed for
# the pipeline's visual-caption feature).
# ---------------------------------------------------------------------------
OCR_VLM_PROVIDER = os.environ.get("OCR_VLM_PROVIDER", "auto").lower()  # auto|gemini|openrouter|none
OCR_VLM_MODEL = os.environ.get("OCR_VLM_MODEL", "").strip()
_OCR_VLM_MAX_PER_MIN = max(1, int(os.environ.get("OCR_VLM_MAX_CALLS_PER_MIN", "12")))
_OCR_VLM_TIMEOUT = float(os.environ.get("OCR_VLM_TIMEOUT", "20"))
# Also try the VLM on panels the detector called blank (zero regions) —
# off by default; a stylised full-bleed SFX with no bubble is the case it
# would catch, at the cost of a call on every genuinely-silent panel.
OCR_VLM_ON_ZERO_REGIONS = os.environ.get("OCR_VLM_ON_ZERO_REGIONS", "0").lower() not in ("0", "false", "no")

_vlm_client = None      # type: Any
_vlm_label = None       # type: Optional[str]   # None = not resolved, "" = unavailable
_vlm_lock = threading.Lock()
_vlm_call_times = []    # type: List[float]     # sliding-window timestamps
_vlm_stats = {"calls": 0, "hits": 0, "skipped_ratelimit": 0, "errors": 0}

# ---------------------------------------------------------------------------
# LOCAL VLM OCR TIER — GOT-OCR2.0 (580M, Apache-2.0, transformers-native).
# A KEY-FREE last-resort reader for the same UNCERTAIN-with-regions panels
# the cloud VLM tier targets: stylised bold lettering, jagged SFX bubbles,
# text on busy art. Measured on this 4-vCPU box: ~10s/panel (vs RapidOCR's
# ~0.25s), so it is OFF by default and hard-capped per process. It correctly
# reads cases RapidOCR mangles (e.g. a spiky "HE'S BEHIND US!!!" bubble that
# PP-OCRv5 returned as "A HE'S BEHIND iiisn"). Runs in the shared project
# venv (transformers/torch already present). Set OCR_LOCAL_VLM=1 to enable.
# ---------------------------------------------------------------------------
OCR_LOCAL_VLM = os.environ.get("OCR_LOCAL_VLM", "").strip().lower() in (
    "1", "true", "yes", "on", "got", "got-ocr2", "gotocr2")
OCR_LOCAL_VLM_MODEL = os.environ.get("OCR_LOCAL_VLM_MODEL", "stepfun-ai/GOT-OCR-2.0-hf")
# 0 = unlimited. Default keeps a pathological chapter (every panel UNCERTAIN)
# from turning into a ~15-min OCR stall.
_OCR_LOCAL_VLM_MAX = max(0, int(os.environ.get("OCR_LOCAL_VLM_MAX_CALLS", "120")))
_OCR_LOCAL_VLM_TOKENS = max(32, int(os.environ.get("OCR_LOCAL_VLM_TOKENS", "256")))
_local_vlm_lock = threading.Lock()
_local_vlm = {"resolved": False, "model": None, "proc": None}
_local_vlm_stats = {"calls": 0, "hits": 0, "skipped_cap": 0, "errors": 0}


def _get_vlm_client():
    # type: () -> Tuple[Any, str]
    """Lazily build the OpenAI-compatible client for the configured VLM
    provider. Returns (client, model_label); (None, "") when unavailable."""
    global _vlm_client, _vlm_label
    if _vlm_label is not None:
        return _vlm_client, _vlm_label
    with _vlm_lock:
        if _vlm_label is not None:
            return _vlm_client, _vlm_label
        prov = OCR_VLM_PROVIDER
        gem = os.environ.get("GEMINI_API_KEY", "").strip()
        orouter = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if prov == "none" or (not gem and not orouter):
            _vlm_client, _vlm_label = None, ""
            return _vlm_client, _vlm_label
        try:
            from openai import OpenAI
        except Exception as exc:
            logger.warning("VLM OCR tier: `openai` client not importable (%s) — disabled", exc)
            _vlm_client, _vlm_label = None, ""
            return _vlm_client, _vlm_label
        try:
            if prov in ("auto", "gemini") and gem:
                _vlm_client = OpenAI(api_key=gem, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                _vlm_label = OCR_VLM_MODEL or os.environ.get("GEMINI_MODEL", "").strip() or "gemini-2.0-flash"
            elif prov in ("auto", "openrouter") and orouter:
                _vlm_client = OpenAI(api_key=orouter, base_url="https://openrouter.ai/api/v1")
                _vlm_label = OCR_VLM_MODEL or "google/gemini-2.0-flash-exp:free"
            else:
                _vlm_client, _vlm_label = None, ""
            if _vlm_client is not None:
                logger.info("VLM OCR fallback tier ENABLED (provider=%s model=%s, <=%d calls/min)",
                            prov, _vlm_label, _OCR_VLM_MAX_PER_MIN)
        except Exception as exc:
            logger.warning("VLM OCR tier: client init failed (%s) — disabled", exc)
            _vlm_client, _vlm_label = None, ""
        return _vlm_client, _vlm_label


_VLM_OCR_PROMPT = (
    "You are an OCR engine for English-language comic / manhwa / webtoon panels. "
    "Transcribe EVERY piece of readable text in this image EXACTLY as written: speech "
    "bubbles, caption/narration boxes, and large stylised sound-effect lettering. "
    "Separate the contents of distinct bubbles/boxes with ' / '. Preserve wording, "
    "punctuation, capitalisation and ellipses. Do NOT translate, paraphrase, summarise, "
    "describe the artwork, name characters, or add any commentary or labels. "
    "If there is no readable lettering at all, reply with exactly: NO_TEXT"
)


def _normalize_vlm_ocr_text(txt):
    # type: (str) -> str
    """Clean a VLM transcription (cloud or local GOT-OCR2). GOT-OCR2 in
    particular emits an apostrophe with a stray space ("HE' S" / "DON' T"),
    glues a pronoun to its verb ("IDON'T"), and run-joins short words
    ("RUNAWAY"). Route it through the same repair chain the classical
    engines use, after fixing those VLM-specific tics."""
    if not txt:
        return ""
    txt = re.sub(r"[ \t]*\n[ \t]*", " / ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip(" /")
    # apostrophe with a stray space on either side: "HE' S" / "DON 'T" -> "HE'S"
    txt = re.sub(r"([A-Za-z])\s*['’]\s*([A-Za-z])", r"\1'\2", txt)
    # pronoun glued to a following contraction/verb: "IDON'T" -> "I DON'T"
    txt = re.sub(r"\bI(DON'T|CAN'T|WON'T|DIDN'T|WASN'T|COULDN'T|SHOULDN'T|"
                 r"AM|WAS|WILL|HAVE|HAD|THINK|KNOW|GUESS|SEE|DO|CAN|NEED)\b",
                 r"I \1", txt)
    txt = re.sub(r"\bi(don't|can't|won't|didn't|am|was|will|have|had|think|"
                 r"know|guess|see|do|can|need)\b", r"i \1", txt)
    # collapse an immediately-repeated short function word ("I I DON'T",
    # "THE THE") the glue-fix or the model itself doubled.
    txt = re.sub(r"\b(I|A|THE|TO|OF|IT|HE|WE|IS|AND)\s+\1\b", r"\1", txt, flags=re.I)
    out = []
    for seg in txt.split(" / "):
        seg = _repair_and_denoise(_clean_and_normalize_ocr_text(seg)).strip()
        if seg:
            out.append(seg)
    return " / ".join(out)


def _vlm_rate_ok():
    # type: () -> bool
    now = time.time()
    with _vlm_lock:
        while _vlm_call_times and now - _vlm_call_times[0] > 60.0:
            _vlm_call_times.pop(0)
        if len(_vlm_call_times) >= _OCR_VLM_MAX_PER_MIN:
            _vlm_stats["skipped_ratelimit"] += 1
            return False
        _vlm_call_times.append(now)
        return True


def _run_vlm_ocr(img):
    # type: (np.ndarray) -> Tuple[str, float]
    """Send one panel crop to the configured VLM and return (text, pseudo_conf).
    ("", 0.0) on any failure, rate-limit, or NO_TEXT."""
    client, label = _get_vlm_client()
    if client is None:
        return "", 0.0
    if not _vlm_rate_ok():
        return "", 0.0
    try:
        rgb = img
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        m = max(rgb.shape[:2])
        if m > 1280:
            s = 1280.0 / m
            rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * s)), max(1, int(rgb.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return "", 0.0
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        _vlm_stats["calls"] += 1
        resp = client.chat.completions.create(
            model=label,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _VLM_OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]}],
            max_tokens=400,
            temperature=0.0,
            timeout=_OCR_VLM_TIMEOUT,
        )
        txt = (resp.choices[0].message.content or "").strip()
        if not txt or txt.strip(" .!\"'").upper() in ("NO_TEXT", "NOTEXT", "NO TEXT"):
            return "", 0.0
        # guard against the model narrating instead of transcribing
        if len(txt) > 600 or re.match(r"(?i)^(the image|this (image|panel)|a comic|in this)", txt):
            return "", 0.0
        txt = _normalize_vlm_ocr_text(txt)
        if not txt:
            return "", 0.0
        _vlm_stats["hits"] += 1
        return txt, 0.90
    except Exception as exc:
        _vlm_stats["errors"] += 1
        logger.warning("VLM OCR call failed (%s)", exc)
        return "", 0.0


def _get_local_vlm_ocr():
    # type: () -> Tuple[Any, Any]
    """Lazy-load GOT-OCR2.0. Returns (model, processor) or (None, None).
    First call downloads ~1.4 GB to the HF cache and takes ~15 s to init;
    any failure (transformers/torch missing, download blocked, OOM) is
    swallowed and the tier stays disabled."""
    if _local_vlm["resolved"]:
        return _local_vlm["model"], _local_vlm["proc"]
    with _local_vlm_lock:
        if _local_vlm["resolved"]:
            return _local_vlm["model"], _local_vlm["proc"]
        _local_vlm["resolved"] = True
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            try:
                torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))
            except Exception:
                pass
            proc = AutoProcessor.from_pretrained(OCR_LOCAL_VLM_MODEL)
            model = AutoModelForImageTextToText.from_pretrained(
                OCR_LOCAL_VLM_MODEL, dtype=torch.float32, low_cpu_mem_usage=True).eval()
            _local_vlm["model"] = model
            _local_vlm["proc"] = proc
            logger.info("Local VLM OCR tier ENABLED (GOT-OCR2.0 = %s, cap=%s calls/process)",
                        OCR_LOCAL_VLM_MODEL, _OCR_LOCAL_VLM_MAX or "unlimited")
        except Exception as exc:
            logger.warning("Local VLM OCR tier: load failed (%s) — disabled", exc)
            _local_vlm["model"] = None
            _local_vlm["proc"] = None
    return _local_vlm["model"], _local_vlm["proc"]


def _run_local_vlm_ocr(img):
    # type: (np.ndarray) -> Tuple[str, float]
    """Transcribe one panel crop with local GOT-OCR2.0. ("", 0.0) on any
    failure or when the per-process call cap is hit. GOT-OCR2 is a pure OCR
    model (no instruction following), so its output needs only whitespace
    normalisation — but guard against it echoing an image-describing phrase
    just as the cloud tier does."""
    if not OCR_LOCAL_VLM:
        return "", 0.0
    with _local_vlm_lock:
        if _OCR_LOCAL_VLM_MAX and _local_vlm_stats["calls"] >= _OCR_LOCAL_VLM_MAX:
            _local_vlm_stats["skipped_cap"] += 1
            return "", 0.0
    model, proc = _get_local_vlm_ocr()
    if model is None:
        return "", 0.0
    try:
        import torch
        rgb = img
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        m = max(rgb.shape[:2])
        if m > 1280:
            s = 1280.0 / m
            rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * s)), max(1, int(rgb.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        from PIL import Image as _PILImage
        pil = _PILImage.fromarray(rgb)
        _local_vlm_stats["calls"] += 1
        inp = proc(pil, return_tensors="pt")
        # Serialize generate() — one shared torch model, and concurrent
        # decodes on 4 CPU cores only thrash.
        with _local_vlm_lock, torch.no_grad():
            ids = model.generate(**inp, do_sample=False,
                                 max_new_tokens=_OCR_LOCAL_VLM_TOKENS,
                                 stop_strings="<|im_end|>", tokenizer=proc.tokenizer)
        txt = proc.decode(ids[0, inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        if not txt or len(txt) > 600:
            return "", 0.0
        if re.match(r"(?i)^(the image|this (image|panel)|a comic|in this image)", txt):
            return "", 0.0
        txt = _normalize_vlm_ocr_text(txt)
        # GOT-OCR2 has no prompt to say "no text" — on a near-blank crop it
        # emits a stray letter or two. Require some real lettering before it
        # can outrank a classical candidate.
        alnum = re.sub(r"[^0-9A-Za-z]", "", txt)
        if len(alnum) < 3 or not re.search(r"[A-Za-z]{2,}", txt):
            return "", 0.0
        _local_vlm_stats["hits"] += 1
        return txt, 0.88
    except Exception as exc:
        _local_vlm_stats["errors"] += 1
        logger.warning("Local VLM OCR call failed (%s)", exc)
        return "", 0.0


def _vlm_zero_region_candidate(img):
    # type: (np.ndarray) -> Tuple[str, float]
    """For OCR_VLM_ON_ZERO_REGIONS mode: the classical detector found no text
    box at all. Skip obviously-blank / flat-art crops with a cheap pixel test
    before spending a VLM call, then transcribe."""
    if not OCR_VLM_ON_ZERO_REGIONS or _get_vlm_client()[0] is None:
        return "", 0.0
    try:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if float(g.std()) < 12.0:            # flat colour / gradient — no lettering
            return "", 0.0
        edges = cv2.Canny(g, 60, 160)
        if float((edges > 0).mean()) < 0.006:  # almost no high-freq detail
            return "", 0.0
    except Exception:
        pass
    return _run_vlm_ocr(img)


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

                # A confident RapidOCR result that is visibly garbled ("...
                # iiisn") must NOT short-circuit the cascade — let the
                # PaddleOCR variants + VLM tier(s) get a shot at reading it.
                _rapid_suspect = (rapid_status == "SUCCESS"
                                  and _looks_like_ocr_garble(rapid_text))
                _orig_garble = rapid_text
                if _rapid_suspect:
                    logger.info("RapidOCR result looks garbled (%r) — attempting recovery", rapid_text[:80])
                    # FAST recovery (free, ~0.2s): re-run RapidOCR with tighter
                    # box dilation. unclip 1.6 over-merges some panels' lines
                    # into garble ("IT SEEOH SS SWIIS" for "IT SEEMS AS THOUGH");
                    # a 1.3 pass reads those cleanly. Keep whichever is cleaner.
                    try:
                        _tr = _run_rapidocr_on_image(img, opts, unclip_override=1.3, box_override=0.5)
                        _tt, _tc, _tn = _merge_regions(_tr, is_ui_box=is_ui_box, engine="rapidocr")
                        if _tt and _tn and _ocr_text_quality(_tt) < _ocr_text_quality(rapid_text):
                            candidates.append({"text": _tt, "confidence": _tc, "regions": _tn,
                                               "provider": "rapidocr", "model": RAPIDOCR_MODEL_NAME,
                                               "variant": "tight_unclip"})
                            rapid_text, rapid_conf, rapid_count = _tt, _tc, _tn
                            rapid_status, rapid_quality, rapid_reason = _quality_status(_tt, _tc, _tn)
                            best_tuple = (rapid_text, rapid_conf, rapid_count, rapid_status,
                                          rapid_quality, list(candidates),
                                          "rapidocr_tight_unclip_recovered:" + rapid_reason)
                            logger.info("  tight-unclip pass cleaner -> %r", rapid_text[:80])
                            _learn_ocr_correction(_orig_garble, rapid_text,
                                                  _ocr_series_ctx["slug"], source="tight-unclip")
                            _rapid_suspect = _looks_like_ocr_garble(rapid_text)
                    except Exception as _te:
                        logger.debug("tight-unclip recovery failed (%s)", _te)

                if _rapid_suspect and (OCR_LOCAL_VLM or _get_vlm_client()[0] is not None
                                       or ocr is not None):
                    rapid_status = "UNCERTAIN"
                    rapid_reason = "confident_but_garbled_forcing_fallback:" + rapid_reason
                    # Deflate the stored quality so a clean PaddleOCR / Tesseract /
                    # VLM read can outrank this garble in best-candidate selection
                    # (its real 0.84 conf would otherwise beat a correct 0.75).
                    rapid_quality = min(rapid_quality, 0.34)
                    best_tuple = (rapid_text, rapid_conf, rapid_count, rapid_status,
                                  rapid_quality, list(candidates), rapid_reason)
                    # Ask GOT-OCR2 NOW (before PaddleOCR), because a downstream
                    # PaddleOCR "SUCCESS" that is clean-but-truncated would
                    # short-circuit the cascade before the VLM tier is reached.
                    if OCR_LOCAL_VLM:
                        _gt, _gc = _run_local_vlm_ocr(img)
                        if _gt and not _looks_like_ocr_garble(_gt):
                            _slug = _ocr_series_ctx["slug"]
                            _learn_ocr_correction(_orig_garble, _gt, _slug, source="got-ocr2")
                            _gt = _apply_known_corrections(_gt, _slug)
                            _gr = max(1, len(re.split(r"\s+/\s+", _gt)))
                            candidates.append({"text": _gt, "confidence": _gc, "regions": _gr,
                                               "provider": "vlm", "model": "got-ocr2",
                                               "variant": "garble_recovery"})
                            _gs, _gq, _grs = _quality_status(_gt, 0.95, _gr)
                            # Trust the clean VLM read over the flagged garble.
                            best_tuple = (_gt, 0.95, _gr, _gs, 0.95, list(candidates),
                                          "got_ocr2_recovered_garble:" + _grs)
                            logger.info("GOT-OCR2 recovered garbled panel -> %r", _gt[:80])
                            return best_tuple

                if rapid_status == "SUCCESS":
                    return best_tuple
                if rapid_count == 0:
                    if OCR_VLM_ON_ZERO_REGIONS:
                        _zt, _zc = _vlm_zero_region_candidate(img)
                        if _zt:
                            _zr = max(1, len(re.split(r"\s+/\s+", _zt)))
                            candidates.append({"text": _zt, "confidence": _zc, "regions": _zr,
                                               "provider": "vlm", "model": _get_vlm_client()[1],
                                               "variant": "zero_region"})
                            _zs, _zq, _zrs = _quality_status(_zt, _zc, _zr)
                            return (_zt, _zc, _zr, _zs, _zq, list(candidates),
                                    "vlm_zero_region_recovered:" + _zrs)
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

        # --- LAST RESORT: VLM OCR (opt-in) ---
        # Only when a text region WAS localized but no classical engine could
        # read it to SUCCESS — stylised SFX, warped/curved text, text on busy
        # art. Local GOT-OCR2.0 first (key-free, OCR_LOCAL_VLM=1), then the
        # cloud VLM (needs a key). Both return NO_TEXT / "" for untranslatable
        # / non-Latin lettering so garbage doesn't reach narration.
        try:
            bt = best_tuple
            reached_via_regions = bt is not None and (bt[2] > 0 or (bt[0] or "").strip())
            if bt is not None and bt[3] != "SUCCESS" and reached_via_regions:
                vlm_variants = []  # type: List[Tuple[str, str, float]]
                if OCR_LOCAL_VLM:
                    _lt, _lc = _run_local_vlm_ocr(img)
                    if _lt:
                        vlm_variants.append(("got-ocr2", _lt, _lc))
                if _get_vlm_client()[0] is not None:
                    _ct, _cc = _run_vlm_ocr(img)
                    if _ct:
                        vlm_variants.append((_get_vlm_client()[1], _ct, _cc))
                _classical_before = (bt[0] or "").strip()
                for _model_label, vlm_text, vlm_conf in vlm_variants:
                    _src = "got-ocr2" if "got" in _model_label.lower() else "cloud-vlm"
                    _learn_ocr_correction(_classical_before, vlm_text,
                                          _ocr_series_ctx["slug"], source=_src)
                    vlm_text = _apply_known_corrections(vlm_text, _ocr_series_ctx["slug"])
                    vlm_regions = max(1, len(re.split(r"\s+/\s+", vlm_text)))
                    candidates.append({
                        "text": vlm_text, "confidence": vlm_conf, "regions": vlm_regions,
                        "provider": "vlm", "model": _model_label, "variant": "standard",
                    })
                    vlm_status, vlm_quality, vlm_reason = _quality_status(vlm_text, vlm_conf, vlm_regions)
                    if best_tuple is None or vlm_quality > best_tuple[4]:
                        best_tuple = (vlm_text, vlm_conf, vlm_regions, vlm_status, vlm_quality,
                                      list(candidates),
                                      "vlm_fallback_beat_prior_engines:" + vlm_reason)
        except Exception as vlm_err:
            logger.warning("VLM OCR cascade tier failed gracefully: %s", vlm_err)

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
    _vlm_c, _vlm_l = _get_vlm_client()
    return HealthResponse(
        status="ok" if SERVICE_STATE == ServiceState.READY else ("initializing" if SERVICE_STATE == ServiceState.INITIALIZING else "degraded"),
        model=active_model,
        ready=(RAPIDOCR_READY or MODEL_READY),
        state=SERVICE_STATE,
        error=INIT_ERROR,
        rapidocr_ready=RAPIDOCR_READY,
        paddleocr_ready=MODEL_READY,
        vlm_fallback=(_vlm_l or None) if _vlm_c is not None else None,
        vlm_stats=dict(_vlm_stats) if _vlm_c is not None else None,
        local_vlm=(OCR_LOCAL_VLM_MODEL if OCR_LOCAL_VLM else None),
        local_vlm_stats=(dict(_local_vlm_stats) if OCR_LOCAL_VLM else None),
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


# How many images a /ocr/batch request processes at once. Now that RapidOCR
# runs through a pool of independent engines (see RAPIDOCR_POOL_SIZE) this is
# real parallelism, not just queue depth. Default 3 on this 4-core box (bench:
# pool=3 ~1.85x over serial; 4+ flattens). Frames that fall through to the
# PaddleOCR tier still serialize on _inference_lock — that's intentional.
OCR_CONCURRENCY = max(1, int(os.environ.get("OCR_CONCURRENCY", "3")))
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

        # Known-mistakes correction dictionary — final pass over the winning
        # text (whichever engine produced it), so per-series fixes also catch
        # VLM output and phrase-level fixes hit the merged result.
        _corr = _apply_known_corrections(text, _ocr_series_ctx["slug"])
        if _corr != text:
            reason = (reason or "") + "|dict"
            text = _corr

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
    _ocr_series_ctx["slug"] = _slugify_series(
        request.series or os.environ.get("RECAP_OCR_SERIES", ""))
    logger.info("[Batch OCR] Received request for %d images (concurrency limit: %d, series: %s)",
                total, OCR_CONCURRENCY, _ocr_series_ctx["slug"])

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

    # F2: hand freed pages back to the OS so RSS doesn't ratchet across a
    # many-chapter job (one batch == one chapter).
    _reclaim_memory("post-batch")

    return BatchOCRResponse(
        results=results,
        model=_active_model_name(),
        processing_time_ms=elapsed_ms,
    )


@app.post("/ocr/vlm")
async def ocr_vlm(request: Base64OCRRequest):
    """Force a GOT-OCR2 pixel re-read of one crop, ignoring the per-process
    call cap. For the offline accuracy / dict-training pass — NOT on the hot
    path. Also returns the classical read + whether the classical looked
    garbled, so the caller can pick a winner."""
    import asyncio
    img_array = _decode_base64_image(request.image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    _saved = _local_vlm_stats["calls"]
    try:
        _local_vlm_stats["calls"] = 0            # bypass the cap for this call
        vlm_text, vlm_conf = await asyncio.to_thread(_run_local_vlm_ocr, img_array)
    finally:
        _local_vlm_stats["calls"] = _saved + 1
    try:
        _regions = await asyncio.to_thread(
            _run_rapidocr_on_image, img_array, request.options)
        classic, _cc, _cn = _merge_regions(_regions, engine="rapidocr")
    except Exception:
        classic = ""
    classic = _repair_and_denoise(_clean_and_normalize_ocr_text(classic or ""))
    return {
        "vlm_text": vlm_text or "",
        "vlm_conf": vlm_conf,
        "classic_text": classic,
        "classic_garble": _looks_like_ocr_garble(classic),
        "model": OCR_LOCAL_VLM_MODEL,
    }


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
