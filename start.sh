#!/bin/bash
# start.sh — the ONE script that starts everything on Oracle Cloud VM:
# PaddleOCR (RapidOCR primary + PaddleOCR fallback, port 3002) +
# pipeline-service (port 3001) + Next.js (port 3000) + watchdog.
#
# This used to be two separate, overlapping scripts (start.sh and
# start-services.sh) with different levels of robustness in different
# areas — start.sh had the TTS/OCR dependency bootstrap and watchdog
# launch, start-services.sh had more robust process killing and
# readiness polling. Merged into this single script so there's exactly
# one way to start the stack and nothing can drift between two copies of
# similar logic again. install-systemd.sh's unit file calls this script
# by name; start-services.sh has been removed.

set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Guard against two overlapping invocations racing each other. Real-world
# trigger seen in production: a user rapid-clicking "retry" on a stuck job
# while services were still coming up — each click can independently
# trigger a service-start attempt, and two copies of this script running
# at once both try to kill/rebind the same ports, leaving one instance's
# process orphaned/zombied and bound to a port while the other believes
# it owns that port ("address already in use" fighting a stale process,
# seen directly in production logs). flock serializes the whole script
# body: a second invocation waits for the first to finish instead of
# interleaving kill/start calls with it. Bounded wait, not indefinite, so
# a genuinely wedged first instance doesn't hang every later caller
# forever.
LOCK_FILE="$PROJECT_DIR/.start-services.lock"
exec 200>"$LOCK_FILE"
if ! flock -w 180 200; then
    echo "❌ Another start.sh is already running and didn't finish within 180s — aborting to avoid racing it. Check for a wedged process, or just wait and retry." >&2
    exit 1
fi

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# Make setup.sh's venv (created at $PROJECT_DIR/.venv) the one every
# process this script launches actually uses, regardless of whether the
# calling shell happened to have it activated already. Without this,
# paddleocr-service's own start.sh and pipeline-service's spawned
# `python3` calls for master_pipeline.py (see PYTHON_BIN in
# mini-services/pipeline-service/lib.ts) both silently fall back to
# whatever bare `python3` resolves to on PATH — correct if this script
# was run from a shell where the venv was manually activated first, but
# broken with no error at all (just missing-import failures downstream)
# from any other context, e.g. a systemd unit or a fresh SSH session
# after a reboot. Safe no-op if this venv doesn't exist yet.
if [ -x "$PROJECT_DIR/.venv/bin/python3" ]; then
    export PATH="$PROJECT_DIR/.venv/bin:$PATH"
    export VIRTUAL_ENV="$PROJECT_DIR/.venv"
    echo "  (using venv: $PROJECT_DIR/.venv)"
fi

echo "🚀 Starting Manhwa Recap Studio..."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# DEPENDENCY BOOTSTRAP
#
# Production-mode TTS (pipeline/master_pipeline.py) refuses to silently
# substitute silence when Piper and eSpeak-NG are both missing — it raises
# and fails the chapter/job instead, by design (see production plan: "never
# turn TTS failure into success by writing silence"). That's the correct
# behavior, but it means a box that never had these installed will fail
# every real job with a confusing runtime error instead of failing loudly
# and obviously at startup. Everything below is idempotent (checks before
# installing) so re-running start.sh is fast and side-effect-free once
# everything is already present.
# ═══════════════════════════════════════════════════════════════════════════
echo "🔧 Checking pipeline dependencies..."

PIPER_VOICE_DIR="$PROJECT_DIR/pipeline/voices"
PIPER_VOICE_NAME="en_US-ryan-high"
PIPER_VOICE_MODEL_PATH="$PIPER_VOICE_DIR/${PIPER_VOICE_NAME}.onnx"

# --- System packages: eSpeak-NG (TTS fallback) + Tesseract (OCR last-resort fallback) ---
_missing_pkgs=()
command -v espeak-ng >/dev/null 2>&1 || _missing_pkgs+=("espeak-ng")
command -v tesseract >/dev/null 2>&1 || _missing_pkgs+=("tesseract-ocr")

if [ ${#_missing_pkgs[@]} -gt 0 ]; then
    echo "  Installing missing system packages: ${_missing_pkgs[*]}"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq "${_missing_pkgs[@]}" \
            || echo "  ⚠️  apt-get install failed for one or more packages — TTS/OCR fallback tiers may be degraded"
    elif command -v dnf >/dev/null 2>&1; then
        # RHEL/Oracle Linux package names differ slightly for tesseract.
        _rhel_pkgs=()
        for p in "${_missing_pkgs[@]}"; do
            [ "$p" = "tesseract-ocr" ] && _rhel_pkgs+=("tesseract") || _rhel_pkgs+=("$p")
        done
        sudo dnf install -y -q "${_rhel_pkgs[@]}" \
            || echo "  ⚠️  dnf install failed for one or more packages — TTS/OCR fallback tiers may be degraded"
    else
        echo "  ⚠️  No supported package manager found (apt/dnf) — install manually: ${_missing_pkgs[*]}"
    fi
else
    echo "  ✅ espeak-ng, tesseract-ocr already installed"
fi

# --- Piper TTS (primary production TTS engine) ---
# master_pipeline.py's _synthesize_with_piper() does `shutil.which("piper")`
# and reads the PIPER_VOICE_MODEL env var — both must be set correctly in
# THIS script's environment before pipeline-service is launched below,
# since pipeline-service spawns master_pipeline.py with `env: {...process.env}`
# (a straight passthrough of whatever it itself was started with).
#
# NOTE: this deliberately does NOT `pip install piper-tts`. piper-tts's
# native dependency (piper-phonemize) only ships wheels for Python 3.9+ —
# there is no cp38 wheel at all. On a box with an older root venv (e.g.
# Ubuntu 20.04's stock Python 3.8, as opposed to a separately-built 3.11
# venv some services here use), pip's resolver falls back to ancient
# piper-tts 1.1.0/1.2.0 releases and then fails with a version conflict
# between them, not a solvable dependency issue. Piper also publishes a
# fully self-contained prebuilt binary (bundles its own espeak-ng,
# libpiper_phonemize, and libonnxruntime — zero Python involved), which
# sidesteps this entirely and works regardless of which Python the box
# happens to have. That's what this installs instead.
PIPER_DIR="$PROJECT_DIR/pipeline/piper"
PIPER_BIN="$PIPER_DIR/piper/piper"
PIPER_ARCH="$(uname -m)"
case "$PIPER_ARCH" in
    x86_64)  PIPER_ASSET="piper_linux_x86_64.tar.gz" ;;
    aarch64) PIPER_ASSET="piper_linux_aarch64.tar.gz" ;;
    armv7l)  PIPER_ASSET="piper_linux_armv7l.tar.gz" ;;
    *)       PIPER_ASSET="" ;;
esac

if [ -x "$PIPER_BIN" ]; then
    echo "  ✅ piper binary already installed at $PIPER_BIN"
elif [ -z "$PIPER_ASSET" ]; then
    echo "  ⚠️  No prebuilt Piper release for architecture '$PIPER_ARCH' — will fall back to eSpeak-NG at render time"
else
    echo "  Downloading Piper prebuilt binary ($PIPER_ASSET)..."
    mkdir -p "$PIPER_DIR"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/${PIPER_ASSET}"
    if curl -fsSL -o "$PIPER_DIR/piper.tar.gz" "$PIPER_URL" \
        && tar -xzf "$PIPER_DIR/piper.tar.gz" -C "$PIPER_DIR" \
        && rm -f "$PIPER_DIR/piper.tar.gz"; then
        echo "  ✅ Piper binary installed"
    else
        echo "  ⚠️  Piper binary download/extract failed — will fall back to eSpeak-NG at render time"
    fi
fi

if [ ! -f "$PIPER_VOICE_MODEL_PATH" ]; then
    echo "  Downloading Piper voice model ($PIPER_VOICE_NAME)..."
    mkdir -p "$PIPER_VOICE_DIR"
    PIPER_VOICE_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
    if curl -fsSL -o "$PIPER_VOICE_MODEL_PATH" "$PIPER_VOICE_BASE_URL/${PIPER_VOICE_NAME}.onnx" \
        && curl -fsSL -o "$PIPER_VOICE_MODEL_PATH.json" "$PIPER_VOICE_BASE_URL/${PIPER_VOICE_NAME}.onnx.json"; then
        echo "  ✅ Piper voice model downloaded"
    else
        echo "  ⚠️  Piper voice model download failed — will fall back to eSpeak-NG at render time"
        rm -f "$PIPER_VOICE_MODEL_PATH" "$PIPER_VOICE_MODEL_PATH.json"
    fi
else
    echo "  ✅ Piper voice model already present: $PIPER_VOICE_MODEL_PATH"
fi

# Make piper's binary reachable and tell master_pipeline.py which model to
# use, for every process this script launches from here on. piper's own
# bundled shared libraries (libespeak-ng.so, libonnxruntime.so, etc.) live
# alongside the binary in the same directory, so no LD_LIBRARY_PATH needed
# — the binary was built with a relative rpath for exactly this layout.
if [ -x "$PIPER_BIN" ]; then
    export PATH="$(dirname "$PIPER_BIN"):$PATH"
fi
if [ -f "$PIPER_VOICE_MODEL_PATH" ]; then
    export PIPER_VOICE_MODEL="$PIPER_VOICE_MODEL_PATH"
fi

echo ""

# --- Manga panel/text detector (YOLO26-nano, fine-tuned on Manga109-s) ---
# See the YOLO_TEXT_MODEL_PATH comment in pipeline/master_pipeline.py for
# the full reasoning. Apache-2.0, ~15MB, benchmarked ~100-180ms/image on
# CPU. Purely additive: if this download fails or is skipped, the
# pipeline silently falls back to its pre-existing pixel-only content
# mask (logged once, not an error) -- same "degrade, don't crash"
# pattern as the Piper download above.
YOLO_TEXT_MODEL_DIR="$PROJECT_DIR/pipeline/models"
YOLO_TEXT_MODEL_PATH="$YOLO_TEXT_MODEL_DIR/manga_panel_detector_fp32.pt"
if [ ! -f "$YOLO_TEXT_MODEL_PATH" ]; then
    echo "  Downloading manga panel/text detection model (YOLO26-nano)..."
    mkdir -p "$YOLO_TEXT_MODEL_DIR"
    YOLO_TEXT_MODEL_URL="https://huggingface.co/leoxs22/manga-panel-detector-yolo26n/resolve/main/manga_panel_detector_fp32.pt"
    if curl -fsSL -o "$YOLO_TEXT_MODEL_PATH" "$YOLO_TEXT_MODEL_URL"; then
        # Model card lists the source file as 14.8MB -- sanity-check the
        # download landed roughly that size rather than e.g. an HTML error
        # page saved under the .pt filename (curl -f catches most HTTP
        # error statuses already, but not a redirect to a valid-but-wrong
        # small page).
        DOWNLOADED_SIZE=$(stat -c%s "$YOLO_TEXT_MODEL_PATH" 2>/dev/null || stat -f%z "$YOLO_TEXT_MODEL_PATH" 2>/dev/null || echo 0)
        if [ "$DOWNLOADED_SIZE" -lt 1000000 ]; then
            echo "  ⚠️  Downloaded file is only ${DOWNLOADED_SIZE} bytes (expected ~15MB) — discarding, falling back to pixel-only panel/caption detection"
            rm -f "$YOLO_TEXT_MODEL_PATH"
        else
            echo "  ✅ Manga panel/text model downloaded (${DOWNLOADED_SIZE} bytes)"
        fi
    else
        echo "  ⚠️  Manga panel/text model download failed — falling back to pixel-only panel/caption detection"
        rm -f "$YOLO_TEXT_MODEL_PATH"
    fi
else
    echo "  ✅ Manga panel/text model already present: $YOLO_TEXT_MODEL_PATH"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# SHARED PIPELINE_SECRET
#
# pipeline-service's /internal/* endpoints and its socket.io connection both
# require a PIPELINE_SECRET (see mini-services/pipeline-service/index.ts,
# checkAuth). If unset, pipeline-service generates a random one at startup
# that nothing else knows — the Next.js API routes (jobs POST/cancel) and
# the browser's socket.io client both need the SAME value or every request
# is silently rejected with 401 (which the Next.js side then reported as
# "Pipeline service is not running", even though the service was up).
# Generate once and persist so it's stable across restarts, and export it
# for every process this script launches from here on: PIPELINE_SECRET for
# the two Node/Python server processes, and NEXT_PUBLIC_PIPELINE_SECRET
# (same value) so it gets baked into the browser bundle at build time below.
# ═══════════════════════════════════════════════════════════════════════════
SECRET_FILE="$PROJECT_DIR/.pipeline-secret"
if [ ! -s "$SECRET_FILE" ]; then
    echo "🔑 Generating pipeline secret (first run)..."
    head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32 > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
fi
export PIPELINE_SECRET="$(cat "$SECRET_FILE")"
export NEXT_PUBLIC_PIPELINE_SECRET="$PIPELINE_SECRET"

# mini-services/pipeline-service/lib.ts computes
# `PROJECT_ROOT = process.env.PROJECT_ROOT || process.cwd()` at MODULE
# LOAD TIME (a top-level const). pipeline-service's own index.ts does try
# to load the project-root .env (which setup.sh does populate with a
# correct PROJECT_ROOT=... line) via an absolute path immune to cwd —
# but ES module import resolution evaluates ALL of a file's imports
# (including a later `import ... from './lib'`) before ANY of that
# file's own top-level statements run, regardless of where the import
# is textually positioned relative to the .env-loading call. Confirmed
# directly with a minimal reproduction: lib.ts's PROJECT_ROOT constant
# is locked in from its process.cwd() fallback before index.ts's
# loadDotenv() call ever executes. Since this script `cd`s into
# mini-services/pipeline-service/ before launching it (correctly, since
# that's where `bun run start` needs to run from), that fallback
# resolves to the wrong directory — confirmed directly against a real
# job: every path master_pipeline.py was invoked with (the script path
# itself, --input-dir, --output, --work-dir, --progress-file) had
# "mini-services/pipeline-service" wrongly appended into the middle of
# it. Exporting this explicitly here, the same way PIPELINE_SECRET
# already is, sidesteps the module-ordering issue entirely: shell-
# exported environment variables exist before any JS/TS code runs at
# all, so process.env.PROJECT_ROOT is correct from the very first line
# of module evaluation, no import-order subtlety involved.
export PROJECT_ROOT="$PROJECT_DIR"
# Same reasoning and same module-load-timing bug as PROJECT_ROOT above —
# lib.ts's DATA_DIR constant has its own independent process.cwd()-based
# fallback (not derived from PROJECT_ROOT), so fixing PROJECT_ROOT alone
# does not fix this one too. Confirmed directly: the actual broken job
# command line showed every data/jobs/... path built from the wrong
# base directory, matching this constant's fallback exactly.
export DATA_DIR="$PROJECT_DIR/data"

# Same bug class, a third victim: lib.ts's PYTHON_BIN falls back to the
# bare string 'python3' via the same too-late process.env read (see
# PROJECT_ROOT above). Since it's a bare command name, not a path,
# resolving it depends entirely on process.env.PATH at the moment
# spawnSync actually runs it -- and index.ts's loadDotenv({override:
# true}) OVERWRITES process.env.PATH with .env's own PATH= value
# (deliberately just piper's directory + standard system dirs, written
# by setup.sh, which predates this venv-PATH-prepending export and was
# never meant to duplicate it) once it finally runs. That value does not
# include this venv's bin/ directory, so by job-processing time the
# bare 'python3' resolves to the system interpreter instead of this
# venv's -- confirmed directly against a real job: it failed with
# "ModuleNotFoundError: No module named 'PIL'", a package that is
# installed in the venv (verified during setup.sh's own dependency
# verification step) but not system-wide. Exporting an ABSOLUTE path
# here, rather than just fixing the bare command name, sidesteps
# process.env.PATH entirely regardless of what later overwrites it --
# spawnSync uses this exact binary directly with no PATH search
# involved at all.
if [ -x "$PROJECT_DIR/.venv/bin/python3" ]; then
    export PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
fi

# ═══════════════════════════════════════════════════════════════════════════
# PROCESS MANAGEMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

kill_service() {
    # Plain `sleep 1` after pkill isn't reliable: uvicorn/bun install SIGTERM
    # handlers for graceful shutdown, so a process mid-request (or mid-init,
    # once past module load) can take longer than 1s to actually release its
    # port -- the next step then binds too early, the new process dies on
    # "address already in use", and the OLD process silently keeps serving
    # (exactly what happened on this box: health checks kept answering from
    # a stale process while the new one's log showed nothing but a bind
    # error). Poll until the process is actually gone instead of guessing.
    #
    # Every pkill/kill below ends in `|| true`: under this script's `set -e`,
    # pkill returns 1 (and would otherwise abort the whole script) whenever
    # there's simply nothing matching to kill -- the normal case on a fresh
    # start or after a clean shutdown. Confirmed directly: without `|| true`
    # here, the very first call to this function silently killed the entire
    # script on any run where nothing was already running.
    local pattern="$1"
    pkill -f "$pattern" 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f "$pattern" > /dev/null || return 0
        sleep 0.5
    done
    # Still alive after 10s of SIGTERM -- force it so we don't proceed with
    # two processes fighting over the same port.
    pkill -9 -f "$pattern" 2>/dev/null || true
    sleep 1
}

kill_port() {
    # kill_service matches by process command-line pattern, which silently
    # misses anything started with a slightly different invocation (a
    # different interpreter path, an old checkout, a leftover from before a
    # refactor). Whatever is actually bound to the port is unambiguous --
    # use that as the ground truth instead of guessing a name pattern.
    local port="$1"
    local pids
    pids=$(ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" | grep -oP 'pid=\K[0-9]+' | sort -u)
    if [ -n "$pids" ]; then
        echo "  (killing stale process(es) on port $port: $pids)"
        kill -9 $pids 2>/dev/null || true
    fi
    for _ in $(seq 1 20); do
        ss -ltn 2>/dev/null | grep -qE "[:.]${port}[[:space:]]" || return 0
        sleep 0.5
    done
}

echo "=== Starting Manhwa Recap Studio services ==="

# ─────────────────────────────────────────────────────────────────────────
# 1. PaddleOCR service (port 3002) — RapidOCR PP-OCRv6 primary,
#    PaddleOCR PP-OCRv4 fallback (see mini-services/paddleocr-service/main.py)
# ─────────────────────────────────────────────────────────────────────────
kill_service "uvicorn main:app"
kill_service "python3 main.py"
kill_port 3002
echo "[1/3] Starting PaddleOCR on port 3002..."
cd "$PROJECT_DIR/mini-services/paddleocr-service"
# Go through this service's own start.sh, not `python3 main.py` directly —
# that nested start.sh binds uvicorn to 127.0.0.1 only ("C16 FIX": external
# access should go through Caddy). Calling main.py directly binds 0.0.0.0
# (see its own `if __name__ == "__main__"` block), exposing OCR publicly
# with no auth.
setsid bash start.sh > "$LOG_DIR/paddleocr.log" 2>&1 < /dev/null &
PADDLEOCR_PID=$!
cd "$PROJECT_DIR"
# Poll for real readiness instead of a fixed sleep + a substring grep that
# matches "ready":false as happily as "ready":true. Model init (RapidOCR's
# ONNX warmup, or PaddleOCR's self-healing kwarg retries / a cold model
# download) can legitimately take longer than a fixed few seconds, so give
# it a real timeout instead of a guess.
ready=0
for _ in $(seq 1 60); do
    if curl -s http://localhost:3002/health | grep -q '"ready":[[:space:]]*true'; then
        ready=1
        break
    fi
    sleep 2
done
if [ "$ready" = "1" ]; then
    echo "  ✅ PaddleOCR ready (PID: $PADDLEOCR_PID)"
else
    echo "  ❌ PaddleOCR failed - check $LOG_DIR/paddleocr.log"
fi

# ─────────────────────────────────────────────────────────────────────────
# 2. Pipeline service (port 3001)
# ─────────────────────────────────────────────────────────────────────────
kill_service "bun index.ts"
kill_port 3001
echo "[2/3] Starting Pipeline on port 3001..."
cd "$PROJECT_DIR/mini-services/pipeline-service"
setsid bun run start > "$LOG_DIR/pipeline.log" 2>&1 < /dev/null &
PIPELINE_PID=$!
cd "$PROJECT_DIR"
ready=0
for _ in $(seq 1 20); do
    if curl -s http://localhost:3001/internal/health 2>/dev/null | grep -q "ok"; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" = "1" ]; then
    echo "  ✅ Pipeline-service running (PID: $PIPELINE_PID)"
else
    echo "  ❌ Pipeline-service failed - check $LOG_DIR/pipeline.log"
fi

# ─────────────────────────────────────────────────────────────────────────
# 3. Next.js (port 3000)
# ─────────────────────────────────────────────────────────────────────────
kill_service "server.js"
kill_port 3000
echo "[3/3] Starting Next.js on port 3000..."
cd "$PROJECT_DIR"

# Clear stale .next build cache to fix "Failed to find Server Action" errors.
# This happens when the production build has a server-action registry that
# no longer matches the current source (actions added/removed since last
# build). Checking only "does server.js exist at all" misses the common
# case here — a `git pull` that changed source without rebuilding — so
# compare against the checked-out commit instead: any commit change means
# source may have moved and the build must be regenerated, not just "does
# a build exist at all".
CURRENT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo "")"
BUILD_STAMP_FILE=".next/standalone/.build-commit"
NEED_BUILD=0
if [ ! -f ".next/standalone/server.js" ]; then
    NEED_BUILD=1
    echo "  .next/standalone/server.js not found — building..."
elif [ -n "$CURRENT_COMMIT" ] && [ "$(cat "$BUILD_STAMP_FILE" 2>/dev/null)" != "$CURRENT_COMMIT" ]; then
    NEED_BUILD=1
    echo "  .next build is stale (source changed since last build) — rebuilding..."
fi
if [ "$NEED_BUILD" = "1" ]; then
    rm -rf .next
    # `if !` (not a bare statement + separate `$?` check) is required here
    # under this script's `set -e`: a bare failing `bun run build` would
    # abort the whole script immediately, before ever reaching a
    # subsequent `if [ $? -ne 0 ]` check.
    if ! bun run build > "$LOG_DIR/nextjs-build.log" 2>&1; then
        echo "  ⚠️  Build failed — check $LOG_DIR/nextjs-build.log"
    else
        echo "  ✅ Build succeeded"
        [ -n "$CURRENT_COMMIT" ] && mkdir -p "$(dirname "$BUILD_STAMP_FILE")" && echo "$CURRENT_COMMIT" > "$BUILD_STAMP_FILE"
    fi
fi
# H2 FIX: Bind Next.js to localhost only (Caddy proxies externally)
#
# PORT=3000 is set explicitly here, not left to inherit from the
# environment, because setup.sh writes a shared PORT=3001 into the
# project-root .env (intended for pipeline-service, which reads it from
# its own process environment). Next.js's standalone server has built-in
# .env auto-loading and runs with its cwd at the project root (unlike
# pipeline-service, which runs from its own subdirectory and never sees
# that file at all), so without an explicit override here it picks up
# that same PORT=3001, collides with pipeline-service which is already
# bound to it, and crashes immediately with EADDRINUSE -- confirmed
# directly against a real deployment: `bun .next/standalone/server.js`
# run standalone failed with exactly "Failed to start server. Is port
# 3001 in use?", and the watchdog log showed it restart-looping
# indefinitely every cycle for over 10 hours as a result.
HOSTNAME=127.0.0.1 PORT=3000 setsid bun .next/standalone/server.js > "$LOG_DIR/nextjs.log" 2>&1 < /dev/null &
NEXT_PID=$!
# Real HTTP status check with a poll loop (a cold Turbopack/Next start can
# take a few seconds) instead of a fixed sleep + an always-false substring
# check.
ready=0
for _ in $(seq 1 20); do
    # `|| true` is required here under this script's `set -e`: unlike the
    # PaddleOCR/pipeline-service checks above (curl piped into grep,
    # inside an if-condition -- exempt from set -e either way), this one
    # assigns curl's own output to a variable via bare command
    # substitution with no pipe. curl exits 7 ("failed to connect") on
    # every iteration until Next.js actually binds the port -- which is
    # true by definition on the very first loop iteration, moments after
    # backgrounding it. Confirmed directly: without `|| true`, this
    # single line killed the entire script (and therefore the whole
    # systemd-managed stack, including the already-running PaddleOCR and
    # pipeline-service) on every single start, deterministically, before
    # ever reaching the `if [ "$code" = "200" ]` check below it.
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/ 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" = "1" ]; then
    echo "  ✅ Next.js running (PID: $NEXT_PID)"
else
    echo "  ⚠️  Next.js may not be ready yet - check $LOG_DIR/nextjs.log"
fi

# ─────────────────────────────────────────────────────────────────────────
# Watchdog: everything above only starts each service once — nothing was
# watching them afterward, so a native crash (the PaddleOCR PIR-interpreter
# SIGSEGV in particular — see mini-services/paddleocr-service/main.py) took
# OCR down permanently until someone noticed and reran this script by hand.
# Restart it in place so OCR — and the other two services — recover
# automatically within seconds instead.
# ─────────────────────────────────────────────────────────────────────────
pkill -f "watchdog.sh" 2>/dev/null || true
WATCHDOG_LOG_DIR="$LOG_DIR" setsid bash "$PROJECT_DIR/watchdog.sh" > "$LOG_DIR/watchdog.log" 2>&1 < /dev/null &
echo "  ✅ Watchdog started — auto-restarts any of the 3 services if it dies (see $LOG_DIR/watchdog.log)"

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_VM_IP")
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎉  Manhwa Recap Studio is running!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  🌐 Website:  http://$PUBLIC_IP"
echo "  📊 API:      http://$PUBLIC_IP/api/stats"
echo "  🔧 Pipeline: http://$PUBLIC_IP:3001/internal/health"
echo "  🔎 OCR:      http://$PUBLIC_IP:3002/health"
echo ""
echo "  To stop:     fuser -k 3000/tcp; pkill -f 'index.ts'; pkill -f 'uvicorn main:app'; pkill -f 'watchdog.sh'"
echo "  To restart:  bash start.sh"
echo "  Logs:        tail -f $LOG_DIR/nextjs.log $LOG_DIR/pipeline.log $LOG_DIR/paddleocr.log $LOG_DIR/watchdog.log"
echo ""
echo "═══════════════════════════════════════════════════════════════"
