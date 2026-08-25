#!/usr/bin/env bash
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Manhwa Recap Studio v3 — One-Click Oracle Cloud Setup Script       ║
# ║                                                                      ║
# ║  Run after cloning:                                                  ║
# ║    git clone <repo-url> manhwa-recap-studio                          ║
# ║    cd manhwa-recap-studio                                            ║
# ║    chmod +x setup.sh && ./setup.sh                                   ║
# ║                                                                      ║
# ║  Supports: Oracle Linux, Ubuntu, Debian, RHEL, Amazon Linux         ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
set -uo pipefail
# NOTE: We intentionally do NOT use `set -e` because some package
# installs legitimately fail on certain OSes and we handle them.

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1" >&2; }
log_step()  { echo -e "${BLUE}${BOLD}[STEP $1]${NC} $2"; }

# ── Guard: must run from project root ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "package.json" ]]; then
    log_error "package.json not found. Run this script from the project root."
    exit 1
fi

echo -e "${CYAN}${BOLD}"
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║     Manhwa Recap Studio v3 — Setup Script          ║"
echo "  ║     Oracle Cloud Auto-Configuration                 ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_VISION_MODEL="${OLLAMA_VISION_MODEL:-llava:7b}"
OLLAMA_TEXT_MODEL="${OLLAMA_TEXT_MODEL:-llama3.2:3b}"
PORT_WEB=3000
PORT_PIPELINE=3001
PORT_CADDY=80
PROJECT_DIR="$(pwd)"
PYTHON_VENV="$PROJECT_DIR/.venv"
PYTHON_BIN="$PYTHON_VENV/bin/python3"
# FIX #7: Initialize CADDY_ENABLED upfront so set -u never crashes
CADDY_ENABLED=false

# ── Detect OS ─────────────────────────────────────────────────────────────────
detect_os() {
    if [[ -f /etc/oracle-release ]]; then
        echo "oracle"
    elif [[ -f /etc/lsb-release ]] && grep -q "Ubuntu" /etc/lsb-release 2>/dev/null; then
        echo "ubuntu"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/redhat-release ]]; then
        echo "rhel"
    elif [[ -f /etc/amazon-linux-release ]]; then
        echo "amazon"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
log_info "Detected OS: $OS"

# ── Package manager detection ─────────────────────────────────────────────────
is_deb=false

if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
    is_deb=true
    pkg_update()  { sudo apt-get update -qq 2>&1 | tail -1; }
    pkg_install() { sudo apt-get install -y -qq "$@" 2>&1 | tail -1 || log_warn "Some packages failed to install"; }
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
    pkg_update()  { sudo dnf update -q 2>&1 | tail -1 || true; }
    pkg_install() { sudo dnf install -y -q "$@" 2>&1 | tail -1 || log_warn "Some packages failed to install"; }
elif command -v yum &>/dev/null; then
    PKG_MGR="yum"
    pkg_update()  { sudo yum update -q 2>&1 | tail -1 || true; }
    pkg_install() { sudo yum install -y -q "$@" 2>&1 | tail -1 || log_warn "Some packages failed to install"; }
else
    log_error "No supported package manager found (apt/dnf/yum)"
    exit 1
fi

log_info "Package manager: $PKG_MGR (${OS})"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0a: Disk Space Check (FIX #8)
# ═══════════════════════════════════════════════════════════════════════════════
log_step "0a" "Checking disk space..."

AVAILABLE_GB=$(LC_ALL=C df -BG "$PROJECT_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
MIN_DISK_GB=12  # Ollama models (~7GB) + torch (~2GB) + bun deps + headroom

if [[ -z "$AVAILABLE_GB" || "$AVAILABLE_GB" -lt "$MIN_DISK_GB" ]]; then
    log_warn "Low disk space: ${AVAILABLE_GB:-unknown}GB available (need ${MIN_DISK_GB}GB+)"
    log_warn "Ollama model pulls and pip installs may fail. Consider adding a block volume."
    echo -n "  Continue anyway? [y/N] "
    read -r REPLY
    if [[ "$REPLY" != "y" && "$REPLY" != "Y" ]]; then
        log_error "Aborted due to insufficient disk space."
        exit 1
    fi
else
    log_info "Disk space OK: ${AVAILABLE_GB}GB available"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0b: Swap File (Oracle Cloud ARM can get tight with Ollama models)
# ═══════════════════════════════════════════════════════════════════════════════
log_step "0b" "Checking swap space..."

CURRENT_SWAP_KB=$(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
CURRENT_SWAP_KB="${CURRENT_SWAP_KB:-0}"
if [[ "$CURRENT_SWAP_KB" -lt 2097152 ]]; then
    # Less than 2GB swap — create a 4GB swap file
    SWAPFILE="/swapfile"
    if [[ ! -f "$SWAPFILE" ]]; then
        log_info "Creating 4GB swap file (current: $((CURRENT_SWAP_KB / 1024))MB)"
        sudo fallocate -l 4G "$SWAPFILE" 2>/dev/null || sudo dd if=/dev/zero of="$SWAPFILE" bs=1M count=4096 status=progress 2>/dev/null
        sudo chmod 600 "$SWAPFILE"
        sudo mkswap "$SWAPFILE" >/dev/null
        sudo swapon "$SWAPFILE" 2>/dev/null && log_info "4GB swap activated" || log_warn "Could not activate swap (may need root)"
        # Persist across reboot
        if ! grep -q "$SWAPFILE" /etc/fstab 2>/dev/null; then
            echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
            log_info "Swap entry added to /etc/fstab"
        fi
    else
        log_info "Swap file already exists ($((CURRENT_SWAP_KB / 1024))MB)"
    fi
else
    log_info "Swap space adequate ($((CURRENT_SWAP_KB / 1024))MB)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: System Packages
# ═══════════════════════════════════════════════════════════════════════════════
log_step 1 "Installing system packages..."

pkg_update

if $is_deb; then
    # ── Debian/Ubuntu packages ──
    pkg_install curl wget git unzip zstd build-essential \
        ffmpeg python3 python3-pip python3-venv python3-dev \
        libgl1 libglib2.0-0 libcap2-bin \
        ca-certificates gnupg lsb-release jq sqlite3 tmux htop \
        espeak-ng tesseract-ocr
else
    # ── RHEL/Oracle Linux/Amazon Linux packages ──
    # FIX #10: Oracle Linux uses oracle-epel-release, not epel-release
    if [[ "$OS" == "oracle" ]]; then
        sudo dnf install -y oracle-epel-release-el9 2>/dev/null || \
            sudo dnf install -y oracle-epel-release-el8 2>/dev/null || \
            log_warn "Could not install Oracle EPEL (non-critical — some packages may be missing)"
    elif [[ "$OS" == "amazon" ]]; then
        sudo amazon-linux-extras install epel -y 2>/dev/null || true
    else
        sudo dnf install -y epel-release 2>/dev/null || true
    fi

    pkg_install curl wget git unzip zstd gcc gcc-c++ make \
        ffmpeg python3 python3-pip python3-devel \
        mesa-libGL glib2 libcap \
        ca-certificates gnupg2 redhat-lsb-core jq sqlite tmux htop \
        espeak-ng tesseract

    # FIX #12: Ensure python3 venv module is available on RHEL/Oracle
    if ! python3 -m venv --help &>/dev/null; then
        # Try installing venv package first
        pkg_install python3-virtualenv 2>/dev/null || true
        # If still not available, fall back to virtualenv command
        if ! python3 -m venv --help &>/dev/null && command -v virtualenv &>/dev/null; then
            log_warn "python3 -m venv unavailable — will use virtualenv instead"
            USE_VIRTUAL_ENV=true
        fi
    fi
fi

log_info "System packages installed"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Bun Runtime
# ═══════════════════════════════════════════════════════════════════════════════
log_step 2 "Installing Bun runtime..."

BUN_PATH="$HOME/.bun/bin/bun"
if [[ -x "$BUN_PATH" ]]; then
    log_info "Bun already installed: $($BUN_PATH --version)"
else
    log_info "Downloading Bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
    BUN_PATH="$BUN_INSTALL/bin/bun"
    log_info "Bun installed: $($BUN_PATH --version)"
fi

# Ensure bun is in PATH for the rest of this script
export PATH="$HOME/.bun/bin:$PATH"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Ollama + Local LLMs
# ═══════════════════════════════════════════════════════════════════════════════
log_step 3 "Installing Ollama + local LLM models..."

if command -v ollama &>/dev/null; then
    log_info "Ollama already installed: $(ollama --version 2>/dev/null | head -1)"
else
    log_info "Downloading Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    log_info "Ollama installed"
fi

# Start Ollama service if not running
if ! pgrep -x ollama &>/dev/null; then
    if command -v systemctl &>/dev/null; then
        sudo systemctl start ollama 2>/dev/null || {
            ollama serve >/dev/null 2>&1 &
            sleep 5
        }
    else
        ollama serve >/dev/null 2>&1 &
        sleep 5
    fi
    # Wait for Ollama to be ready
    for i in $(seq 1 10); do
        ollama list &>/dev/null && break
        sleep 2
    done
    log_info "Ollama service started"
else
    log_info "Ollama service already running"
fi

# Helper: check if a specific Ollama model tag is already pulled
# FIX #3: Use exact tag match instead of loose base-name grep
ollama_has_model() {
    local model="$1"
    ollama list 2>/dev/null | awk '{print $1}' | grep -qxF "$model"
}

# Pull vision model (for panel text transcription)
log_info "Pulling vision model: $OLLAMA_VISION_MODEL (this may take a few minutes on first run)..."
if ollama_has_model "$OLLAMA_VISION_MODEL"; then
    log_info "Vision model '$OLLAMA_VISION_MODEL' already pulled"
else
    ollama pull "$OLLAMA_VISION_MODEL" && \
        log_info "Vision model '$OLLAMA_VISION_MODEL' ready" || \
        log_warn "Failed to pull vision model — run later: ollama pull $OLLAMA_VISION_MODEL"
fi

# Pull text model (for narrative rewriting)
log_info "Pulling text model: $OLLAMA_TEXT_MODEL..."
if ollama_has_model "$OLLAMA_TEXT_MODEL"; then
    log_info "Text model '$OLLAMA_TEXT_MODEL' already pulled"
else
    ollama pull "$OLLAMA_TEXT_MODEL" && \
        log_info "Text model '$OLLAMA_TEXT_MODEL' ready" || \
        log_warn "Failed to pull text model — run later: ollama pull $OLLAMA_TEXT_MODEL"
fi

log_info "Ollama models ready:"
ollama list 2>/dev/null | tail -n +2 | while read -r line; do
    log_info "  $line"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Python Virtual Environment + ML Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
log_step 4 "Setting up Python venv + ML dependencies..."

# ── Resolve which system Python to use (openai>=2.49 requires 3.10+) ──
SYSTEM_PYTHON="${SYSTEM_PYTHON:-}"

if [[ -n "$SYSTEM_PYTHON" && -x "$SYSTEM_PYTHON" ]]; then
    # User explicitly set SYSTEM_PYTHON (e.g. miniconda python)
    _py_bin="$SYSTEM_PYTHON"
    log_info "Using custom Python: $_py_bin ($($_py_bin --version 2>&1))"
else
    # Auto-detect: try python3.12, python3.11, python3.10 before falling back to python3
    _py_bin=""
    for _candidate in python3.12 python3.11 python3.10; do
        if command -v "$_candidate" &>/dev/null; then
            _py_bin="$_candidate"
            break
        fi
    done
    _py_bin="${_py_bin:-python3}"
fi

# Verify version is >= 3.10
PY_MAJOR=$($_py_bin -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($_py_bin -c 'import sys; print(sys.version_info.minor)')
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    log_error "Python $($_py_bin --version 2>&1 | awk '{print $2}') is too old — production requires Python 3.10.x (target 3.10.4)"
    log_error ""
    log_error "QUICKEST FIX — install Miniconda (works on any Ubuntu version, ARM or x86):"
    log_error "  wget -qO- https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh | bash"
    log_error "  source ~/miniconda3/bin/activate"
    log_error "  conda create -y -n mrs python=3.10"
    log_error "  conda activate mrs"
    log_error "  Then re-run: SYSTEM_PYTHON=\$CONDA_PREFIX/bin/python ./setup.sh"
    log_error ""
    log_error "ALTERNATIVE — add deadsnakes PPA (may not work on Ubuntu 20.04 focal):"
    log_error "  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get update"
    log_error "  sudo apt-get install python3.10-venv python3.10-dev"
    log_error "  Then re-run: SYSTEM_PYTHON=python3.10 ./setup.sh"
    exit 1
fi
log_info "Python $($_py_bin --version 2>&1 | awk '{print $2}') detected (>= 3.10 OK)"

if [[ -x "$PYTHON_VENV/bin/python3" ]]; then
    log_info "Python venv already exists at $PYTHON_VENV"
else
    # FIX #12: Use virtualenv as fallback if python3 -m venv is unavailable
    rm -rf "$PYTHON_VENV"  # remove incomplete venv if creation failed previously
    if [[ "${USE_VIRTUAL_ENV:-false}" == "true" ]] && command -v virtualenv &>/dev/null; then
        if virtualenv -p "$_py_bin" "$PYTHON_VENV" 2>&1; then
            log_info "Created Python venv (via virtualenv) at $PYTHON_VENV"
        else
            log_error "virtualenv failed — cannot create Python venv"
            exit 1
        fi
    else
        if "$_py_bin" -m venv "$PYTHON_VENV" 2>&1; then
            log_info "Created Python venv at $PYTHON_VENV"
        else
            log_error "$_py_bin -m venv failed — cannot create Python venv"
            log_error "Try: SYSTEM_PYTHON=\$CONDA_PREFIX/bin/python ./setup.sh"
            exit 1
        fi
    fi
fi

# Activate venv for this shell session
source "$PYTHON_VENV/bin/activate"

log_info "Installing Python dependencies from pipeline/requirements.txt..."
pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1

# FIX #6: Use CPU-only torch on all platforms to avoid massive GPU builds.
# This is critical on ARM Oracle Cloud where CUDA builds don't exist anyway.
# We install torch CPU first, then the rest of requirements.
log_info "Installing PyTorch (CPU-only build for compatibility)..."
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision 2>&1 | tail -3

# Install remaining ML deps (torch/torchvision already satisfied, will be skipped)
log_info "Installing remaining ML dependencies..."
pip install --no-deps torch torchvision 2>/dev/null
pip install -r pipeline/requirements.txt 2>&1 | tail -5

# FIX #14: Verify critical imports
log_info "Verifying Python environment..."
if python3 -c "
import PIL; print(f'  Pillow: {PIL.__version__}')
import cv2; print(f'  opencv: {cv2.__version__}')
import numpy; print(f'  numpy: {numpy.__version__}')
import torch; print(f'  torch: {torch.__version__}')
import ultralytics; print(f'  ultralytics: {ultralytics.__version__}')
print('  All Python deps OK')
"; then
    log_info "Python environment verified successfully"
else
    log_error "Python import verification FAILED — check the output above for missing packages"
    log_error "You may need to run: source $PYTHON_VENV/bin/activate && pip install -r pipeline/requirements.txt"
    exit 1
fi

log_info "Python venv: $PYTHON_BIN"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4b: Piper TTS (primary production TTS engine)
#
# Uses Piper's prebuilt standalone binary release, NOT `pip install
# piper-tts`. piper-tts's native dependency (piper-phonemize) only ships
# wheels for Python 3.9+ — there is no cp38 wheel at all — so on a box
# whose Python venv is older (e.g. Ubuntu 20.04's stock Python 3.8), pip's
# resolver falls back to ancient piper-tts 1.1.0/1.2.0 releases and then
# fails with a version conflict between them rather than a solvable
# dependency issue. The prebuilt binary bundles its own espeak-ng,
# libpiper_phonemize, and libonnxruntime, so it works regardless of which
# Python the box has.
# ═══════════════════════════════════════════════════════════════════════════════
log_step "4b" "Installing Piper TTS + voice model..."

PIPER_DIR="$PROJECT_DIR/pipeline/piper"
PIPER_BIN="$PIPER_DIR/piper/piper"
PIPER_VOICE_DIR="$PROJECT_DIR/pipeline/voices"
PIPER_VOICE_NAME="en_US-ryan-high"
PIPER_VOICE_MODEL_PATH="$PIPER_VOICE_DIR/${PIPER_VOICE_NAME}.onnx"

PIPER_ARCH="$(uname -m)"
case "$PIPER_ARCH" in
    x86_64)  PIPER_ASSET="piper_linux_x86_64.tar.gz" ;;
    aarch64) PIPER_ASSET="piper_linux_aarch64.tar.gz" ;;
    armv7l)  PIPER_ASSET="piper_linux_armv7l.tar.gz" ;;
    *)       PIPER_ASSET="" ;;
esac

if [[ -x "$PIPER_BIN" ]]; then
    log_info "piper binary already installed at $PIPER_BIN"
elif [[ -z "$PIPER_ASSET" ]]; then
    log_warn "No prebuilt Piper release for architecture '$PIPER_ARCH' — production TTS will fall back to eSpeak-NG"
else
    mkdir -p "$PIPER_DIR"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/${PIPER_ASSET}"
    if curl -fsSL -o "$PIPER_DIR/piper.tar.gz" "$PIPER_URL" \
        && tar -xzf "$PIPER_DIR/piper.tar.gz" -C "$PIPER_DIR" \
        && rm -f "$PIPER_DIR/piper.tar.gz"; then
        log_info "Piper binary installed"
    else
        log_warn "Piper binary download/extract failed — production TTS will fall back to eSpeak-NG"
    fi
fi

if [[ -f "$PIPER_VOICE_MODEL_PATH" ]]; then
    log_info "Piper voice model already present: $PIPER_VOICE_MODEL_PATH"
else
    mkdir -p "$PIPER_VOICE_DIR"
    PIPER_VOICE_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
    if curl -fsSL -o "$PIPER_VOICE_MODEL_PATH" "$PIPER_VOICE_BASE_URL/${PIPER_VOICE_NAME}.onnx" \
        && curl -fsSL -o "$PIPER_VOICE_MODEL_PATH.json" "$PIPER_VOICE_BASE_URL/${PIPER_VOICE_NAME}.onnx.json"; then
        log_info "Piper voice model downloaded"
    else
        log_warn "Piper voice model download failed — production TTS will fall back to eSpeak-NG"
        rm -f "$PIPER_VOICE_MODEL_PATH" "$PIPER_VOICE_MODEL_PATH.json"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Node.js / Bun Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
log_step 5 "Installing Node.js/Bun dependencies..."

log_info "Installing main project dependencies..."
if bun install 2>&1 | tail -3; then
    log_info "Main project dependencies installed"
else
    log_error "bun install FAILED for main project — check network/disk space"
    exit 1
fi

log_info "Installing pipeline-service dependencies..."
if (cd mini-services/pipeline-service || exit 1
    bun install 2>&1 | tail -3
    BUN_EXIT=${PIPESTATUS[0]}
    exit $BUN_EXIT); then
    log_info "Pipeline-service dependencies installed"
else
    log_error "bun install FAILED for pipeline-service — check network/disk space"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Database Setup (Prisma + SQLite) + .env
# ═══════════════════════════════════════════════════════════════════════════════
log_step 6 "Setting up database (Prisma + SQLite)..."

mkdir -p db

# Create .env if not exists
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        # FIX #2: After copying .env.example, replace placeholders with actual paths
        cp .env.example .env
        sed -i.bak "s|/path/to/your/project|$PROJECT_DIR|g" .env
        sed -i.bak "s|PYTHON_BIN=.*|PYTHON_BIN=$PYTHON_BIN|" .env
        # FIX #2b: Also replace the DATABASE_URL with an absolute path
        # (.env.example uses relative "file:../db/custom.db" which sed above won't match)
        sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=file:$PROJECT_DIR/db/custom.db|" .env
        rm -f .env.bak
        log_info "Created .env from .env.example"
    else
        # Fallback: create minimal .env
        # FIX #2: Use absolute path for DATABASE_URL to avoid any ambiguity
        cat > .env << ENVEOF
# Manhwa Recap Studio v3 — Environment Configuration
DATABASE_URL=file:$PROJECT_DIR/db/custom.db

# Python (venv)
PYTHON_BIN=$PYTHON_BIN
PROJECT_ROOT=$PROJECT_DIR
DATA_DIR=$PROJECT_DIR/data

# Ollama (local LLMs — free, no API key needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=$OLLAMA_VISION_MODEL
OLLAMA_TEXT_MODEL=$OLLAMA_TEXT_MODEL

# Pipeline service port
PORT=$PORT_PIPELINE

# Optional: External VLM providers for faster transcription
# GROQ_API_KEY=
# GEMINI_API_KEY=
# OPENROUTER_API_KEY=
# OPENAI_API_KEY=
ENVEOF
        log_info "Created .env with defaults"
    fi
else
    log_info ".env already exists — not overwriting"
fi

# Ensure critical paths are in .env (even if user already had a .env)
for VAR_NAME in PYTHON_BIN PROJECT_ROOT DATA_DIR OLLAMA_BASE_URL OLLAMA_VISION_MODEL OLLAMA_TEXT_MODEL PORT PIPER_VOICE_MODEL PATH; do
    if ! grep -q "^${VAR_NAME}=" .env 2>/dev/null; then
        case $VAR_NAME in
            PYTHON_BIN)         VAL="$PYTHON_BIN" ;;
            PROJECT_ROOT)       VAL="$PROJECT_DIR" ;;
            DATA_DIR)           VAL="$PROJECT_DIR/data" ;;
            OLLAMA_BASE_URL)    VAL="http://localhost:11434" ;;
            OLLAMA_VISION_MODEL) VAL="$OLLAMA_VISION_MODEL" ;;
            OLLAMA_TEXT_MODEL)   VAL="$OLLAMA_TEXT_MODEL" ;;
            PORT)               VAL="$PORT_PIPELINE" ;;
            PIPER_VOICE_MODEL)   VAL="$PIPER_VOICE_MODEL_PATH" ;;
            # Prepend piper's directory so shutil.which("piper") finds it
            # when master_pipeline.py is spawned from the systemd service.
            # systemd's EnvironmentFile does NOT shell-expand $PATH — a
            # value like "/piper/dir:$PATH" would be taken completely
            # literally (including the two characters "$P..."), silently
            # breaking PATH resolution for python3/bun/ffmpeg/everything
            # else system-wide. Use systemd's actual default PATH as the
            # base instead of trying to reference "the current PATH".
            PATH)               VAL="$(dirname "$PIPER_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" ;;
        esac
        echo "${VAR_NAME}=${VAL}" >> .env
        log_info "Added ${VAR_NAME} to .env"
    fi
done

# Push Prisma schema for main project
log_info "Running Prisma db push (main project)..."
bunx prisma db push --accept-data-loss 2>&1 | tail -5
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log_error "Prisma db push FAILED for main project"
    exit 1
fi

# FIX #4: Also generate Prisma client for pipeline-service
log_info "Running Prisma generate for pipeline-service..."
(
    cd mini-services/pipeline-service || exit 1
    # Pass DATABASE_URL from project root .env so env() in schema resolves
    export DATABASE_URL="$(grep '^DATABASE_URL=' "$PROJECT_DIR/.env" 2>/dev/null | cut -d= -f2-)"
    bunx prisma generate 2>&1 | tail -3
    # Capture prisma's exit code before pipe (PIPESTATUS is subshell-local)
    PRISMA_EXIT=${PIPESTATUS[0]}
    exit $PRISMA_EXIT
)
if [[ $? -ne 0 ]]; then
    log_warn "Prisma generate failed for pipeline-service (non-critical — will retry on first run)"
fi

log_info "Database ready"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Caddy Reverse Proxy
# ═══════════════════════════════════════════════════════════════════════════════
log_step 7 "Setting up Caddy reverse proxy..."

if command -v caddy &>/dev/null; then
    log_info "Caddy already installed: $(caddy version 2>/dev/null | head -1)"
else
    log_info "Installing Caddy..."
    if $is_deb; then
        sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https 2>/dev/null || true
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
            sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg 2>/dev/null || true
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
            sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
        sudo apt-get update -qq 2>/dev/null
        sudo apt-get install -y caddy 2>/dev/null || log_warn "Caddy apt install failed (non-critical)"
    else
        # FIX #11: Use official Caddy RPM on RHEL/Oracle/Amazon Linux
        # The COPR method is unreliable. Use direct RPM from Caddy GitHub releases.
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64) CADDY_ARCH="amd64" ;;
            aarch64) CADDY_ARCH="arm64" ;;
            *) CADDY_ARCH="amd64" ;;
        esac

        # Try the Caddy official repo method first
        sudo dnf install -y 'dnf-command(copr)' 2>/dev/null || true
        sudo dnf copr enable @caddy/caddy -y 2>/dev/null && \
            sudo dnf install -y caddy 2>/dev/null || {
            # FIX #11 fallback: Direct binary install if RPM method fails
            log_warn "Caddy COPR/RPM install failed — falling back to direct binary"
            curl -fsSL "https://github.com/caddyserver/caddy/releases/latest/download/caddy_linux_${CADDY_ARCH}.tar.gz" | \
                sudo tar -xz -C /usr/local/bin caddy 2>/dev/null && \
                sudo chmod +x /usr/local/bin/caddy && \
                log_info "Caddy installed via direct binary" || \
                log_warn "Caddy binary install also failed (non-critical — web traffic will use port 3000)"
        }
    fi
fi

if command -v caddy &>/dev/null; then
    log_info "Caddy installed: $(caddy version 2>/dev/null | head -1)"
    CADDY_BIN=$(which caddy)
else
    CADDY_BIN=""
    log_warn "Caddy not installed — web traffic will use port $PORT_WEB directly"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Production Caddyfile + Systemd Services (auto-start on boot)
# ═══════════════════════════════════════════════════════════════════════════════
log_step 8 "Installing systemd services..."

# FIX #13: Create Caddyfile.prod BEFORE systemd service files (so it exists
# when referenced), and also create it even without systemd

# ── Create production Caddyfile (plain :80, no env var syntax) ──
# FIX #5: Added WebSocket support headers for socket.io
# FIX #15: Removed port 3000 from external access since Next.js binds to 0.0.0.0
# but Caddy handles external traffic on port 80
if [[ -n "$CADDY_BIN" && -x "$CADDY_BIN" ]]; then
    CADDY_ENABLED=true
    cat > Caddyfile.prod << 'CPFEEOF'
# Manhwa Recap Studio v3 — Production Caddy Config
# Change :80 to your domain (e.g. recap.example.com) for auto-HTTPS

:80 {
        @pipeline {
                path /socket.io/*
        }
        handle @pipeline {
                reverse_proxy localhost:3001
        }

        @internal_api {
                path /internal/*
        }
        handle @internal_api {
                reverse_proxy localhost:3001
        }

        handle {
                reverse_proxy localhost:3000
        }
}
CPFEEOF
    log_info "Caddyfile.prod created"
fi

if command -v systemctl &>/dev/null; then
    # Resolve absolute paths at write-time (heredocs expand variables immediately)
    ABS_BUN="$BUN_PATH"

    # ── Next.js App Service ──
    # FIX #5: Use -H 0.0.0.0 so Next.js listens on all interfaces
    # FIX #5b: Remove `tee dev.log` from systemd — it prevents crash detection
    # because systemd tracks the `tee` process PID, not the `next` PID.
    # Logs go to journal instead: journalctl -u manhwa-web -f
    sudo tee /etc/systemd/system/manhwa-web.service >/dev/null << EOF
[Unit]
Description=Manhwa Recap Studio - Next.js Web App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$ABS_BUN --bun next dev -p $PORT_WEB -H 0.0.0.0
Restart=always
RestartSec=5
EnvironmentFile=$PROJECT_DIR/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # ── Pipeline Service ──
    sudo tee /etc/systemd/system/manhwa-pipeline.service >/dev/null << EOF
[Unit]
Description=Manhwa Recap Studio - Pipeline Service (Socket.IO)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/mini-services/pipeline-service
ExecStart=$ABS_BUN --hot index.ts
Restart=always
RestartSec=5
Environment=PORT=$PORT_PIPELINE
EnvironmentFile=$PROJECT_DIR/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # ── Caddy Service (only if Caddy was installed) ──
    if [[ -n "$CADDY_BIN" && -x "$CADDY_BIN" ]]; then
        sudo tee /etc/systemd/system/manhwa-caddy.service >/dev/null << EOF
[Unit]
Description=Manhwa Recap Studio - Caddy Reverse Proxy
After=network.target manhwa-web.service manhwa-pipeline.service

[Service]
Type=simple
User=root
ExecStart=$CADDY_BIN run --config $PROJECT_DIR/Caddyfile.prod
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    fi

    # Reload systemd and enable services
    sudo systemctl daemon-reload
    sudo systemctl enable manhwa-web manhwa-pipeline 2>/dev/null
    if [[ -n "$CADDY_BIN" && -x "$CADDY_BIN" ]]; then
        sudo systemctl enable manhwa-caddy 2>/dev/null || log_warn "Could not enable Caddy service"
    fi
    log_info "Systemd services installed and enabled"
else
    log_warn "systemd not available — services will run in tmux sessions"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: Oracle Cloud Firewall (defense-in-depth)
# ═══════════════════════════════════════════════════════════════════════════════
log_step 9 "Configuring firewall..."

if command -v iptables &>/dev/null; then
    # FIX #15: Only open ports that are actually needed externally
    # Port 80 (Caddy) or 3000 (Next.js direct, only if no Caddy)
    EXTERNAL_PORT=80
    if [[ "$CADDY_ENABLED" != "true" ]]; then
        EXTERNAL_PORT="$PORT_WEB"
    fi

    for port in 22 "$EXTERNAL_PORT"; do
        sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null || \
            sudo iptables -A INPUT -p tcp --dport "$port" -j ACCEPT
    done

    # FIX #9: Persist iptables rules across reboot
    if command -v iptables-save &>/dev/null; then
        if [[ -d /etc/iptables ]]; then
            sudo sh -c 'iptables-save > /etc/iptables/rules.v4' 2>/dev/null && \
                log_info "Firewall rules persisted to /etc/iptables/rules.v4" || \
                log_warn "Could not persist iptables rules"
        elif command -v netfilter-persistent &>/dev/null; then
            sudo netfilter-persistent save 2>/dev/null && \
                log_info "Firewall rules persisted via netfilter-persistent" || \
                log_warn "Could not persist iptables rules"
        else
            # Manual persistence for RHEL/Oracle: add to rc.local or crontab
            log_warn "No iptables persistence tool found — rules will be lost on reboot"
            log_warn "Install iptables-services: sudo dnf install iptables-services"
        fi
    fi

    log_info "Firewall rules configured (ports 22, $EXTERNAL_PORT)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10: Start Everything
# ═══════════════════════════════════════════════════════════════════════════════
log_step 10 "Starting all services..."

mkdir -p logs data

if command -v systemctl &>/dev/null; then
    sudo systemctl stop manhwa-web manhwa-pipeline manhwa-caddy 2>/dev/null || true
    sudo systemctl start manhwa-web
    sleep 3
    sudo systemctl start manhwa-pipeline
    sleep 2
    if [[ -n "$CADDY_BIN" && -x "$CADDY_BIN" ]]; then
        sudo systemctl start manhwa-caddy 2>/dev/null || true
    fi
    log_info "Services started via systemd"
else
    # Fallback: use tmux sessions
    tmux kill-session -t manhwa-web 2>/dev/null || true
    tmux kill-session -t manhwa-pipeline 2>/dev/null || true
    # FIX #5: Use -H 0.0.0.0 for direct access without Caddy
    tmux new-session -d -s manhwa-web   "cd $PROJECT_DIR && $BUN_PATH --bun next dev -p $PORT_WEB -H 0.0.0.0 2>&1 | tee logs/web.log"
    sleep 3
    tmux new-session -d -s manhwa-pipeline "cd $PROJECT_DIR/mini-services/pipeline-service && PORT=$PORT_PIPELINE $BUN_PATH --hot index.ts 2>&1 | tee logs/pipeline.log"
    log_info "Services started in tmux sessions (attach: tmux attach -t manhwa-web)"
fi

# Wait for services to come up
echo -n "  Waiting for web server"
for i in $(seq 1 45); do
    if curl -sf "http://localhost:$PORT_WEB/" >/dev/null 2>&1; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 2
done

# Verify pipeline
if curl -sf "http://localhost:$PORT_PIPELINE/internal/health" >/dev/null 2>&1; then
    log_info "Pipeline service healthy"
else
    log_warn "Pipeline service not responding yet (may need a moment)"
fi

# Verify Caddy
if [[ "$CADDY_ENABLED" == "true" ]]; then
    if curl -sf "http://localhost:$PORT_CADDY/" >/dev/null 2>&1; then
        log_info "Caddy reverse proxy healthy"
    else
        log_warn "Caddy not responding — check: sudo journalctl -u manhwa-caddy -f"
    fi
fi

# ── Final Summary ──────────────────────────────────────────────────────────────
# Determine the externally-facing port for the OCI reminder
OCI_PORT=80
if [[ "$CADDY_ENABLED" != "true" ]]; then
    OCI_PORT="$PORT_WEB"
fi

echo ""
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}                    SETUP COMPLETE                               ${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Services:${NC}"
if [[ "$CADDY_ENABLED" == "true" ]]; then
    echo -e "    Web App (via Caddy):  ${GREEN}http://<your-ip>:80${NC}"
else
    echo -e "    Web App (direct):     ${GREEN}http://<your-ip>:$PORT_WEB${NC}"
fi
echo -e "    Pipeline (internal):  localhost:$PORT_PIPELINE"
echo -e "    Ollama API:           localhost:11434"
echo ""
echo -e "  ${BOLD}Local LLMs (Ollama):${NC}"
echo -e "    Vision (transcription): ${GREEN}$OLLAMA_VISION_MODEL${NC}"
echo -e "    Text (narration):       ${GREEN}$OLLAMA_TEXT_MODEL${NC}"
echo ""
echo -e "  ${BOLD}Python:${NC}"
echo -e "    Venv:                 ${CYAN}${PYTHON_VENV}${NC}"
echo -e "    Binary:               ${CYAN}${PYTHON_BIN}${NC}"
echo ""
echo -e "  ${BOLD}Useful commands:${NC}"
if command -v systemctl &>/dev/null; then
    echo -e "    View web logs:        ${CYAN}journalctl -u manhwa-web -f${NC}"
    echo -e "    View pipeline logs:   ${CYAN}journalctl -u manhwa-pipeline -f${NC}"
    if [[ "$CADDY_ENABLED" == "true" ]]; then
        echo -e "    View Caddy logs:       ${CYAN}journalctl -u manhwa-caddy -f${NC}"
        echo -e "    Restart Caddy:        ${CYAN}sudo systemctl restart manhwa-caddy${NC}"
        echo -e "    Stop all:             ${CYAN}sudo systemctl stop manhwa-web manhwa-pipeline manhwa-caddy${NC}"
    else
        echo -e "    Stop all:             ${CYAN}sudo systemctl stop manhwa-web manhwa-pipeline${NC}"
    fi
    echo -e "    Restart web:          ${CYAN}sudo systemctl restart manhwa-web${NC}"
    echo -e "    Restart pipeline:      ${CYAN}sudo systemctl restart manhwa-pipeline${NC}"
else
    echo -e "    View web logs:        ${CYAN}tmux attach -t manhwa-web${NC}"
    echo -e "    View pipeline logs:   ${CYAN}tmux attach -t manhwa-pipeline${NC}"
    echo -e "    Stop all:             ${CYAN}tmux kill-session -t manhwa-web; tmux kill-session -t manhwa-pipeline${NC}"
fi
echo -e "    Ollama status:         ${CYAN}ollama list${NC}"
echo -e "    Pull another model:   ${CYAN}ollama pull llama3.1:8b${NC}"
echo ""
echo -e "  ${BOLD}Optional (add to .env for faster transcription):${NC}"
echo -e "    GROQ_API_KEY=           ${YELLOW}(free — console.groq.com/keys)${NC}"
echo -e "    GEMINI_API_KEY=         ${YELLOW}(free — aistudio.google.com/apikey)${NC}"
echo -e "    OPENROUTER_API_KEY=     ${YELLOW}(free tier — openrouter.ai/keys)${NC}"
echo -e "    OPENAI_API_KEY=         ${YELLOW}(paid — platform.openai.com/api-keys)${NC}"
echo ""
echo -e "  ${BOLD}${RED}Oracle Cloud — IMPORTANT:${NC}"
echo "    Open port $OCI_PORT in OCI Console:"
echo "    Networking → Security Lists → Add Ingress Rule"
echo "    Source: 0.0.0.0/0  Port: $OCI_PORT  Protocol: TCP"
echo ""
log_info "Setup complete!"