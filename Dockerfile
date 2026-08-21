# =============================================================================
# Manhwa Recap Studio — Docker image for Hugging Face Spaces
# Runs the entire app (Next.js + Python pipeline + socket.io + Caddy) in one
# container. Free 16 GB RAM, 50 GB storage, 24/7 online.
# =============================================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    caddy \
    curl ca-certificates git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Bun (JavaScript runtime) ----------------------------------------------
RUN curl -fsSL https://bun.sh/install | bash
ENV BUN_INSTALL="/root/.bun"
ENV PATH="${BUN_INSTALL}/bin:${PATH}"

# ---- Python dependencies (Python 3.10.4 target; local-first production TTS) ------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /app

# Copy package files first (better Docker layer caching)
COPY package.json bun.lock ./
COPY mini-services/pipeline-service/package.json mini-services/pipeline-service/bun.lock ./mini-services/pipeline-service/
COPY mini-services/paddleocr-service/requirements.txt ./mini-services/paddleocr-service/
COPY pipeline/requirements.txt ./pipeline/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r pipeline/requirements.txt && \
    pip install --no-cache-dir -r mini-services/paddleocr-service/requirements.txt

RUN bun install
RUN cd mini-services/pipeline-service && bun install

# Copy source code
COPY . .

# Build Next.js for production (standalone output)
RUN bun run build

# ---- Runtime config ---------------------------------------------------------
# Persistent storage on HF Spaces: /data survives container restarts
ENV DATA_DIR=/data/manhwa-data
ENV DATABASE_URL=file:/data/db/custom.db
ENV PROJECT_ROOT=/app
ENV PYTHON_BIN=/opt/venv/bin/python3
ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
ENV PORT=3000

RUN mkdir -p /data/db /data/manhwa-data /data/cache

# HF Spaces expects the app on port 7860 (Caddy listens here, proxies to 3000/3001)
EXPOSE 7860

# Start script launches: pipeline-service (3001) + Next.js (3000) + Caddy (7860)
CMD ["bash", "start-hf.sh"]
