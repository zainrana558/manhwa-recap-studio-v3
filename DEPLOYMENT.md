# Deploying Manhwa Recap Studio (Free Tier)

This guide deploys the app for **$0** using a split architecture:

- **Frontend + API** → Vercel (free, 24/7 online)
- **Database** → Turso (free 9 GB hosted SQLite)
- **Video storage** → Cloudflare R2 (free 10 GB, streams with seek)
- **Compute (Python pipeline)** → Your laptop (free, exposed via Cloudflare Tunnel)

The website is always online (Vercel + Turso + R2). Your laptop only needs to
be on when rendering new videos.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  VERCEL (free)  —  frontend + API, 24/7 online       │
│  - Next.js app                                       │
│  - /api/* routes (talk to Turso + pipeline service)  │
│  - Streams finished videos from R2                   │
└──────────────┬──────────────────────────────────────┘
               │ job start/cancel + socket.io
┌──────────────▼──────────────────────────────────────┐
│  YOUR LAPTOP (free)  —  compute, on-demand           │
│  - pipeline-service (port 3001, Python + ffmpeg)     │
│  - Exposed via Cloudflare Tunnel → public HTTPS URL  │
│  - Uploads finished videos to R2, frees local disk   │
└──────────────┬──────────────────────────────────────┘
               │ uploads video
┌──────────────▼──────────────────────────────────────┐
│  CLOUDFLARE R2 (free 10 GB)  —  streaming storage    │
│  - Browser streams videos with HTTP Range (seeking)  │
│  - Vercel /api/download redirects to presigned URL   │
└──────────────────────────────────────────────────────┘
               │ reads/writes
┌──────────────────────────────────────────────────────┐
│  TURSO (free 9 GB)  —  shared database               │
│  - Both Vercel API and your laptop's pipeline-service│
│    connect to the SAME Turso DB                      │
└──────────────────────────────────────────────────────┘
```

---

## Step 1 — Set up Turso (database)

Turso is free hosted SQLite — your existing Prisma schema works unchanged.

1. Sign up at **https://turso.tech** (free, no credit card).
2. Create a database:
   ```bash
   # Install the Turso CLI
   curl -sSfL https://get.tur.so/install.sh | bash

   # Log in + create a DB
   turso auth login
   turso db create manhwa-recap

   # Get your connection URL + auth token
   turso db show manhwa-recap --url
   turso db tokens create manhwa-recap
   ```
3. Save these two values — you'll need them for both Vercel and your laptop:
   - `DATABASE_URL` = `libsql://manhwa-recap-<you>.turso.io`
   - `DATABASE_AUTH_TOKEN` = `eyJhbGciOiJF...`

4. Push your schema to Turso (from your laptop, in the project root):
   ```bash
   # Temporarily point at Turso to create the tables
   export DATABASE_URL="libsql://manhwa-recap-<you>.turso.io"
   export DATABASE_AUTH_TOKEN="eyJhbGciOiJF..."
   bun run db:push
   ```

---

## Step 2 — Set up Cloudflare R2 (video storage)

1. Sign up at **https://cloudflare.com** → R2 (free 10 GB, no credit card).
2. Create a bucket, e.g. `manhwa-recaps`.
3. Generate API credentials: R2 → Manage R2 API Tokens → Create.
   - You need: Account ID, Access Key ID, Secret Access Key.
4. Save these four values:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET` = `manhwa-recaps`
   - `R2_PUBLIC_URL` = leave empty (use presigned URLs for security)

Your code already handles the rest: when a job finishes, the pipeline-service
uploads the MP4 to R2 and deletes the local file. The `/api/download` route
detects `r2Key` and redirects to a presigned R2 URL.

---

## Step 3 — Deploy the frontend to Vercel

1. Push your project to GitHub (private repo is fine):
   ```bash
   git init
   git add .
   git commit -m "Prepare for Vercel deployment"
   git remote add origin https://github.com/yourname/manhwa-recap-studio.git
   git push -u origin main
   ```

2. Go to **https://vercel.com** → New Project → import your GitHub repo.

3. Vercel auto-detects Next.js. Before deploying, add these **Environment
   Variables** in the Vercel dashboard:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | `libsql://manhwa-recap-<you>.turso.io` |
   | `DATABASE_AUTH_TOKEN` | `eyJhbGciOiJF...` |
   | `PIPELINE_SERVICE_URL` | `https://your-laptop.trycloudflare.com` (set in Step 5) |
   | `NEXT_PUBLIC_PIPELINE_SERVICE_URL` | `https://your-laptop.trycloudflare.com` (same) |
   | `R2_ACCOUNT_ID` | your Cloudflare account ID |
   | `R2_ACCESS_KEY_ID` | your R2 access key |
   | `R2_SECRET_ACCESS_KEY` | your R2 secret key |
   | `R2_BUCKET` | `manhwa-recaps` |

4. Deploy. You'll get a URL like `https://manhwa-recap-studio.vercel.app`.

> **Note:** Vercel cannot run the Python pipeline or the socket.io mini-service
> (serverless = no long-running processes, no Python, no ffmpeg). That's why
> compute stays on your laptop (Step 5).

---

## Step 4 — Set up your laptop as the compute host

Your laptop already has everything installed (we set this up in the sandbox).
On your own laptop:

1. Clone the repo + install deps:
   ```bash
   git clone https://github.com/yourname/manhwa-recap-studio.git
   cd manhwa-recap-studio
   bun install
   cd mini-services/pipeline-service && bun install && cd ../..
   ```

2. Install Python deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install edge-tts openai Pillow opencv-python numpy torch torchvision ultralytics huggingface-hub
   ```

3. Create a `.env` file on your laptop with the **same** Turso + R2 credentials
   (so the laptop shares the Vercel DB):
   ```env
   DATABASE_URL=libsql://manhwa-recap-<you>.turso.io
   DATABASE_AUTH_TOKEN=eyJhbGciOiJF...
   R2_ACCOUNT_ID=...
   R2_ACCESS_KEY_ID=...
   R2_SECRET_ACCESS_KEY=...
   R2_BUCKET=manhwa-recaps
   DATA_DIR=/Volumes/MyHDD/manhwa-data    # point at an external HDD
   PROJECT_ROOT=/path/to/manhwa-recap-studio
   PYTHON_BIN=/path/to/manhwa-recap-studio/.venv/bin/python
   ```

4. Start the pipeline-service:
   ```bash
   cd mini-services/pipeline-service
   bun run dev
   ```

---

## Step 5 — Expose your laptop to the internet (Cloudflare Tunnel)

This lets Vercel (and your friends' browsers) reach your laptop's
pipeline-service over HTTPS — no port forwarding or dynamic DNS needed.

1. Install `cloudflared`:
   - Mac: `brew install cloudflared`
   - Windows: download from https://github.com/cloudflare/cloudflared/releases
   - Linux: `sudo apt install cloudflared`

2. Start a tunnel to port 3001:
   ```bash
   cloudflared tunnel --url http://localhost:3001
   ```

3. It prints a URL like `https://random-words-123.trycloudflare.com`.
   Copy it and set it as `PIPELINE_SERVICE_URL` and
   `NEXT_PUBLIC_PIPELINE_SERVICE_URL` in **both**:
   - Your laptop's `.env`
   - Vercel's environment variables (then redeploy)

> The tunnel URL changes each time you restart `cloudflared` (free tier). For a
> stable URL, create a named tunnel with your own domain (also free if you own
> a domain on Cloudflare).

---

## Step 6 — Verify the full flow

1. Open your Vercel URL (`https://manhwa-recap-studio.vercel.app`).
2. Search for a manga, configure, start a job.
3. The job request goes: **Vercel → Turso (create Job row) → your laptop
   (tunnel) → pipeline-service starts rendering**.
4. Live progress streams via socket.io: **browser → your laptop (tunnel)**.
5. When done, the video uploads to **R2**; the browser streams it from R2.

Your laptop's disk stays clean (videos go to R2). The website works even when
your laptop is off (for viewing existing videos; new jobs just queue until the
laptop is back online).

---

## One-click laptop launcher

Create `start-laptop.sh` so you don't run multiple terminals each time:

```bash
#!/bin/bash
# start-laptop.sh — launches the pipeline-service + cloudflare tunnel
cd "$(dirname "$0")"
source .venv/bin/activate
cd mini-services/pipeline-service
bun run dev &
cd ../..
cloudflared tunnel --url http://localhost:3001
```

Make it executable: `chmod +x start-laptop.sh`. Run `./start-laptop.sh` to
start everything; Ctrl+C to stop.

---

## Cost summary

| Service | Free tier | What it covers |
|---|---|---|
| Vercel | Hobby (free) | Frontend + API hosting, 100 GB bandwidth/mo |
| Turso | Free tier | 9 GB database, 1B reads/mo |
| Cloudflare R2 | Free tier | 10 GB storage, **free egress** |
| Cloudflare Tunnel | Free | Public HTTPS URL for your laptop |
| Your laptop | $0 | Compute (only on when rendering) |
| edge-tts | Free | Unlimited TTS |
| **Total** | **$0/mo** | **~14 streaming videos online, unlimited archived** |

For more than ~14 streaming videos, archive old ones to Mega (20 GB
free) or Terabox (1 TB free) — see the storage tiering notes in the chat.

---

## Troubleshooting

**"Voice preview doesn't work on Vercel"**
→ Your laptop's pipeline-service must be running (it generates previews). Check
that `PIPELINE_SERVICE_URL` is set correctly and the tunnel is up.

**"Job stays at 'pending' forever"**
→ Vercel can't reach your laptop. Verify the tunnel URL is correct and your
laptop is online. The pipeline-service also auto-requeues pending jobs on
startup, so restarting it usually fixes stuck jobs.

**"Video won't play / download 404s"**
→ R2 isn't configured, or the job finished before R2 creds were added. Check
that `R2_*` env vars are set on your laptop (where uploads happen) and that
the job's `r2Key` field is populated in Turso.

**"Socket.io won't connect"**
→ `NEXT_PUBLIC_PIPELINE_SERVICE_URL` must be set on Vercel (client-side env var)
and point to your laptop's tunnel URL. The tunnel must allow WebSocket
upgrades (Cloudflare Tunnels do this by default).

## Oracle Cloud production deployment (CPU-only)

Validated production deployments should use an isolated Python 3.10.4 install such as `/opt/python3.10`; never replace Ubuntu's system Python. Create the runtime environment with:

```bash
sudo mkdir -p /opt/manhwa-recap-studio
/opt/python3.10/bin/python3.10 -m venv /opt/manhwa-recap-studio/.venv
source /opt/manhwa-recap-studio/.venv/bin/activate
python --version  # must be Python 3.10.4 for the pinned production target
pip install --upgrade pip setuptools wheel
pip install -r pipeline/requirements.txt
pip install -r mini-services/paddleocr-service/requirements.txt \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -c "import paddle; print(paddle.__version__)"      # expected 3.1.1
python -c "import paddleocr; print(paddleocr.__version__)" # expected 3.7.0
```

Install local media/OCR/TTS runtime components:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg espeak-ng
# Install Piper into PATH, then download one CPU-friendly English ONNX voice.
export PIPER_VOICE_MODEL=/opt/piper/voices/en_US-lessac-medium.onnx
piper --version
espeak-ng --version
ffmpeg -version
ffprobe -version
```

Production work uses stable job IDs and a work directory. The SQLite state file is `<work-dir>/state.sqlite`; artifacts are promoted only after QA to `<work-dir>/artifacts`; corrupt or partial files are moved to `<work-dir>/quarantine`.

```bash
export PRODUCTION_PIPELINE=1
export PYTHONPATH=/path/to/manhwa-recap-studio
export PIPER_VOICE_MODEL=/opt/piper/voices/en_US-lessac-medium.onnx
python pipeline/production_canary.py --work-dir /tmp/mrs-canary --job-id canary
for stage in OCR TTS AUDIO_ASSEMBLY VIDEO_RENDER MERGE; do
  python pipeline/crash_resume_harness.py --work-dir /tmp/mrs-crash-$stage --job-id crash-$stage --stage $stage
done
python pipeline/master_pipeline.py --input-dir /data/input --output-path /data/output/final.mp4 \
  --work-dir /data/work --production-mode --job-id real-job-001
```

Recovery procedure: restart the same command with the same `--job-id` and `--work-dir`. RUNNING stages are reconciled against their real artifacts; valid artifacts become COMPLETE, while missing/corrupt artifacts become RETRYABLE and are rebuilt. Use `ffprobe` on the final MP4 and inspect `state.sqlite` before declaring a job complete.

Resource baseline for Oracle CPU-only production is 2+ vCPU, at least 11 GiB RAM, swap enabled when possible, and tens of GiB of free disk. `ResourceGuard` checks disk and available RAM before expensive stages and records RESOURCE/RETRYABLE instead of proceeding when thresholds are not met.
