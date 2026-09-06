# Migrating to a fresh instance (one script)

This is the whole procedure for standing the stack up on a **new box** — in
particular an Oracle Cloud **Ampere A1 "Always Free"** ARM instance
(Ubuntu 24.04, aarch64). The short version:

```bash
git clone <your-repo-url> manhwa-recap-studio && cd manhwa-recap-studio
SKIP_OLLAMA=1 ./setup.sh
# then paste your secrets into .env  (see "Secrets" below) and:
bash start.sh
```

`setup.sh` is idempotent — re-run it any time; it skips whatever is already
done.

---

## What `setup.sh` does for you

| Area | Handled automatically |
|---|---|
| System packages | ffmpeg, python3-venv, libgl, espeak-ng, tesseract, sqlite, jq, tmux (apt/dnf) |
| Swap | creates a 4 GB swapfile if the box has < 2 GB swap |
| Bun | installs to `~/.bun` |
| Python venv | `.venv/` with CPU-only torch, `pipeline/requirements.txt`, OCR deps |
| OCR engine | RapidOCR (PP-OCRv5) — **primary**, works on ARM |
| Piper TTS | downloads the correct binary for the box's arch + the `en_US-ryan-high` voice |
| Node deps | `bun install` for the app **and** `mini-services/pipeline-service` |
| Database | `prisma db push` creates `db/custom.db` from `prisma/schema.prisma` |
| `.env` | generated with correct absolute paths (see `.env.example` for the full list) |
| systemd | `install-systemd.sh` → `manhwa-recap-studio.service` (reboot-survivable) |
| Caddy | reverse proxy on :80 → :3000 / :3001 |
| Firewall | opens 22 + 80 (or 3000) via iptables |

### Models — all fetched at first run, nothing to copy

- **`comic-text-and-bubble-detector`** (RT-DETR caption/bubble splitter) —
  `start.sh` pulls it from Hugging Face on first launch.
- **`manga-panel-yolo`** ONNX — committed in the repo.
- **RapidOCR PP-OCRv5 mobile** ONNX — auto-downloaded by RapidOCR on first
  OCR call (`start.sh` pre-fetches it to warm the cache).
- **OCR-correction MLM** (`ocr_correction_mlm/`, bert-tiny, ~18 MB) —
  committed in the repo. If it is somehow missing, the OCR service just logs
  "disambiguation disabled" and keeps working.
- **Piper voice** (`en_US-ryan-high.onnx`, 120 MB) — downloaded by `setup.sh`
  / `start.sh`, not in git.

---

## ARM (aarch64) specifics — already automatic

`setup.sh` detects `uname -m` and on ARM:

- **Skips PaddleOCR / paddlepaddle.** There is no aarch64 wheel for the
  pinned `paddlepaddle==2.6.2`. It was only ever the *fallback* OCR tier;
  RapidOCR (primary) + Tesseract (last resort) cover OCR on ARM. The OCR
  service starts READY on RapidOCR alone.
- **Auto-skips Ollama** on boxes with < 8 GB RAM (llava:7b needs ~6 GB).
  Pass `SKIP_OLLAMA=1` explicitly to be sure. Use a free `GEMINI_API_KEY`
  in `.env` if you want VLM panel transcription instead of OCR-only.
- Pulls `piper_linux_aarch64.tar.gz`.

Nothing extra to do — just run `SKIP_OLLAMA=1 ./setup.sh`.

If your recap flow is the same as the current one — per-frame **RapidOCR**
transcription + `--narration-provider none` — you need **no API keys at all**.

---

## Secrets — the only manual step

`setup.sh` writes a valid `.env`, but these are blank and only you have them.
Edit `.env` (or export before `start.sh`):

| Key | Needed when | Where to get it |
|---|---|---|
| `PIPELINE_SECRET` | always (auto-generated into `.pipeline-secret` if left blank — fine) | any 32-char random string |
| `GEMINI_API_KEY` | only if you want VLM transcription instead of OCR-only | aistudio.google.com/apikey (free) |
| `GROQ_API_KEY` / `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | alternative VLM providers | respective consoles |
| `DATABASE_AUTH_TOKEN`, `R2_*` | only for the hosted Vercel+Turso+R2 split (see `DEPLOYMENT.md`) | Turso / Cloudflare |

**Never commit `.env`, `.pipeline-secret`, `.gemini_key`, or `*.pem`.** They
are gitignored; keep it that way.

---

## Optional — carry over data from the old box

A fresh install starts with an empty library and no rendered videos. To keep
what the old box had, copy these **before** the old instance is reclaimed
(they are all gitignored / not in the repo by design):

```bash
# from the OLD box, to the NEW one (adjust host/paths):
NEW=ubuntu@<new-ip>
OLD_DIR=/home/ubuntu/manhwa-recap-studio-v3
NEW_DIR=/home/ubuntu/manhwa-recap-studio

# 1. the library / job history DB  (schema is recreated by setup.sh; this
#    overwrites it with your real rows — do it AFTER setup.sh, service stopped)
scp $OLD_DIR/db/custom.db  $NEW:$NEW_DIR/db/custom.db

# 2. finished recap videos + any in-flight job work dirs
rsync -avz $OLD_DIR/data/  $NEW:$NEW_DIR/data/

# 3. your real .env values (or just re-enter the handful of secrets by hand)
scp $OLD_DIR/.env  $NEW:$NEW_DIR/.env.oldbox   # then merge paths by hand —
                                               # the OLD absolute paths won't match
```

The current job's output lands at
`data/jobs/<job-id>/output/master_recap.mp4`. Pull that off the old box
first if the job finished there.

---

## Verify

```bash
curl -s localhost:3002/health | jq        # {"state":"READY", ...}  RapidOCR up
curl -s localhost:3001/internal/health    # ok
curl -sI localhost:3000/ | head -1        # HTTP/1.1 200 OK
sudo systemctl status manhwa-recap-studio # active (running), enabled
```

Then open `http://<your-ip>/` and start a job.

---

## Sizing note

The recap render is CPU-bound. On a 1-OCPU ARM free instance set
`RECAP_TTS_WORKERS=2` in `.env` (the default). On a 2+ core box use `4`.
Rendering ~328 chapters is a multi-hour job either way; it resumes cleanly
if interrupted (per-chapter `chap_NNN.mp4` files are checkpoints — re-running
the same `--job-id` picks up where it left off).
