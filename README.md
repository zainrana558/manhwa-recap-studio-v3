---
title: Manhwa Recap Studio
emoji: 🎬
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Manhwa Recap Studio

Auto-scrape manhwa/manga/webtoon chapters, transcribe panel text with
**PaddleOCR PP-OCRv5** (primary, no API keys needed, VLM fallback),
translate to English, and render a narrated recap video with text-to-speech
audio. All in one Docker container — runs free on Hugging Face Spaces.

## Quick Deploy (5 steps, ~15 min)

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full guide. Short version:

1. **Create a free Hugging Face account** at huggingface.co
2. **Create a new Space** → SDK: Docker → Space name: `manhwa-recap-studio`
3. **Upload this repo** to the Space (or clone + push via git)
4. **Enable persistent storage** (Settings → Persistent storage → 20 GB free)
5. Wait ~10 min for the Docker build → your app is live at
   `https://yourname-manhwa-recap-studio.hf.space`

No environment variables required to start. Add `MEGA_EMAIL` / `MEGA_PASSWORD`
(optional) for cloud archive to Mega (20 GB free).

## What runs inside the container

| Service | Port | Purpose |
|---|---|---|
| Next.js (standalone) | 3000 | Frontend + API routes |
| Pipeline-service | 3001 | Job queue + VLM fallback + socket.io |
| PaddleOCR service | 3002 | PP-OCRv5 OCR engine (primary transcriptor) |
| Caddy | 7860 | Reverse proxy (HF Spaces entry point) |

## Optional env vars (Settings → Repository secrets)

| Variable | Purpose |
|---|---|
| `MEGA_EMAIL` | Mega cloud archive (20 GB free) |
| `MEGA_PASSWORD` | Mega cloud archive |
| `GROQ_API_KEY` | Better narration/translation (free at console.groq.com/keys) |
| `AUTO_ARCHIVE` | `true` (default) to auto-upload videos to Mega |

## Features

- Search 6 sources: MangaHere, FanFox, Webtoons, AsuraScans, MAL, AniList
- 55 narration voices with inline preview playback
- **PaddleOCR PP-OCRv5** primary transcription (+13% accuracy over v4, no API keys)
- VLM providers (SiliconFlow, Gemini, Groq) as automatic OCR fallback
- YOLO panel detection
- edge-tts narration with clean audio (no pops/clicks)
- Mega cloud archive (auto-upload + on-demand restore)
- Full HTTP Range video streaming (seek support)
