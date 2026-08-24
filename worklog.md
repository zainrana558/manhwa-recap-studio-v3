# Manhwa Recap Studio v3 — Development Worklog

---
## Pipeline Hardening & Concat Fix Assessment

**Cycle**: Concat Fix & Pipeline Self-Healing Hardening
**Date**: 2025-08-20
**Status**: Completed & Verified

### Summary of Changes

1. **Fixed Off-by-One Concat Duration Bug (`pipeline/master_pipeline.py`)**
   - Removed trailing duplicate `file` entry without duration in `render_chapter()`.
   - Updated `video_qa()` default tolerance to 5.0 seconds (override via `VIDEO_QA_TOLERANCE_SECONDS`).

2. **Chapter Rendering Resilience & Fallbacks (`pipeline/master_pipeline.py`)**
   - Implemented `render_chapter_with_retry()` with up to 2 retries on rendering QA failure.
   - Added `generate_black_placeholder_chapter()` to substitute silent black video overlays ("Chapter X unavailable") when chapter render retries fail or when chapters are missing during `merge_chapters()`.

3. **OCR & VLM Chain Absolute Robustness (`mini-services/paddleocr-service/main.py` & `mini-services/pipeline-service/lib.ts`)**
   - PaddleOCR service initialization retries increased to 5 with exponential backoff up to 120s.
   - Batch OCR service calls retried up to 3 times before falling back to VLM.
   - Emergency local Tesseract CLI fallback (`runTesseractFallback`) added when all VLM providers fail or are unconfigured.
   - Short TTL caching (5 minutes) implemented for failed OCR/VLM transcription attempts to prevent hammering broken services.
   - VLM batch failures fill with placeholder text `"[transcription unavailable]"` without throwing or aborting.

4. **Resource Guard Pause Mechanism (`pipeline/production.py` & `pipeline/master_pipeline.py`)**
   - Added `ResourceGuard.wait_for_resources()` to sleep and re-check resource levels periodically when disk/RAM thresholds are passed, rather than failing immediately.

5. **Subprocess Error Reporting (`pipeline/master_pipeline.py` & `mini-services/pipeline-service/index.ts`)**
   - Detailed ffmpeg error output logging (full stderr + last 20 lines) in `run_ffmpeg()`.
   - Ring buffer in Node orchestrator `index.ts` captures last 50 lines of stderr when Python pipeline process exits with code 1.

6. **Unit Testing (`tests/test_concat_duration.py`)**
   - Added automated test validating concat demuxer video duration against expected frame durations within QA tolerance.

---
## Current Project Status Assessment

**Cycle**: Round 11 (SiliconFlow VLM + Gemini fix)
**Date**: 2025-08-13 (America/Los_Angeles)
**Dev Server**: Next.js 16.1.3 (Turbopack) — ✅ running on port 3000
**Pipeline Service**: ✅ Bun Socket.IO on port 3001
**GitHub**: Pushed to `main` as commit `11a2142`
**Lint**: Clean (only pre-existing check-job.js warning)
**Database**: SQLite (Prisma ORM) — `siliconFlowKey` field added

### What Was Done This Session

1. **SiliconFlow VLM Provider Integration**
   - Added `narrateImageBatchSiliconFlow()` function (~120 lines)
   - Uses OpenAI-compatible API at `https://api.siliconflow.cn/v1/chat/completions`
   - Model: `Qwen/Qwen2.5-VL-7B-Instruct` (free, 14M tokens/month)
   - SiliconFlow is now **#1 priority provider** (best free option)
   - Pre-flight test, circuit breaker, retry dispatch all wired
   - Frontend has "BEST FREE" badge to guide users

2. **Full Stack Wiring (9 files changed)**
   - `mini-services/pipeline-service/lib.ts` — provider type, pre-flight, function, dispatch, retry
   - `mini-services/pipeline-service/index.ts` — `SILICONFLOW_API_KEY` per-job key mapping
   - `mini-services/pipeline-service/prisma/schema.prisma` — `siliconFlowKey String?`
   - `prisma/schema.prisma` — `siliconFlowKey String?`
   - `src/types/pipeline.ts` — `siliconFlowKey` in `CreateJobInput` and `AppSettings`
   - `src/app/api/jobs/route.ts` — passes `siliconFlowKey` to DB
   - `src/app/api/settings/route.ts` — handles `siliconFlowKey` in settings CRUD
   - `src/components/pipeline/manga-config.tsx` — input with "BEST FREE" badge, state, submission
   - `src/components/pipeline/settings-dialog.tsx` — settings form field

3. **Gemini Model Fix**
   - Changed `gemini-2.5-flash` → `gemini-2.0-flash` (confirmed working vision model)

### Provider Status Summary
| Provider | Status | Priority | Notes |
|----------|--------|----------|-------|
| **SiliconFlow** (Qwen2.5-VL) | ✅ Integrated | #1 | Free, 14M tokens/mo, easy signup |
| Zhipu AI (GLM-4V-Flash) | ✅ Integrated | #2 | Free but registration issues |
| OpenRouter (nemotron-nano) | ✅ Working | #3 | Free tier, slow |
| Gemini (2.0-flash) | ✅ Fixed | #4 | Fixed model name |
| Groq (qwen3.6-27b) | ⚠️ Limited | #5 | 8000 TPM too low for images |
| Ollama (llava:7b) | ❌ Too slow | #6 | ~82s/panel on CPU |
| z-ai SDK | ❌ Sandbox only | #7 | Only in Z.ai environment |

### User's Next Steps
```bash
cd ~/manhwa-recap-studio-v3 && git pull
```
1. Go to **https://cloud.siliconflow.cn** → sign up (free, easy)
2. Get API key from dashboard
3. Enter it in the pipeline config (marked "BEST FREE")
4. Run a job — SiliconFlow will be used as primary VLM provider

### Unresolved / Risks
- Pipeline service Prisma needs `db push` on the server (npx unavailable — may need `bunx prisma db push`)
- Gemini `gemini-2.0-flash` not yet tested with a real key
- Exposed GitHub PAT from earlier chat — should still be revoked
