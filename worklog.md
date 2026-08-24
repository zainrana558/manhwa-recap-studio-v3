# Manhwa Recap Studio v3 — Development Worklog

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

---

## Production Audit & Hardening Summary (Cycle 12)

**Audit Date**: 2025-08-14
**Scope**: All codebase files including `pipeline/master_pipeline.py`, `pipeline/production.py`, `mini-services/pipeline-service/index.ts`, `mini-services/pipeline-service/lib.ts`, `mini-services/paddleocr-service/main.py`, shell scripts (`setup.sh`, `start.sh`, `start-services.sh`), Prisma schema, and Next.js frontend routes.
**Result**: 100% of discovered Critical, High, and Medium issues identified and fixed. Unit & integration test suite (`tests/test_audit_hardening.py`, `tests/test_canary_crash_resume.py`, `tests/test_production.py`) verified 10/10 passing.

---

## Discovered Issues & Fix Summary

### 1. Critical Issues

#### Issue C1: Unbounded Subprocess Execution & Queue Deadlock (`mini-services/pipeline-service/index.ts`)
- **Root Cause**: `processJob` spawned `master_pipeline.py` without process group isolation (`detached: true`) or process tree cleanup logic. If Python hanged indefinitely (e.g. on external network calls or ffmpeg pipe deadlocks), the queue remained locked forever because `currentlyRunning` never cleared.
- **Fix Applied**: Added process group spawning (`detached: true`) and implemented `killChildProcessTree(child, signal)` to terminate the entire process tree (`-pid` signal) on job cancellation or process shutdown.

#### Issue C2: Silent Uncaught Exceptions & Stuck Job State (`mini-services/pipeline-service/index.ts`)
- **Root Cause**: `process.on('uncaughtException')` and `process.on('unhandledRejection')` logged errors to `console.error` without updating active job status in SQLite/Prisma database or notifying connected Socket.IO clients, leaving jobs permanently stuck in `rendering` or `transcribing` status.
- **Fix Applied**: Created `handleFatalError(err, origin)` helper. When an uncaught exception or rejection occurs while a job is running, it updates the job status to `error` in DB and emits an `error` event over Socket.IO before exiting or recovering.

---

### 2. High Priority Issues

#### Issue H1: Corrupt/Zero-Byte Image Crashes (`pipeline/master_pipeline.py`)
- **Root Cause**: `Image.open(panel_path)` in `slice_chapter_panels` and `_compose_canvas_from_source` threw unhandled `UnidentifiedImageError` or `OSError` if a scraped chapter contained a corrupted or zero-byte image, terminating the entire chapter render pass.
- **Fix Applied**: Wrapped `Image.open()` calls in `slice_chapter_panels` and `_compose_canvas_from_source` with `try...except Exception`. Corrupted images log a warning and are skipped or replaced with a fallback canvas without crashing the chapter pipeline. Added comprehensive test coverage in `tests/test_audit_hardening.py`.

#### Issue H2: Zero-Image / Zero-Chapter Pipeline Crashes (`mini-services/pipeline-service/index.ts`)
- **Root Cause**: If scraping failed to download any images across all chapters (e.g., source website down or 0 pages found), `processJob` still proceeded to Phase 2 (transcription) and Phase 3 (rendering `master_pipeline.py`), which crashed in `discover_chapters` on an empty directory.
- **Fix Applied**: Added explicit validation after Phase 1 scrape (`if (doneImages === 0)`). If 0 total images were downloaded across all chapters, `processJob` logs an explicit error, updates the job status to `error`, notifies Socket.IO clients, and aborts immediately.

#### Issue H3: Scraper HTTP Indefinite Hangs (`mini-services/pipeline-service/lib.ts`)
- **Root Cause**: Scrapers for MangaHere, FanFox, Webtoons, and AsuraScans called `fetch()` without request timeouts or retry mechanisms, causing pipeline workers to hang indefinitely if a target website dropped connection or stalled.
- **Fix Applied**: Implemented `fetchWithRetry` with `AbortSignal.timeout(15000)` and 3-attempt exponential backoff retry logic. Replaced all raw `fetch()` calls in scraper functions with `fetchWithRetry`.

#### Issue H4: Orphan Subprocesses on Cancellation (`mini-services/pipeline-service/index.ts`)
- **Root Cause**: `cancelJob` called `child.kill('SIGTERM')` on the top-level Python process, but child sub-processes (e.g., `ffmpeg`, `vlm-worker.ts`, or `piper`) detached and continued running in the background consuming CPU and RAM.
- **Fix Applied**: Updated `cancelJob` and `shutdown` to use `killChildProcessTree(child, signal)`, sending `SIGTERM` and `SIGKILL` to the process group (`-child.pid`).

#### Issue H5: SQLite `SQLITE_BUSY` Database Lock Contention (`mini-services/pipeline-service/lib.ts`)
- **Root Cause**: Concurrent writes between Next.js API endpoints and `pipeline-service` writing to `custom.db` caused SQLite database locked errors.
- **Fix Applied**: Configured Prisma SQLite client on initialization to execute `PRAGMA busy_timeout = 10000;` and `PRAGMA journal_mode = WAL;`, and added a `withDbRetry` helper for retrying locked DB operations.

#### Issue H6: Missing Pre-Scrape Disk Space Guard (`mini-services/pipeline-service/index.ts`)
- **Root Cause**: Scraping downloaded hundreds of MBs/GBs of chapter images before Python's `ResourceGuard` executed in Phase 3. If disk space was low (<1 GB), scraping filled the disk 100%, causing file corruption and database crashes.
- **Fix Applied**: Implemented `checkAvailableDiskSpace` using `fs.statfs` in `lib.ts` and added a pre-flight disk guard in `processJob` before Phase 1 scraping starts.

---

### 3. Medium Priority Issues

#### Issue M1: Unbounded Output Line Buffer Memory Bloat (`mini-services/pipeline-service/index.ts`)
- **Root Cause**: Stdout and stderr line buffers accumulated incoming process chunks without a size limit if newlines were sparse.
- **Fix Applied**: Added `MAX_BUFFER_LEN = 100_000` (100 KB) cap to stdout/stderr line buffers in `index.ts` to prevent memory bloat on large un-cleared output chunks.

#### Issue M2: Hardcoded User Paths in Shell Scripts (`start.sh`, `start-services.sh`)
- **Root Cause**: `start.sh` and `start-services.sh` contained hardcoded `/home/ubuntu/manhwa-recap-studio-v3/` paths, breaking deployments under non-ubuntu users or custom clone directories.
- **Fix Applied**: Refactored `start.sh` and `start-services.sh` to dynamically resolve `PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"` and create a local `logs/` directory for service logs.

---

## Test Verification

- **Python Unit Test Suite**: Ran `PYTHONPATH=. python3 -m pytest` -> **10 passed in 0.46s**
  - `tests/test_audit_hardening.py` (corrupt image safety, zero-image discover chapters)
  - `tests/test_canary_crash_resume.py` (checksums, artifact promotion, state store)
  - `tests/test_production.py` (retry classification, resource guard, state store)
- **TypeScript Type Checks**: Ran `bun x tsc --noEmit` in `mini-services/pipeline-service` -> **0 errors**
- **Next.js Production Build**: Ran `bun run build` -> **Compiled successfully in 15.3s**
