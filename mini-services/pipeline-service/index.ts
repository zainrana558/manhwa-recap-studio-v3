// Load the parent project's .env so API keys (GROQ_API_KEY, GEMINI_API_KEY, etc.) are available.
// The pipeline-service has no .env of its own — it inherits from the Next.js project root.
import { config as loadDotenv } from 'dotenv'
import { resolve } from 'path'
loadDotenv({ path: resolve(import.meta.dirname, '../..', '.env'), override: true })

/**
 * index.ts — Master Recap Pipeline mini-service (port 3001).
 *
 * Responsibilities:
 *  1. socket.io server on path `/` (required by Caddy gateway).
 *  2. HTTP internal endpoints for the Next.js API to trigger/cancel jobs.
 *  3. Job queue (one-at-a-time processing) that:
 *       a) scrapes ALL chapter images from MangaDex (rate-limited, with Referer header)
 *       b) transcribes per-image bubble/caption text using the z-ai-web-dev-sdk VLM
 *       c) spawns the Python master_pipeline.py as a subprocess
 *       d) polls progress.json and streams progress over socket.io
 *
 * The frontend connects via `io("/?XTransformPort=3001")`.
 * The Next.js API triggers via `POST http://localhost:3001/internal/start`.
 */

import { createServer, IncomingMessage, ServerResponse } from 'http'
import { Server, Socket } from 'socket.io'
import { spawn, spawnSync, ChildProcess } from 'child_process'
import { promises as fs } from 'fs'
import path from 'path'

import {
  db,
  ensureDir,
  chapterDir,
  datasetDir,
  workDir,
  outputDir,
  outputVideoPath,
  progressFilePath,
  jobDir,
  PIPELINE_SCRIPT,
  PYTHON_BIN,
  DATA_DIR,
  PROJECT_ROOT,
  getSourceFromId,
  fetchChaptersForSource,
  fetchImagesForSource,
  downloadImageForSource,
  extFromFilename,
  generateImageNarrations,
  generateImageNarrationsOCR,
  runTesseractFallback,
  isPaddleOCRAvailable,
  getOCRModelName,
  filterCreditPanels,
  filterJunkTextPanels,
  sleep,
  fileExists,
} from './lib'
import { isR2Configured, uploadFileToR2 } from './r2'
import { Storage as MegaStorage } from 'megajs'
import { createReadStream } from 'fs'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PORT = 3001

// C3/C5/C10 FIX: Pipeline secret for authenticating internal endpoints.
// Generated at startup if not set via env var.
const PIPELINE_SECRET = process.env.PIPELINE_SECRET || (() => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let s = ''
  for (let i = 0; i < 32; i++) s += chars[Math.floor(Math.random() * chars.length)]
  console.log(`[pipeline-service] Generated PIPELINE_SECRET: ${s}`)
  return s
})()

function checkAuth(req: IncomingMessage): boolean {
  const auth = req.headers.authorization || ''
  if (auth === `Bearer ${PIPELINE_SECRET}`) return true
  // Also allow the secret as a query parameter for socket.io initial connect
  const url = new URL(req.url || '/', 'http://localhost')
  if (url.searchParams.get('secret') === PIPELINE_SECRET) return true
  return false
}

// ---------------------------------------------------------------------------
// HTTP server + socket.io
//
// IMPORTANT: socket.io is configured with `path: '/'` (required by the Caddy
// gateway — the frontend connects via `io("/?XTransformPort=3001")`). With
// path `/`, engine.io's attach() wrapper claims ALL HTTP requests, including
// our `/internal/*` endpoints. To work around this, we install an engine.io
// middleware that intercepts `/internal/*` requests and routes them to our
// HTTP handler before engine.io processes them.
// ---------------------------------------------------------------------------

async function httpHandler(req: IncomingMessage, res: ServerResponse) {
  // M20 FIX: Restrict CORS to configured origin
  const allowedOrigin = process.env.CORS_ORIGIN || '*'
  res.setHeader('Access-Control-Allow-Origin', allowedOrigin)
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  if (req.method === 'OPTIONS') {
    res.writeHead(204)
    res.end()
    return
  }

  const url = (req.url || '/').split('?')[0]

  // Collect request body for POSTs.
  const body = await readBody(req)

  if (req.method === 'GET' && url === '/internal/health') {
    sendJson(res, 200, { ok: true, port: PORT, queue: queueState() })
    return
  }

  if (req.method === 'POST' && url === '/internal/start') {
    if (!checkAuth(req)) { sendJson(res, 401, { error: 'Unauthorized' }); return }
    const { jobId } = body as { jobId?: string }
    if (!jobId) {
      sendJson(res, 400, { error: 'jobId required' })
      return
    }
    const job = await db.job.findUnique({ where: { id: jobId } })
    if (!job) {
      sendJson(res, 404, { error: 'job not found' })
      return
    }
    enqueueJob(jobId)
    sendJson(res, 202, { ok: true, jobId, queued: true })
    return
  }

  if (req.method === 'POST' && url === '/internal/cancel') {
    if (!checkAuth(req)) { sendJson(res, 401, { error: 'Unauthorized' }); return }
    const { jobId } = body as { jobId?: string }
    if (!jobId) {
      sendJson(res, 400, { error: 'jobId required' })
      return
    }
    await cancelJob(jobId)
    sendJson(res, 200, { ok: true, jobId, cancelled: true })
    return
  }

  // --- Voice preview ---------------------------------------------------------
  // GET /preview/voice?voice={voiceId}
  // Generates (or serves from cache) a short ~4-7s edge-tts preview MP3 so
  // users can hear how a narration voice sounds before starting a job. This
  // lives in the mini-service (not the Next.js API) because edge-tts requires
  // Python, which isn't available on serverless hosts like Vercel. The Next.js
  // /api/voice-preview route proxies to this endpoint.
  if (req.method === 'GET' && url === '/preview/voice') {
    const fullUrl = new URL(req.url || '', 'http://localhost')
    const voice = fullUrl.searchParams.get('voice') || ''
    const VOICE_ID_RE = /^[a-z]{2}-[A-Z]{2}-[A-Za-z0-9]+Neural$/
    if (!VOICE_ID_RE.test(voice)) {
      sendJson(res, 400, { error: 'Invalid or missing voice parameter.' })
      return
    }
    const path = await import('path')
    const cacheDir = path.join(DATA_DIR, 'cache', 'voice-preview')
    const cacheFile = path.join(cacheDir, `${voice}.mp3`)
    const fs = await import('fs/promises')

    try {
      // Serve from cache if present + non-empty.
      try {
        const stat = await fs.stat(cacheFile)
        if (stat.size > 100) {
          const data = await fs.readFile(cacheFile)
          res.writeHead(200, {
            'Content-Type': 'audio/mpeg',
            'Content-Length': String(stat.size),
            'Cache-Control': 'public, max-age=86400, immutable',
          })
          res.end(data)
          return
        }
      } catch {
        // not cached — fall through to generation
      }

      // Generate via Python.
      await fs.mkdir(cacheDir, { recursive: true })
      const scriptPath = path.join(PROJECT_ROOT, 'pipeline', 'voice_preview.py')
      const result = spawnSync(PYTHON_BIN, [scriptPath, '--voice', voice, '--output', cacheFile], {
        encoding: 'utf8',
        timeout: 25000,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      })
      if (result.status !== 0 || !(await fileExists(cacheFile))) {
        console.error('[pipeline-service] voice preview failed for', voice, (result.stderr || '').slice(-200))
        sendJson(res, 502, { error: 'Failed to generate voice preview.' })
        return
      }
      const data = await fs.readFile(cacheFile)
      res.writeHead(200, {
        'Content-Type': 'audio/mpeg',
        'Content-Length': String(data.length),
        'Cache-Control': 'public, max-age=86400, immutable',
      })
      res.end(data)
      return
    } catch (err) {
      console.error('[pipeline-service] voice preview error:', err)
      sendJson(res, 500, { error: 'Internal error.' })
      return
    }
  }

  sendJson(res, 404, { error: 'not found' })
}

const httpServer = createServer((req, res) => {
  // IMPORTANT: engine.io (socket.io) also registers a 'request' listener on
  // this server. Both listeners fire for EVERY request. To avoid conflicts:
  //   - For /internal/* and /preview/*: we handle it and call res.end().
  //     Engine.io will also fire but will see the response is already finished
  //     and silently skip (or get a "headers sent" error that it swallows).
  //   - For all other requests: we do nothing. Engine.io handles them.
  const reqUrl: string = req.url || '/'
  const urlPath = reqUrl.split('?')[0]

  if (urlPath.startsWith('/internal/') || urlPath.startsWith('/preview/')) {
    void httpHandler(req, res)
    return
  }
  // Do nothing for socket.io paths — engine.io handles them.
})

const io = new Server(httpServer, {
  path: '/',
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    methods: ['GET', 'POST', 'OPTIONS'],
  },
  pingTimeout: 60000,
  pingInterval: 25000,
  // C3 FIX: Require secret for initial connection
  allowRequest: (req, callback) => {
    const url = new URL(req.url || '/', 'http://localhost')
    const secret = url.searchParams.get('secret') || req.headers['x-pipeline-secret']
    if (secret === PIPELINE_SECRET) {
      callback(null, true)
    } else {
      callback('Unauthorized', false)
    }
  },
})

// Patch engine.io's handleRequest to skip /internal/* and /preview/* paths.
// This is the reliable way to prevent engine.io from intercepting our HTTP
// endpoints, since io.engine.use() doesn't work for POST requests.
const originalHandleRequest = io.engine.handleRequest.bind(io.engine)
io.engine.handleRequest = function (req: any, res: any) {
  const reqUrl: string = req.url || '/'
  const urlPath = reqUrl.split('?')[0]
  if (urlPath.startsWith('/internal/') || urlPath.startsWith('/preview/')) {
    // Let our createServer callback handle it — do nothing here.
    // If the response is already finished (by our callback), skip.
    if (res.writableEnded || res.headersSent) return
    // If not yet handled, handle it now as fallback.
    void httpHandler(req, res)
    return
  }
  return originalHandleRequest(req, res)
}

// ---------------------------------------------------------------------------
// Socket.io connection handling
// ---------------------------------------------------------------------------

io.on('connection', (socket: Socket) => {
  console.log(`[io] connected ${socket.id}`)

  // C3 FIX: Track authenticated socket to prevent cross-job data leakage
  socket.data.authenticated = true

  socket.on('subscribe', async (payload: unknown) => {
    const jobId = extractJobId(payload)
    if (!jobId) return
    const room = `job:${jobId}`
    await socket.join(room)
    socket.emit('subscribed', { type: 'subscribed', jobId })
    // Immediately emit current status + recent logs.
    await emitStatus(jobId)
    await emitRecentLogs(jobId, socket)
  })

  socket.on('unsubscribe', async (payload: unknown) => {
    const jobId = extractJobId(payload)
    if (!jobId) return
    await socket.leave(`job:${jobId}`)
  })

  socket.on('cancel', async (payload: unknown) => {
    const jobId = extractJobId(payload)
    if (!jobId) return
    await cancelJob(jobId)
  })

  socket.on('disconnect', () => {
    // rooms are auto-cleaned
  })
})

function extractJobId(payload: unknown): string | null {
  if (!payload) return null
  if (typeof payload === 'string') return payload
  if (typeof payload === 'object') {
    const p = payload as { jobId?: string }
    if (typeof p.jobId === 'string') return p.jobId
  }
  return null
}


type VideoQaResult = { ok: true; sizeBytes: number; durationSec: number } | { ok: false; error: string }

async function validateFinalVideoArtifact(filePath: string): Promise<VideoQaResult> {
  let st
  try {
    st = await fs.stat(filePath)
  } catch (err) {
    return { ok: false, error: `output file is missing or unreadable: ${err instanceof Error ? err.message : String(err)}` }
  }
  if (!st.isFile() || st.size < 1024) {
    return { ok: false, error: `output file is too small or not a regular file (${st.size} bytes)` }
  }

  const probe = spawnSync('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration:stream=codec_type',
    '-of', 'json',
    filePath,
  ], { encoding: 'utf8', timeout: 30000, shell: false })

  if (probe.error) return { ok: false, error: `ffprobe unavailable or failed to start: ${probe.error.message}` }
  if (probe.status !== 0) return { ok: false, error: `ffprobe failed: ${(probe.stderr || probe.stdout || '').slice(0, 500)}` }

  try {
    const parsed: unknown = JSON.parse(probe.stdout || '{}')
    if (typeof parsed !== 'object' || parsed === null) return { ok: false, error: 'ffprobe returned malformed JSON' }
    const rec = parsed as { format?: { duration?: unknown }; streams?: Array<{ codec_type?: unknown }> }
    const duration = Number(rec.format?.duration)
    const streams = Array.isArray(rec.streams) ? rec.streams : []
    if (!Number.isFinite(duration) || duration <= 0) return { ok: false, error: 'video has no positive duration' }
    if (!streams.some((stream) => stream.codec_type === 'video')) return { ok: false, error: 'video stream missing' }
    if (!streams.some((stream) => stream.codec_type === 'audio')) return { ok: false, error: 'audio stream missing' }
    return { ok: true, sizeBytes: st.size, durationSec: duration }
  } catch (err) {
    return { ok: false, error: `could not parse ffprobe output: ${err instanceof Error ? err.message : String(err)}` }
  }
}

// ---------------------------------------------------------------------------
// Emit helpers
// ---------------------------------------------------------------------------

async function emitStatus(jobId: string): Promise<void> {
  const job = await loadJobDetail(jobId)
  if (!job) return
  io.to(`job:${jobId}`).emit('status', { type: 'status', job })
}

async function emitLog(
  jobId: string,
  level: 'info' | 'warn' | 'error' | 'success',
  stage: string | null,
  message: string,
): Promise<void> {
  const log = await db.jobLog.create({
    data: { jobId, level, stage, message },
  })
  const entry = {
    id: log.id,
    jobId: log.jobId,
    level: log.level as 'info' | 'warn' | 'error' | 'success',
    stage: log.stage,
    message: log.message,
    createdAt: log.createdAt.toISOString(),
  }
  io.to(`job:${jobId}`).emit('log', { type: 'log', log: entry })

  // Also update Job.message so subscribers get it on next status poll.
  await db.job.update({
    where: { id: jobId },
    data: { message: message.slice(0, 500), stage: stage ?? undefined },
  }).catch(() => undefined)
}

async function emitProgress(
  jobId: string,
  fields: {
    progress: number
    doneChapters: number
    totalChapters: number
    doneImages: number
    totalImages: number
    stage: string
    message: string
  },
): Promise<void> {
  await db.job.update({
    where: { id: jobId },
    data: {
      progress: Math.max(0, Math.min(100, Math.round(fields.progress))),
      doneChapters: fields.doneChapters,
      totalChapters: fields.totalChapters,
      doneImages: fields.doneImages,
      totalImages: fields.totalImages,
      stage: fields.stage,
      message: fields.message.slice(0, 500),
    },
  }).catch(() => undefined)

  io.to(`job:${jobId}`).emit('progress', {
    type: 'progress',
    jobId,
    progress: Math.max(0, Math.min(100, Math.round(fields.progress))),
    doneChapters: fields.doneChapters,
    totalChapters: fields.totalChapters,
    doneImages: fields.doneImages,
    totalImages: fields.totalImages,
    stage: fields.stage,
    message: fields.message,
  })
}

async function emitChapter(jobId: string, chapter: {
  id: string
  jobId: string
  index: number
  mangadexId: string
  chapterNum: string | null
  title: string | null
  language: string
  pageCount: number
  translated: boolean
  transcribed: boolean
  rendered: boolean
  folder: string
  status: string
  error: string | null
}): Promise<void> {
  io.to(`job:${jobId}`).emit('chapter', {
    type: 'chapter',
    jobId,
    chapter: {
      index: chapter.index,
      mangadexId: chapter.mangadexId,
      chapterNum: chapter.chapterNum,
      title: chapter.title,
      language: chapter.language,
      pageCount: chapter.pageCount,
      translated: chapter.translated,
      transcribed: chapter.transcribed,
      rendered: chapter.rendered,
      status: chapter.status,
      error: chapter.error,
    },
  })
}

async function emitRecentLogs(jobId: string, socket: Socket): Promise<void> {
  const logs = await db.jobLog.findMany({
    where: { jobId },
    orderBy: { createdAt: 'desc' },
    take: 50,
  })
  for (const log of logs.reverse()) {
    socket.emit('log', {
      type: 'log',
      log: {
        id: log.id,
        jobId: log.jobId,
        level: log.level as 'info' | 'warn' | 'error' | 'success',
        stage: log.stage,
        message: log.message,
        createdAt: log.createdAt.toISOString(),
      },
    })
  }
}

// ---------------------------------------------------------------------------
// DB -> JobDetail mapping
// ---------------------------------------------------------------------------

async function loadJobDetail(jobId: string) {
  const job = await db.job.findUnique({
    where: { id: jobId },
    include: { chapters: { orderBy: { index: 'asc' } } },
  })
  if (!job) return null
  return {
    id: job.id,
    mangaId: job.mangaId,
    mangaTitle: job.mangaTitle,
    coverUrl: job.coverUrl ?? null,
    language: job.language,
    sourceLang: job.sourceLang,
    status: job.status as any,
    progress: job.progress,
    stage: job.stage,
    message: job.message,
    totalChapters: job.totalChapters,
    doneChapters: job.doneChapters,
    totalImages: job.totalImages,
    doneImages: job.doneImages,
    outputDir: job.outputDir,
    outputVideo: job.outputVideo,
    r2Key: job.r2Key ?? null,
    archiveProvider: job.archiveProvider ?? null,
    archiveFileId: job.archiveFileId ?? null,
    autoArchive: job.autoArchive ?? false,
    error: job.error,
    voice: job.voice,
    chapterLimit: job.chapterLimit,
    translate: job.translate,
    bgmPath: job.bgmPath ?? null,
    useBgm: job.useBgm ?? true,
    createdAt: job.createdAt.toISOString(),
    updatedAt: job.updatedAt.toISOString(),
    chapters: job.chapters.map((c) => ({
      index: c.index,
      mangadexId: c.mangadexId,
      chapterNum: c.chapterNum,
      title: c.title,
      language: c.language,
      pageCount: c.pageCount,
      translated: c.translated,
      transcribed: c.transcribed,
      rendered: c.rendered,
      status: c.status,
      error: c.error,
    })),
  }
}

// ---------------------------------------------------------------------------
// Job queue (one at a time)
// ---------------------------------------------------------------------------

const queue: string[] = []
let currentlyRunning: string | null = null
const childProcesses = new Map<string, ChildProcess>()
const cancelledJobs = new Set<string>()

function queueState() {
  return {
    queueLength: queue.length,
    currentlyRunning,
    cancelled: Array.from(cancelledJobs),
  }
}

function enqueueJob(jobId: string) {
  if (queue.includes(jobId) || currentlyRunning === jobId) return
  queue.push(jobId)
  void processQueue()
}

async function processQueue() {
  if (currentlyRunning) return
  const next = queue.shift()
  if (!next) return
  currentlyRunning = next
  try {
    await processJob(next)
  } catch (err) {
    console.error(`[queue] processJob threw for ${next}:`, err)
    try {
      await db.job.update({
        where: { id: next },
        data: {
          status: 'error',
          error: err instanceof Error ? err.message : String(err),
        },
      })
      io.to(`job:${next}`).emit('error', {
        type: 'error',
        jobId: next,
        error: err instanceof Error ? err.message : String(err),
      })
    } catch {
      // ignore
    }
  } finally {
    currentlyRunning = null
    cancelledJobs.delete(next)
    // Process the next queued job (if any).
    void processQueue()
  }
}

// ---------------------------------------------------------------------------
// Cancel
// ---------------------------------------------------------------------------

async function cancelJob(jobId: string): Promise<void> {
  cancelledJobs.add(jobId)
  const child = childProcesses.get(jobId)
  if (child) {
    try {
      child.kill('SIGTERM')
      // Force-kill after 3s if still alive.
      setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch {
          // ignore
        }
      }, 3000)
    } catch {
      // ignore
    }
  }
  await db.job.update({
    where: { id: jobId },
    data: { status: 'cancelled', stage: 'cancelled', message: 'Cancelled by user' },
  }).catch(() => undefined)
  await emitLog(jobId, 'warn', 'cancel', 'Job cancelled by user.')
  io.to(`job:${jobId}`).emit('cancelled', { type: 'cancelled', jobId })
  await emitStatus(jobId)
}

// ---------------------------------------------------------------------------
// The core job pipeline
// ---------------------------------------------------------------------------

/**
 * Run the Python pipeline in --slice-only mode for a job.
 *
 * This pre-slices every chapter's page images into individual panel frames
 * (work/temp_slices/chapter_XXX/frame_NNNNN.jpg + manifest.json) BEFORE the
 * VLM transcription step runs. That way the VLM reads ONE panel at a time
 * instead of a full page with multiple panels — dramatically improving
 * transcription accuracy and per-panel narration sync.
 *
 * The full render pipeline later reuses these slices via manifest resume
 * support, so no work is duplicated.
 *
 * Returns true on success, false on failure (non-fatal — the caller falls
 * back to full-page VLM transcription).
 */
async function sliceJobChapters(jobId: string): Promise<boolean> {
  const args = [
    PIPELINE_SCRIPT,
    '--input-dir', datasetDir(jobId),
    '--output', outputVideoPath(jobId),
    '--work-dir', workDir(jobId),
    '--voice', 'en-US-AndrewNeural', // unused by slice-only, but required by argparse
    '--narration-provider', 'none',
    '--job-id', jobId,
    ...(process.env.PRODUCTION_PIPELINE === '0' ? [] : ['--production-mode']),
    '--slice-only',
    '--keep-temp',
  ]

  await emitLog(jobId, 'info', 'slice', `Slicing chapters into individual panels (python --slice-only)…`)

  let spawnCwd = jobDir(jobId)
  try {
    await fs.mkdir(spawnCwd, { recursive: true })
  } catch {
    spawnCwd = process.cwd()
  }

  const result = spawnSync(PYTHON_BIN, args, {
    cwd: spawnCwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    encoding: 'utf8',
    timeout: 10 * 60 * 1000, // 10 min hard cap
  })

  if (result.error) {
    await emitLog(jobId, 'error', 'slice', `Slice step failed to spawn: ${result.error.message}`)
    return false
  }
  if (result.status !== 0) {
    const tail = (result.stderr || result.stdout || '').slice(-500)
    await emitLog(jobId, 'error', 'slice', `Slice step exited ${result.status}: ${tail}`)
    return false
  }

  // Log the last few stdout lines so the user sees per-chapter frame counts.
  const lines = (result.stdout || '').split('\n').filter((l) => l.trim()).slice(-6)
  for (const line of lines) {
    await emitLog(jobId, 'info', 'slice', line)
  }
  return true
}

async function processJob(jobId: string): Promise<void> {
  console.log(`[job:${jobId}] starting`)
  cancelledJobs.delete(jobId)

  const job = await db.job.findUnique({
    where: { id: jobId },
    include: { chapters: { orderBy: { index: 'asc' } } },
  })
  if (!job) {
    console.error(`[job:${jobId}] not found`)
    return
  }
  if (job.status === 'cancelled') {
    console.log(`[job:${jobId}] already cancelled, skipping`)
    return
  }

  // Prepare filesystem.
  await ensureDir(jobDir(jobId))
  await ensureDir(datasetDir(jobId))
  await ensureDir(workDir(jobId))
  await ensureDir(outputDir(jobId))

  // -----------------------------
  // Phase 1: SCRAPE
  // -----------------------------
  await db.job.update({
    where: { id: jobId },
    data: { status: 'scraping', stage: 'scrape', message: 'Starting scrape' },
  })
  await emitStatus(jobId)
  await emitLog(jobId, 'info', 'scrape', `Starting scrape for "${job.mangaTitle}"`)

  // If the API route did not pre-create Chapter rows, fetch them now.
  let chapters = job.chapters
  if (!chapters || chapters.length === 0) {
    const source = getSourceFromId(job.mangaId)
    if (!source) {
      throw new Error(`Cannot determine scraping source from manga ID: ${job.mangaId}`)
    }
    await emitLog(jobId, 'info', 'scrape', `Fetching chapter list from ${source}`)
    const fetched = await fetchChaptersForSource(source, job.mangaId, job.chapterLimit)
    if (fetched.length === 0) {
      throw new Error(`No chapters found for manga ${job.mangaId} on ${source}`)
    }
    // Create Chapter rows.
    for (let i = 0; i < fetched.length; i++) {
      const c = fetched[i]
      await db.chapter.create({
        data: {
          jobId,
          index: i + 1,
          mangadexId: c.mangadexId,
          chapterNum: c.chapterNum,
          title: c.title,
          language: c.language,
          pageCount: 0,
          folder: `chapter_${String(i + 1).padStart(3, '0')}`,
          status: 'pending',
        },
      })
    }
    chapters = await db.chapter.findMany({
      where: { jobId },
      orderBy: { index: 'asc' },
    })
    await db.job.update({
      where: { id: jobId },
      data: {
        totalChapters: chapters.length,
        sourceLang: chapters[0]?.language ?? job.language,
      },
    })
  } else {
    await db.job.update({
      where: { id: jobId },
      data: {
        totalChapters: chapters.length,
        sourceLang: chapters[0]?.language ?? job.language,
      },
    })
  }

  await emitLog(jobId, 'info', 'scrape', `Found ${chapters.length} chapters to scrape`)

  // Track total image counts.
  let totalImages = 0
  let doneImages = 0

  // Scrape each chapter sequentially (rate-limit friendly).
  for (const ch of chapters) {
    if (cancelledJobs.has(jobId)) {
      await emitLog(jobId, 'warn', 'scrape', 'Cancelled during scrape')
      return
    }
    // Skip chapters already scraped (resume support).
    if (ch.status === 'scraped') {
      const cDir = chapterDir(jobId, ch.index)
      try {
        const existing = (await fs.readdir(cDir)).filter(f => /\.(jpe?g|png|webp|gif)$/i.test(f)).length
        if (existing > 0) {
          doneImages += existing
          totalImages = Math.max(totalImages, doneImages)
          await emitLog(jobId, 'info', 'scrape', `Chapter ${ch.index} already scraped (${existing} images) — skipping`)
          continue
        }
      } catch { /* dir missing, re-scrape */ }
    }
    try {
      const cDir = chapterDir(jobId, ch.index)
      await ensureDir(cDir)

      const source = getSourceFromId(job.mangaId)
      if (!source) {
        throw new Error(`Cannot determine scraping source from manga ID: ${job.mangaId}`)
      }
      const imageUrls = await fetchImagesForSource(source, job.mangaId, ch.mangadexId)
      if (imageUrls.length === 0) {
        await emitLog(jobId, 'warn', 'scrape', `Chapter ${ch.index} has 0 pages, skipping`)
        await db.chapter.update({
          where: { id: ch.id },
          data: { status: 'error', error: 'No pages' },
        })
        continue
      }

      let downloaded = 0
      for (let i = 0; i < imageUrls.length; i++) {
        if (cancelledJobs.has(jobId)) {
          await emitLog(jobId, 'warn', 'scrape', 'Cancelled mid-chapter')
          return
        }
        const destName = `${String(i + 1).padStart(3, '0')}.jpg`
        const destPath = path.join(cDir, destName)
        // Skip if already downloaded (resumable).
        if (await fileExists(destPath)) {
          downloaded++
          continue
        }
        try {
          await downloadImageForSource(source, imageUrls[i], destPath)
          downloaded++
          doneImages++
          // Rate limit: 300ms between images.
          await sleep(300)
        } catch (err) {
          await emitLog(
            jobId,
            'warn',
            'scrape',
            `Failed image ${i + 1} of chapter ${ch.index}: ${err instanceof Error ? err.message : String(err)}`,
          )
        }
        totalImages++
        // Emit progress every few images.
        if (i % 3 === 0 || i === imageUrls.length - 1) {
          await emitProgress(jobId, {
            progress: 5 + (doneImages / Math.max(1, totalImages + (imageUrls.length - i - 1))) * 25,
            doneChapters: 0,
            totalChapters: chapters.length,
            doneImages,
            totalImages,
            stage: 'scrape',
            message: `Scraping chapter ${ch.index}/${chapters.length}: image ${i + 1}/${imageUrls.length}`,
          })
        }
      }

      // Recount totalImages for this chapter to keep counts accurate.
      totalImages = Math.max(totalImages, doneImages)

      const updated = await db.chapter.update({
        where: { id: ch.id },
        data: {
          pageCount: downloaded,
          status: 'scraped',
          folder: `chapter_${String(ch.index).padStart(3, '0')}`,
        },
      })
      await db.job.update({
        where: { id: jobId },
        data: { doneImages, totalImages },
      })
      await emitChapter(jobId, updated)
      await emitLog(
        jobId,
        'success',
        'scrape',
        `Chapter ${ch.index}/${chapters.length} scraped: ${downloaded} images`,
      )
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      await emitLog(jobId, 'error', 'scrape', `Chapter ${ch.index} scrape failed: ${msg}`)
      const updated = await db.chapter.update({
        where: { id: ch.id },
        data: { status: 'error', error: msg },
      })
      await emitChapter(jobId, updated)
    }
  }

  // Finalize scrape counts.
  await db.job.update({
    where: { id: jobId },
    data: { totalImages, doneImages, doneChapters: 0 },
  })
  await emitLog(
    jobId,
    'success',
    'scrape',
    `Scrape complete: ${doneImages} images across ${chapters.length} chapters`,
  )

  if (cancelledJobs.has(jobId)) return

  // -----------------------------
  // Phase 1b: CANONICAL FRAME CREATION (Python --slice-only)
  // -----------------------------
  // Slices chapter source pages into canonical panel and scroll frames,
  // creating work/temp_slices/chap_XXX/frame_NNNNN.jpg + manifest.json.
  // The VLM or PaddleOCR then transcribes text mapped to individual canonical frames.
  const sliced = await sliceJobChapters(jobId)
  if (sliced) {
    await emitLog(jobId, 'success', 'slice', 'Canonical frames created and manifest validated.')
  } else {
    await emitLog(jobId, 'warn', 'slice', 'Canonical frame creation failed — falling back to full-page images.')
  }

  // -----------------------------
  // Phase 2: TRANSCRIBE — read bubble/caption text from each panel.
  // Strategy: PaddleOCR PP-OCRv5 (PRIMARY, local, no API keys needed) →
  //           VLM providers (FALLBACK, requires API keys, slower)
  // -----------------------------
  await db.job.update({
    where: { id: jobId },
    data: { status: 'transcribing', stage: 'transcribe', message: 'Transcribing panel text' },
  })
  await emitStatus(jobId)

  // Check PaddleOCR availability ONCE at the start of transcription phase.
  // If the service is up, we use it for ALL chapters. If not, we fall back
  // to VLM for ALL chapters (no per-chapter switching — that would waste
  // time re-initializing providers).
  const ocrAvailable = await isPaddleOCRAvailable()
  let ocrModelName = 'unknown'
  if (ocrAvailable) {
    ocrModelName = await getOCRModelName()
    await emitLog(jobId, 'success', 'transcribe',
      `PaddleOCR ${ocrModelName} detected — using as PRIMARY transcriptor (fast, local, no API keys needed)`,
    )
  } else {
    await emitLog(jobId, 'info', 'transcribe',
      'PaddleOCR service not available — falling back to VLM providers (requires API keys)',
    )
  }
  await emitLog(jobId, 'info', 'transcribe', 'Transcribing speech bubbles and captions from each panel image')

  // Set VLM API keys from the per-job record (user-entered in the UI).
  // Per-job keys ALWAYS override .env keys — the user explicitly chose these.
  // If the job has no per-job key, fall back to the global settings table
  // (saved via the Settings UI). This ensures jobs created before the user
  // saved their API key in settings still work after the key is added.
  // We snapshot the originals so they can be restored after the job finishes.
  const _envBackup: Record<string, string | undefined> = {}
  const SETTING_ENV_MAP: Array<[string, string | null, string]> = [
    ['GROQ_API_KEY', job.groqKey, 'groqKey'],
    ['GEMINI_API_KEY', job.geminiKey, 'geminiKey'],
    ['OPENROUTER_API_KEY', job.openRouterKey, 'openRouterKey'],
    ['ZHIPU_API_KEY', job.zhipuKey, 'zhipuKey'],
    ['SILICONFLOW_API_KEY', job.siliconFlowKey, 'siliconFlowKey'],
    ['OPENAI_API_KEY', job.openaiKey, 'openaiKey'],
  ]
  // Batch-read settings that we need (only for keys missing from the job).
  const missingSettings = SETTING_ENV_MAP.filter(([, jobVal]) => !jobVal).map(([, , settingId]) => settingId)
  const settingRows = missingSettings.length > 0
    ? await db.setting.findMany({ where: { id: { in: missingSettings } } })
    : []
  const settingsMap = new Map(settingRows.map((r) => [r.id, r.value]))
  for (const [envKey, jobValue, settingId] of SETTING_ENV_MAP) {
    const effectiveValue = jobValue || settingsMap.get(settingId) || ''
    if (effectiveValue) {
      _envBackup[envKey] = process.env[envKey]
      process.env[envKey] = effectiveValue
      const source = jobValue ? 'per-job' : 'global settings'
      console.log(`[VLM] Using ${source} ${envKey} for transcription`)
    }
  }

  // Reload chapters to get the latest state.
  const scrapedChapters = await db.chapter.findMany({
    where: { jobId, status: 'scraped' },
    orderBy: { index: 'asc' },
  })

  let transcribedCount = 0
  for (const ch of scrapedChapters) {
    if (cancelledJobs.has(jobId)) {
      await emitLog(jobId, 'warn', 'transcribe', 'Cancelled during transcription')
      return
    }
    const cDir = chapterDir(jobId, ch.index)

    // If slicing succeeded, transcribe each sliced panel frame.
    // The Python slicer creates frames at work/temp_slices/chap_XXX/frame_NNNNN.jpg.
    // VLM reads individual panels for precise per-panel narration.
    // Fallback: transcribe full-page images from the chapter dataset dir.
    let imageFiles: string[] = []
    let frameKeyed = false
    if (sliced) {
      const slicesDir = path.join(workDir(jobId), 'temp_slices', `chap_${String(ch.index).padStart(3, '0')}`)
      try {
        const entries = await fs.readdir(slicesDir)
        imageFiles = entries
          .filter((f) => /^frame_\d+\.jpe?g$/i.test(f))
          .sort()
          .map((f) => path.join(slicesDir, f))
        if (imageFiles.length > 0) {
          frameKeyed = true
        }
      } catch {
        // slices dir missing — fall through to full-page mode
      }
    }
    if (imageFiles.length === 0) {
      // Fallback: full-page images from the chapter dataset dir.
      try {
        const entries = await fs.readdir(cDir)
        imageFiles = entries
          .filter((f) => /\.(jpe?g|png|webp|gif)$/i.test(f))
          .sort()
          .map((f) => path.join(cDir, f))
      } catch {
        // no images, skip
        continue
      }
    }
    if (imageFiles.length === 0) {
      const updated = await db.chapter.update({
        where: { id: ch.id },
        data: { status: 'transcribed', transcribed: true },
      })
      await emitChapter(jobId, updated)
      transcribedCount++
      continue
    }

    // Skip if narration.json already exists (resume support).
    const narrationFile = path.join(cDir, 'narration.json')
    if (await fileExists(narrationFile)) {
      await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index} transcriptions already cached — skipping`)
      const updated = await db.chapter.update({
        where: { id: ch.id },
        data: { status: 'transcribed', transcribed: true },
      })
      await emitChapter(jobId, updated)
      transcribedCount++
      continue
    }

    try {
      const modeLabel = frameKeyed ? 'sliced panels' : 'full-page images'
      let rawNarrations: Array<{ image: string; text: string; status?: string }>
      let usedMethod = 'unknown'

      if (ocrAvailable) {
        // ── PRIMARY: PaddleOCR PP-OCRv5 (fast, local, no API keys) ──
        await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: transcribing ${imageFiles.length} ${modeLabel} with PaddleOCR ${ocrModelName}...`)
        try {
          const ocrOutcome = await generateImageNarrationsOCR(imageFiles, (done, total) => {
            void emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: ${done}/${total} ${frameKeyed ? 'panels' : 'images'} OCR'd`)
          })
          rawNarrations = ocrOutcome.results
          usedMethod = `PaddleOCR ${ocrModelName}`

          const emptyCount = rawNarrations.filter((n) => !n.text.trim()).length
          const emptyRatio = emptyCount / rawNarrations.length

          // Genuine OCR failure signals — worth falling back to VLM for:
          //  1. The service call itself errored (network/service down).
          //  2. The text DETECTOR never fired once across the whole chapter
          //     (totalRegionsDetected === 0) despite processing panels.
          //     This is what a broken model/init/preprocessing bug looks
          //     like — not just "few panels have dialogue".
          // A high empty-text ratio ALONE is explicitly NOT treated as
          // failure: manhwa/manhua chapters are often mostly action,
          // establishing, or transition panels with no bubbles at all, so
          // 80%+ silent panels is frequently correct output, not broken OCR.
          const { totalRegionsDetected, batchCallFailures, imageCallFailures, freshlyProcessed, uncertainWithRegions, panelsWithRegionsDetected } = ocrOutcome.stats
          const detectorNeverFired = freshlyProcessed > 3 && totalRegionsDetected === 0
          // Distinct from detectorNeverFired: text regions WERE found, but
          // the recognizer couldn't read most of them confidently (status
          // UNCERTAIN/FAILED), so that dialogue got silently discarded as
          // if the panel were quiet. Left unchecked this fails exactly the
          // way the OCR plan calls out: "OCR returns empty text while a
          // large speech bubble is visually present" being accepted as a
          // legitimate empty panel instead of triggering the fallback
          // cascade. Only trip this once there's a reasonable sample size
          // and it's clearly systemic, not a couple of genuinely hard crops.
          //
          // Both sides of this ratio must be panel counts: uncertainWithRegions
          // is "how many panels had detected-but-unreadable text", so the
          // denominator has to be "how many panels had any detected text at
          // all" (panelsWithRegionsDetected) — not totalRegionsDetected, which
          // is a summed *region* count across every panel and is on a
          // different scale entirely (that mismatch used to make this ratio
          // nearly impossible to trip even when recognition was genuinely
          // failing on most panels with text).
          const uncertainRatio = panelsWithRegionsDetected > 0 ? uncertainWithRegions / panelsWithRegionsDetected : 0
          const recognizerStruggling = uncertainWithRegions >= 4 && uncertainRatio > 0.5

          // batchCallFailures counts failed HTTP *chunks* (BATCH_SIZE=3
          // images each in lib.ts) — dividing that by freshlyProcessed
          // (an *image* count) mixed units and meant the intended 30%
          // failure threshold didn't actually trip until roughly ~47% of
          // images were failing. imageCallFailures counts individual
          // images that ended up FAILED (via a failed chunk OR a non-
          // SUCCESS per-image status in a chunk that otherwise succeeded),
          // so it's the correct numerator against freshlyProcessed.
          const imageFailureRatio = freshlyProcessed > 0 ? imageCallFailures / freshlyProcessed : 0
          const ocrServiceFailing = freshlyProcessed <= 3 ? imageCallFailures > 0 : imageFailureRatio > 0.3

          if (ocrServiceFailing) {
            await emitLog(jobId, 'warn', 'transcribe',
              `Chapter ${ch.index}: PaddleOCR failed on ${imageCallFailures}/${freshlyProcessed} panels (${batchCallFailures} failed HTTP chunk(s)) — falling back to VLM for this chapter`,
            )
            throw new Error(`OCR service unreachable/erroring (${imageCallFailures}/${freshlyProcessed} panel failures)`)
          }
          if (detectorNeverFired) {
            await emitLog(jobId, 'warn', 'transcribe',
              `Chapter ${ch.index}: PaddleOCR detected zero text regions across ${freshlyProcessed} panels — likely a broken OCR pipeline, not just silent panels. Falling back to VLM.`,
            )
            throw new Error(`OCR detector never fired across ${freshlyProcessed} panels`)
          }
          if (recognizerStruggling) {
            await emitLog(jobId, 'warn', 'transcribe',
              `Chapter ${ch.index}: PaddleOCR detected text in ${uncertainWithRegions} panels but couldn't read it confidently (${Math.round(uncertainRatio * 100)}% uncertain/failed) — dialogue would be silently dropped. Falling back to VLM.`,
            )
            throw new Error(`OCR recognizer low-confidence on ${uncertainWithRegions} panels with detected text`)
          }
          if (emptyRatio > 0.8 && rawNarrations.length > 3) {
            // Informational only — do NOT throw / fall back to VLM. The
            // detector did fire (totalRegionsDetected > 0), so this is most
            // likely a genuinely quiet chapter, not a broken pipeline.
            await emitLog(jobId, 'info', 'transcribe',
              `Chapter ${ch.index}: ${emptyCount}/${rawNarrations.length} panels (${Math.round(emptyRatio * 100)}%) have no dialogue — this is normal for action-heavy chapters. OCR detector is active (${totalRegionsDetected} regions found), keeping PaddleOCR results.`,
            )
          }
        } catch (ocrErr) {
          // OCR failed or returned poor results — attempt local Tesseract fallback before VLM.
          const ocrMsg = ocrErr instanceof Error ? ocrErr.message : String(ocrErr)
          await emitLog(jobId, 'warn', 'transcribe',
            `Chapter ${ch.index}: PaddleOCR failed or low quality (${ocrMsg.slice(0, 100)}) — attempting local Tesseract fallback before VLM`,
          )
          const tessOutcome = await runTesseractFallback(imageFiles)
          if (tessOutcome && tessOutcome.some((n: { text: string }) => n.text.trim())) {
            rawNarrations = tessOutcome
            usedMethod = 'Tesseract OCR (local fallback)'
            await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: transcribed via local Tesseract OCR`)
          } else {
            rawNarrations = await generateImageNarrations(imageFiles, (done, total) => {
              void emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: ${done}/${total} ${frameKeyed ? 'panels' : 'images'} transcribed via VLM`)
            })
            usedMethod = 'VLM (fallback)'
          }
        }
      } else {
        // ── FALLBACK 1: Local Tesseract OCR ──
        const tessOutcome = await runTesseractFallback(imageFiles)
        if (tessOutcome && tessOutcome.some((n: { text: string }) => n.text.trim())) {
          rawNarrations = tessOutcome
          usedMethod = 'Tesseract OCR (local)'
          await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: transcribed ${imageFiles.length} ${modeLabel} with local Tesseract OCR`)
        } else {
          // ── FALLBACK 2: VLM providers (requires API keys) ──
          await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: transcribing ${imageFiles.length} ${modeLabel} with VLM...`)
          rawNarrations = await generateImageNarrations(imageFiles, (done, total) => {
            void emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: ${done}/${total} ${frameKeyed ? 'panels' : 'images'} transcribed`)
          })
          usedMethod = 'VLM'
        }
      }

      // FILTER CREDIT/AUTHOR/WEBSITE PANELS — these are non-story panels
      // (scanlation credits, Discord links, Patreon, "next chapter" teasers,
      // etc.) that shouldn't be narrated. Their text is set to empty so the
      // Python render step treats them as silent/skipped frames.
      const { filtered: creditFiltered, creditsRemoved } = filterCreditPanels(rawNarrations)
      if (creditsRemoved > 0) {
        await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: filtered out ${creditsRemoved} credit/author panel(s)`)
      }

      // FILTER PUNCTUATION-ONLY PANELS — bubbles that are only "......",
      // "?!!", "!!" etc. (real stylistic "silence beat" bubbles, common in
      // manhwa). OCR reads these correctly, but feeding literal punctuation
      // to edge-tts produces garbage narration, not silence. Silence these
      // instead of narrating them — same shape as the credit filter above,
      // applied after it so it only sees panels the credit filter left alone.
      const { filtered: narrations, junkRemoved } = filterJunkTextPanels(creditFiltered)
      if (junkRemoved > 0) {
        await emitLog(jobId, 'info', 'transcribe', `Chapter ${ch.index}: silenced ${junkRemoved} punctuation-only panel(s) (e.g. "...", "?!!") instead of narrating them literally`)
      }

      // SILENCE THE "[transcription unavailable]" FALLBACK MARKER — the deep
      // end of the transcription fallback chain (OCR -> every VLM provider
      // -> Tesseract, all exhausted for one panel/batch) writes this literal
      // string as a last-resort placeholder so the pipeline can keep moving
      // instead of crashing the whole chapter. It's meant to be an honest
      // "we don't know" marker, NOT real dialogue — nothing downstream
      // treats it specially, so left as-is it would get handed straight to
      // Piper/eSpeak and the narrator would literally say the words
      // "transcription unavailable" out loud over that panel. Same shape as
      // the credit/junk filters above: convert it to silence instead.
      // Also mark these panels status: 'FAILED' (not left absent/derived)
      // so master_pipeline.py's OcrStatus.FAILED branch — which logs and
      // handles a genuine transcription failure differently from a
      // legitimately silent panel — actually gets a status to read.
      // Blanking text to '' alone made every one of these indistinguishable
      // from OcrStatus.NO_TEXT downstream.
      let unavailableSilenced = 0
      for (const n of narrations) {
        if (n.text.trim() === '[transcription unavailable]') {
          n.text = ''
          n.status = 'FAILED'
          unavailableSilenced++
        }
      }
      if (unavailableSilenced > 0) {
        await emitLog(jobId, 'warn', 'transcribe', `Chapter ${ch.index}: ${unavailableSilenced} panel(s) exhausted every transcription method (OCR, all VLM providers, Tesseract) — silenced rather than narrating "transcription unavailable" literally`)
      }

      // Save narrations as narration.json. When frameKeyed, keys are sliced
      // frame filenames (frame_00000.jpg) which the Python render step
      // matches to individual frames for per-panel narration sync.
      await fs.writeFile(narrationFile, JSON.stringify(narrations, null, 2), 'utf8')
      // Per-image transcriptions (narration.json) are sufficient.

      // Distinct from the mid-flow emptyRatio>0.8 check above (which
      // correctly does NOT treat a mostly-silent chapter as failure — many
      // manhwa chapters legitimately are). This is the much stronger
      // signal of EVERY SINGLE panel coming back empty across a whole
      // chapter — real manhwa chapters essentially always have at least
      // some dialogue somewhere. Don't change status/DB fields (risks
      // downstream code that pattern-matches specific status strings) —
      // just make sure this isn't silently indistinguishable from a
      // chapter with real, correct narration.
      if (narrations.length > 3 && narrations.every((n) => !n.text.trim())) {
        await emitLog(jobId, 'warn', 'transcribe',
          `Chapter ${ch.index}: ALL ${narrations.length} panels came back with empty narration text — this is unusual for a full chapter and likely indicates a transcription problem, not a legitimately silent chapter`)
      }

      const updated = await db.chapter.update({
        where: { id: ch.id },
        data: { status: 'transcribed', transcribed: true },
      })
      transcribedCount++
      await emitChapter(jobId, updated)
      await emitLog(
        jobId,
        'success',
        'transcribe',
        `Chapter ${ch.index} transcribed (${usedMethod}): ${narrations.length} ${frameKeyed ? 'panels' : 'images'}${creditsRemoved > 0 ? ` (${creditsRemoved} credits filtered)` : ''}`
      )
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      await emitLog(jobId, 'error', 'transcribe', `Chapter ${ch.index} transcription failed: ${msg}`)
      // Total failure (both PaddleOCR and VLM threw) — previously this still
      // marked the chapter `transcribed: true` / status 'transcribed', which
      // is misleading: the chapter has no narration.json at all, and the
      // Python render step will silently produce a fully-silent chapter for
      // it. Record it as a genuine error instead so the UI and job status
      // reflect what actually happened, using the `status`/`error` fields
      // that already exist in the schema for exactly this.
      const updated = await db.chapter.update({
        where: { id: ch.id },
        data: { status: 'error', error: msg.slice(0, 1000), transcribed: false },
      })
      await emitChapter(jobId, updated)
    }

    await emitProgress(jobId, {
      progress: 30 + (transcribedCount / Math.max(1, scrapedChapters.length)) * 15,
      doneChapters: transcribedCount,
      totalChapters: chapters.length,
      doneImages,
      totalImages,
      stage: 'transcribe',
      message: `Transcribing chapter ${ch.index}/${scrapedChapters.length}`,
    })
  }

  await db.job.update({
    where: { id: jobId },
    data: { doneChapters: transcribedCount },
  })
  await emitLog(
    jobId,
    'success',
    'transcribe',
    `Transcription complete: ${transcribedCount}/${scrapedChapters.length} chapters`,
  )

  if (cancelledJobs.has(jobId)) return

  // -----------------------------
  // Phase 3: RENDER (Python pipeline)
  // -----------------------------
  await db.job.update({
    where: { id: jobId },
    data: { status: 'rendering', stage: 'render', message: 'Running master_pipeline.py' },
  })
  await emitStatus(jobId)
  await emitLog(jobId, 'info', 'render', 'Spawning Python master_pipeline.py')

  const outFile = outputVideoPath(jobId)
  const progressFile = progressFilePath(jobId)
  // Reset the progress file so we don't read stale data.
  try {
    await fs.unlink(progressFile)
  } catch {
    // ignore
  }

  const args: string[] = [
    PIPELINE_SCRIPT,
    '--input-dir', datasetDir(jobId),
    '--output', outFile,
    '--work-dir', workDir(jobId),
    '--voice', job.voice,
    '--narration-provider', 'none',
    '--job-id', jobId,
    ...(process.env.PRODUCTION_PIPELINE === '0' ? [] : ['--production-mode']),
    '--progress-file', progressFile,
    '--keep-temp',
  ]
  if (job.groqKey) {
    args.push('--groq-api-key', job.groqKey)
  }
  if (job.openaiKey) {
    args.push('--openai-api-key', job.openaiKey)
  }
  if (!job.translate) {
    args.push('--no-translate')
  }
  // BGM: job.bgmPath is just a filename inside data/bgm/ (see
  // src/app/api/bgm/route.ts) — resolve it to an absolute path and only
  // pass --bgm if the file actually exists, so a stale/deleted track name
  // can't crash the whole render (master_pipeline.py also validates
  // --bgm's existence, but failing here logs a clearer, job-specific reason).
  if (job.useBgm !== false && job.bgmPath) {
    const bgmFile = path.join(DATA_DIR, 'bgm', job.bgmPath)
    if (await fileExists(bgmFile)) {
      args.push('--bgm', bgmFile)
    } else {
      await emitLog(jobId, 'warn', 'render', `BGM track "${job.bgmPath}" not found — rendering without background music`)
    }
  }

  // Log the command but redact API keys so they never appear in the UI log.
  const redactedArgs = args.map((a) => {
    if (/^(gsk_|sk-)/.test(a)) return a.slice(0, 8) + '***REDACTED***'
    return a
  })
  await emitLog(jobId, 'info', 'render', `CMD: ${PYTHON_BIN} ${redactedArgs.join(' ')}`)

  // Ensure the cwd exists before spawning — prevents uv_cwd ENOENT crashes
  // if the job directory was cleaned up or not yet created.
  const spawnCwd = jobDir(jobId)
  try {
    await fs.mkdir(spawnCwd, { recursive: true })
  } catch {
    // fall back to the repo root if the job dir can't be created
  }

  const child = spawn(PYTHON_BIN, args, {
    cwd: spawnCwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  })
  childProcesses.set(jobId, child)

  // Poll the progress file every 1s while the process runs.
  const pollTimer = setInterval(async () => {
    try {
      const raw = await fs.readFile(progressFile, 'utf8')
      const prog = JSON.parse(raw) as {
        stage: string
        chapter_index: number
        total_chapters: number
        progress: number
        message: string
        status: string
        updated_at: number
      }
      let pct = 40
      let stage = 'render'
      if (prog.stage === 'slice') {
        pct = 40 + (prog.chapter_index / Math.max(1, prog.total_chapters)) * 5
        stage = 'slice'
      } else if (prog.stage === 'render') {
        pct = 45 + (prog.chapter_index / Math.max(1, prog.total_chapters)) * 50
        stage = 'render'
      } else if (prog.stage === 'merge') {
        pct = 95
        stage = 'merge'
      } else if (prog.stage === 'bgm') {
        pct = 97
        stage = 'bgm'
      } else if (prog.stage === 'done') {
        pct = 100
        stage = 'done'
      }
      await emitProgress(jobId, {
        progress: pct,
        doneChapters: transcribedCount,
        totalChapters: chapters.length,
        doneImages,
        totalImages,
        stage,
        message: prog.message || `${stage} phase`,
      })
    } catch {
      // progress file not yet written — ignore
    }
  }, 1000)

  // Stream stdout/stderr line-by-line into JobLog and collect ring buffer of stderr lines.
  const lineBuffers: { stdout: string; stderr: string } = { stdout: '', stderr: '' }
  const stderrHistory: string[] = []

  child.stdout?.on('data', (chunk: Buffer) => {
    lineBuffers.stdout += chunk.toString('utf8')
    let idx: number
    while ((idx = lineBuffers.stdout.indexOf('\n')) >= 0) {
      const line = lineBuffers.stdout.slice(0, idx).trim()
      lineBuffers.stdout = lineBuffers.stdout.slice(idx + 1)
      if (line) void emitLog(jobId, 'info', 'render', line)
    }
  })
  child.stderr?.on('data', (chunk: Buffer) => {
    lineBuffers.stderr += chunk.toString('utf8')
    let idx: number
    while ((idx = lineBuffers.stderr.indexOf('\n')) >= 0) {
      const line = lineBuffers.stderr.slice(0, idx).trim()
      lineBuffers.stderr = lineBuffers.stderr.slice(idx + 1)
      if (line) {
        stderrHistory.push(line)
        if (stderrHistory.length > 100) stderrHistory.shift()
        void emitLog(jobId, 'warn', 'render', line)
      }
    }
  })

  const exitCode: number = await new Promise((resolve) => {
    child.on('exit', (code) => resolve(code ?? -1))
    child.on('error', (err) => {
      void emitLog(jobId, 'error', 'render', `spawn error: ${err.message}`)
      resolve(-1)
    })
  })

  clearInterval(pollTimer)
  childProcesses.delete(jobId)

  // Flush any remaining buffered output.
  if (lineBuffers.stdout.trim()) {
    await emitLog(jobId, 'info', 'render', lineBuffers.stdout.trim())
  }
  if (lineBuffers.stderr.trim()) {
    const remainingLine = lineBuffers.stderr.trim()
    stderrHistory.push(remainingLine)
    if (stderrHistory.length > 100) stderrHistory.shift()
    await emitLog(jobId, 'warn', 'render', remainingLine)
  }

  if (cancelledJobs.has(jobId)) {
    await emitLog(jobId, 'warn', 'render', 'Cancelled during render')
    return
  }

  if (exitCode === 0) {
    // Verify the output file actually exists before marking as done.
    const outputExists = await fileExists(outFile)
    if (!outputExists) {
      const msg = `Pipeline exited 0 but output file is missing: ${outFile}`
      await emitLog(jobId, 'error', 'render', msg)
      await db.job.update({
        where: { id: jobId },
        data: { status: 'error', error: msg, stage: 'render' },
      })
      await emitStatus(jobId)
      io.to(`job:${jobId}`).emit('error', { type: 'error', jobId, error: msg })
      return
    }
    const qa = await validateFinalVideoArtifact(outFile)
    if (!qa.ok) {
      const msg = `Rendered artifact failed video QA: ${qa.error}`
      await emitLog(jobId, 'error', 'render', msg)
      await db.job.update({
        where: { id: jobId },
        data: { status: 'error', error: msg, stage: 'render' },
      })
      await emitStatus(jobId)
      io.to(`job:${jobId}`).emit('error', { type: 'error', jobId, error: msg })
      return
    }
    await emitLog(jobId, 'success', 'render', `Video QA passed (${Math.round(qa.sizeBytes / 1024 / 1024)}MB, ${Math.round(qa.durationSec)}s, audio+video streams present)`)

    const outName = path.basename(outFile)

    // -----------------------------
    // YOUTUBE OPTIMIZATION — generate thumbnail + YouTube-ready encode + metadata.
    // Runs BEFORE R2/Mega archiving because youtube_optimize.py needs the local file.
    // -----------------------------
    try {
      await emitLog(jobId, 'info', 'done', 'Generating YouTube thumbnail + optimized encode...')
      const ytScript = path.join(PROJECT_ROOT, 'pipeline', 'youtube_optimize.py')
      const ytOutputDir = path.join(outputDir(jobId), 'youtube')
      const ytResult = spawnSync(PYTHON_BIN, [
        ytScript,
        '--video', outFile,
        '--title', job.mangaTitle,
        '--cover', job.coverUrl || '',
        '--chapters', String(job.totalChapters),
        '--images', String(job.totalImages),
        '--output-dir', ytOutputDir,
      ], {
        encoding: 'utf8',
        timeout: 600000, // 10 min max for re-encode
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      })

      if (ytResult.status === 0) {
        await emitLog(jobId, 'success', 'done', 'YouTube-ready video + thumbnail + metadata generated')
        const ytLog = (ytResult.stdout || '').split('\n').filter(l => l.includes('[YT]')).slice(-6)
        for (const line of ytLog) {
          await emitLog(jobId, 'info', 'done', line.replace('[YT] ', ''))
        }
      } else {
        await emitLog(jobId, 'warn', 'done', `YouTube optimization failed (non-fatal): ${(ytResult.stderr || '').slice(-200)}`)
      }
    } catch (ytErr) {
      await emitLog(jobId, 'warn', 'done', `YouTube optimization error (non-fatal): ${ytErr instanceof Error ? ytErr.message : ytErr}`)
    }

    // -----------------------------
    // Reclaim disk space: intermediate work/ (sliced frames, per-panel
    // audio, per-chapter renders) and dataset/ (raw scraped chapter images)
    // are never needed again once the final video exists. For a large job
    // (e.g. 200 chapters) these dwarf the final compressed video, so this
    // alone is the biggest single win for disk usage.
    // -----------------------------
    for (const dir of [workDir(jobId), datasetDir(jobId)]) {
      try {
        await fs.rm(dir, { recursive: true, force: true })
      } catch (e) {
        await emitLog(jobId, 'warn', 'done', `Could not clean up ${dir}: ${e instanceof Error ? e.message : e}`)
      }
    }
    await emitLog(jobId, 'info', 'done', 'Cleaned up intermediate work/dataset files')

    // -----------------------------
    // Offload the final video to Cloudflare R2, then free the local copy
    // too, so completed jobs don't keep accumulating local disk usage.
    // Only deletes the local file after the upload is verified to have
    // landed — if R2 isn't configured, or the upload fails, the local
    // file is left in place so the job is never left without a copy.
    // -----------------------------
    let r2Key: string | null = null
    if (isR2Configured()) {
      const key = `jobs/${jobId}/${outName}`
      try {
        await uploadFileToR2(outFile, key)
        await fs.rm(outFile, { force: true })
        r2Key = key
        await emitLog(jobId, 'success', 'done', `Uploaded output to R2 (${key}) and freed local copy`)
      } catch (e) {
        await emitLog(
          jobId,
          'warn',
          'done',
          `R2 upload failed, keeping local copy: ${e instanceof Error ? e.message : e}`,
        )
      }
    }

    // -----------------------------
    // Cloud archive: Mega (20 GB free tier).
    // Only runs if R2 didn't already handle the file (i.e. local file still
    // exists). Uploads to Mega, stores the share URL (with decryption key) in
    // the DB, then deletes the local file to free disk space.
    // Uses per-job Mega creds if provided (from the UI), else falls back to
    // .env MEGA_EMAIL/MEGA_PASSWORD.
    // -----------------------------
    let archiveProvider: string | null = null
    let archiveFileId: string | null = null
    // Per-job autoArchive flag takes priority; fallback to env AUTO_ARCHIVE
    const autoArchive = job.autoArchive === true || (job.autoArchive === null && process.env.AUTO_ARCHIVE !== 'false')
    // Per-job Mega creds take priority; fallback to env
    const megaEmail = job.megaEmail || process.env.MEGA_EMAIL
    const megaPassword = job.megaPassword || process.env.MEGA_PASSWORD
    if (autoArchive && r2Key === null) {
      const localExists = await fileExists(outFile)
      const megaOk = !!megaEmail && !!megaPassword

      if (localExists && megaOk) {
        await emitLog(jobId, 'info', 'done', 'Archiving video to Mega…')
        try {
          const safeTitle = (job.mangaTitle || 'recap').replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '')
          const archiveName = `${safeTitle}_recap.mp4`

          await new Promise<void>((resolve, reject) => {
            const s = new MegaStorage({
              email: megaEmail,
              password: megaPassword,
              autoload: true,
            })
            s.on('ready', () => {
              try {
                const uploadStream = s.upload(archiveName)
                const source = createReadStream(outFile)
                source.on('error', (err) => reject(err))

                // megajs ships Deno-oriented stream declarations, while at
                // runtime this is a Node writable stream. Keep that mismatch
                // isolated at the library boundary instead of weakening the
                // rest of the upload path.
                source.pipe(uploadStream as unknown as NodeJS.WritableStream)

                uploadStream.complete
                  .then(async (file) => {
                    archiveProvider = 'mega'
                    archiveFileId = await file.link(false)
                    resolve()
                  })
                  .catch((err: unknown) => reject(err))
              } catch (err) {
                reject(err)
              }
            })
            ;(s as unknown as { on(event: 'error', listener: (err: Error) => void): void }).on('error', (err) => reject(err))
          })

          // Delete local file after successful upload.
          if (archiveFileId) {
            await fs.rm(outFile, { force: true })
            await emitLog(jobId, 'success', 'done', `Uploaded to Mega and freed local copy`)
          }
        } catch (e) {
          await emitLog(
            jobId,
            'warn',
            'done',
            `Mega archive failed, keeping local copy: ${e instanceof Error ? e.message : e}`,
          )
        }
      }
    }

    await db.job.update({
      where: { id: jobId },
      data: {
        status: 'done',
        progress: 100,
        stage: 'done',
        message: 'Pipeline complete',
        outputDir: outputDir(jobId),
        outputVideo: outName,
        r2Key,
        archiveProvider,
        archiveFileId,
      },
    })
    await emitLog(jobId, 'success', 'done', `Pipeline complete. Output: ${outName}`)

    io.to(`job:${jobId}`).emit('done', { type: 'done', jobId, outputVideo: outName })
    await emitStatus(jobId)
    console.log(`[job:${jobId}] done`)
  } else {
    const last50Stderr = stderrHistory.slice(-50).join('\n')
    const err = `master_pipeline.py exited with code ${exitCode}:\n${last50Stderr || '(no stderr output captured)'}`
    await emitLog(jobId, 'error', 'render', `Python process failure details:\n${last50Stderr || '(no stderr output captured)'}`)
    await db.job.update({
      where: { id: jobId },
      data: { status: 'error', stage: 'render', error: err.slice(0, 2000), message: `master_pipeline.py exited with code ${exitCode}` },
    })
    io.to(`job:${jobId}`).emit('error', { type: 'error', jobId, error: err })
    await emitStatus(jobId)
    console.error(`[job:${jobId}] failed: ${err}`)
  }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

// H39 FIX: Limit request body to 1MB to prevent memory exhaustion.
const MAX_BODY_SIZE = 1_024_000 // 1MB
async function readBody(req: IncomingMessage): Promise<any> {
  return await new Promise((resolve) => {
    const chunks: Buffer[] = []
    let totalSize = 0
    req.on('data', (c) => {
      totalSize += c.length
      if (totalSize > MAX_BODY_SIZE) {
        req.destroy()
        resolve({})
        return
      }
      chunks.push(c as Buffer)
    })
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8')
      if (!raw) {
        resolve({})
        return
      }
      try {
        resolve(JSON.parse(raw))
      } catch {
        resolve({})
      }
    })
    req.on('error', () => resolve({}))
  })
}

function sendJson(res: ServerResponse, code: number, body: unknown) {
  const json = JSON.stringify(body)
  res.writeHead(code, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(json),
  })
  res.end(json)
}

// ---------------------------------------------------------------------------
// Boot + graceful shutdown
// ---------------------------------------------------------------------------

httpServer.listen(PORT, async () => {
  console.log(`[pipeline-service] listening on port ${PORT} (socket.io path "/")`)

  // PYTHON DEPENDENCY CHECK: production mode is local-first and must not require
  // edge-tts/openai or install packages at runtime. Legacy/dev mode can still
  // use pipeline/requirements.txt if explicitly selected with PRODUCTION_PIPELINE=0.
  try {
    const productionPipeline = process.env.PRODUCTION_PIPELINE !== '0'
    const depSnippet = productionPipeline ? 'import PIL, cv2, numpy' : 'import edge_tts, openai, PIL, cv2, numpy'
    const checkResult = spawnSync(PYTHON_BIN, ['-c', depSnippet], {
      encoding: 'utf8',
      timeout: 10000,
    })
    if (checkResult.status !== 0) {
      if (productionPipeline) {
        console.error('[pipeline-service] Production Python deps missing; not auto-installing at runtime')
        return
      }
      console.log('[pipeline-service] Legacy Python deps missing — auto-installing from pipeline/requirements.txt')
      const installResult = spawnSync(PYTHON_BIN, ['-m', 'pip', 'install', '-r', path.join(PROJECT_ROOT, 'pipeline', 'requirements.txt')], {
        encoding: 'utf8',
        timeout: 120000,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      })
      if (installResult.status === 0) {
        console.log('[pipeline-service] Python deps installed successfully')
      } else {
        console.error('[pipeline-service] Failed to install Python deps:', installResult.stderr?.slice(-300))
      }
    } else {
      console.log(`[pipeline-service] Python deps OK (${depSnippet})`)
    }
  } catch (err) {
    console.error('[pipeline-service] Python dep check failed:', err)
  }

  // AUTO-REQUEUE: find any jobs stuck in "pending" status (created while the
  // pipeline-service was down) and enqueue them for processing. Without this,
  // a job created while the service was restarting would stay at "pending"
  // forever — appearing as "stuck" in the UI.
  // Also requeue jobs stuck in active states (scraping/rendering/etc.) that
  // were interrupted when the service crashed/restarted. These are reset to
  // "pending" so the pipeline picks them up fresh.
  try {
    // First, reset stuck active jobs to "pending" so they get re-processed.
    const STUCK_STATUSES = ['scraping', 'transcribing', 'translating', 'rendering', 'merging']
    const stuckJobs = await db.job.findMany({
      where: { status: { in: STUCK_STATUSES } },
      orderBy: { createdAt: 'asc' },
      select: { id: true, status: true },
    })
    if (stuckJobs.length > 0) {
      console.log(`[pipeline-service] found ${stuckJobs.length} stuck job(s) (interrupted mid-processing) — resetting to pending`)
      for (const j of stuckJobs) {
        await db.job.update({
          where: { id: j.id },
          data: {
            status: 'pending',
            stage: null,
            message: 'Job was interrupted — automatically re-queued on service restart.',
          },
        })
        await db.jobLog.create({
          data: {
            jobId: j.id,
            level: 'warn',
            stage: 'search',
            message: `Job was stuck in '${j.status}' state — reset to pending on service restart.`,
          },
        })
      }
    }

    // Now requeue all pending jobs (including the ones we just reset).
    const pendingJobs = await db.job.findMany({
      where: { status: 'pending' },
      orderBy: { createdAt: 'asc' },
      select: { id: true },
    })
    if (pendingJobs.length > 0) {
      console.log(`[pipeline-service] found ${pendingJobs.length} pending job(s) on startup — re-queuing`)
      for (const j of pendingJobs) {
        enqueueJob(j.id)
      }
    }
  } catch (err) {
    console.error('[pipeline-service] failed to re-queue pending jobs:', err)
  }
})

async function shutdown(signal: string) {
  console.log(`[pipeline-service] received ${signal}, shutting down`)
  // Kill any running subprocesses.
  for (const [jobId, child] of childProcesses.entries()) {
    try {
      child.kill('SIGTERM')
      console.log(`[pipeline-service] killed subprocess for job ${jobId}`)
    } catch {
      // ignore
    }
  }
  childProcesses.clear()
  // Close socket.io + HTTP server.
  io.close()
  httpServer.close(() => {
    console.log('[pipeline-service] http server closed')
    process.exit(0)
  })
  // Force-exit after 5s if close hangs.
  setTimeout(() => process.exit(0), 5000).unref()
}

process.on('SIGTERM', () => void shutdown('SIGTERM'))
process.on('SIGINT', () => void shutdown('SIGINT'))

// Handle uncaught errors so the service doesn't silently die.
// C14 FIX: Handle uncaught errors AND update job status + notify clients.
function handleFatalError(err: unknown, origin: string) {
  console.error(`[pipeline-service] ${origin}:`, err)
  if (currentlyRunning) {
    const jobId = currentlyRunning
    const msg = err instanceof Error ? err.message : String(err)
    db.job.update({
      where: { id: jobId },
      data: { status: 'error', error: msg.slice(0, 2000), stage: 'fatal' },
    }).catch(() => undefined)
    io.to(`job:${jobId}`).emit('error', { type: 'error', jobId, error: msg })
    console.error(`[pipeline-service] Marked job ${jobId} as error due to ${origin}`)
  }
}
process.on('uncaughtException', (err) => handleFatalError(err, 'uncaughtException'))
process.on('unhandledRejection', (err) => handleFatalError(err, 'unhandledRejection'))
process.on('exit', (code, signal) => {
  const m = process.memoryUsage()
  console.error(`[pipeline-service] EXIT code=${code} signal=${signal} rss=${Math.round(m.rss / 1024 / 1024)}MB`)
})
// Log memory usage every 30s to detect leaks.
setInterval(() => {
  const m = process.memoryUsage()
  console.log(`[mem] rss=${Math.round(m.rss/1024/1024)}MB heap=${Math.round(m.heapUsed/1024/1024)}/${Math.round(m.heapTotal/1024/1024)}MB`)
}, 30_000)

// Prune data/cache/restore/ (temp files written by the web app's archive
// restore endpoint when serving an archived video back from Mega — see
// src/lib/archive.ts, whose own TTL/path this mirrors). That directory was
// never being cleaned anywhere, so on a small Oracle free-tier disk it would
// grow unbounded every time an archived video got viewed. Runs here since
// this is the long-lived process; the web app itself is request-scoped.
const RESTORE_CACHE_DIR = path.join(DATA_DIR, 'cache', 'restore')
const RESTORE_CACHE_TTL_MS = 60 * 60 * 1000 // 1 hour, matches src/lib/archive.ts
async function cleanupRestoreCache(): Promise<void> {
  try {
    const files = await fs.readdir(RESTORE_CACHE_DIR)
    let deleted = 0
    for (const f of files) {
      const p = path.join(RESTORE_CACHE_DIR, f)
      try {
        const stat = await fs.stat(p)
        if (Date.now() - stat.mtimeMs > RESTORE_CACHE_TTL_MS) {
          await fs.unlink(p)
          deleted++
        }
      } catch {
        // ignore individual file errors
      }
    }
    if (deleted > 0) console.log(`[pipeline-service] restore cache: pruned ${deleted} stale file(s)`)
  } catch {
    // directory may not exist yet — nothing to prune
  }
}
setInterval(() => void cleanupRestoreCache(), 15 * 60 * 1000) // every 15 min
void cleanupRestoreCache() // also run once on startup
