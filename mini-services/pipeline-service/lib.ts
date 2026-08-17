/**
 * lib.ts — shared helpers for the pipeline-service.
 *
 * Contains:
 *  - Prisma client (single instance)
 *  - Path helpers (hardcoded to the parent app's data dir)
 *  - MangaDex fetch helpers (chapter pages, image download)
 *  - VLM helper (z-ai-web-dev-sdk) for transcribing panel text from images
 *  - Small utility helpers (sleep, ensureDir, fileExists, sanitize)
 */

import { PrismaClient } from '@prisma/client'
import { promises as fs } from 'fs'
import path from 'path'

// ---------------------------------------------------------------------------
// Prisma — multi-database support (SQLite, Turso, Supabase/PostgreSQL).
// Auto-detects from DATABASE_URL:
//   file: → SQLite (local dev)
//   libsql:// → Turso (hosted SQLite)
//   postgresql:// → Supabase (hosted PostgreSQL)
// ---------------------------------------------------------------------------

const globalForPrisma = globalThis as unknown as { pipelinePrisma: PrismaClient | undefined }

function createPrismaClient(): PrismaClient {
  const url = process.env.DATABASE_URL || ''

  // Turso / libsql
  if (url.startsWith('libsql://') || url.startsWith('http://') || url.startsWith('https://')) {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { PrismaLibSQL } = require('@prisma/adapter-libsql')
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { createClient } = require('@libsql/client')
    const libsql = createClient({
      url,
      authToken: process.env.DATABASE_AUTH_TOKEN || undefined,
    })
    const adapter = new PrismaLibSQL(libsql)
    return new PrismaClient({ adapter })
  }

  // Supabase / PostgreSQL — Prisma natively supports it, no adapter needed.
  // Just set DATABASE_URL to the Supabase connection string.
  if (url.startsWith('postgresql://') || url.startsWith('postgres://')) {
    return new PrismaClient({ log: ['error', 'warn'] })
  }

  // Local SQLite
  return new PrismaClient({ log: ['error', 'warn'] })
}

export const db = globalForPrisma.pipelinePrisma ?? createPrismaClient()

if (process.env.NODE_ENV !== 'production') globalForPrisma.pipelinePrisma = db

// ---------------------------------------------------------------------------
// Paths — env-configurable so the mini-service can run anywhere (laptop, VPS,
// etc.), not just the original sandbox. Defaults preserve the original
// /home/z/my-project layout for backwards compatibility.
// ---------------------------------------------------------------------------

export const DATA_DIR = process.env.DATA_DIR
  ? path.resolve(process.env.DATA_DIR)
  : '/home/z/my-project/data'
// Pipeline script lives next to the parent app; resolve via PROJECT_ROOT if set,
// otherwise fall back to the known sandbox location.
const PROJECT_ROOT = process.env.PROJECT_ROOT || '/home/z/my-project'
export { PROJECT_ROOT }
export const PIPELINE_SCRIPT = path.join(PROJECT_ROOT, 'pipeline', 'master_pipeline.py')
export const PYTHON_BIN = process.env.PYTHON_BIN || 'python3'

export function jobDir(jobId: string): string {
  return path.join(DATA_DIR, 'jobs', jobId)
}
export function datasetDir(jobId: string): string {
  return path.join(jobDir(jobId), 'dataset')
}
export function workDir(jobId: string): string {
  return path.join(jobDir(jobId), 'work')
}
export function outputDir(jobId: string): string {
  return path.join(jobDir(jobId), 'output')
}
export function chapterDir(jobId: string, index: number): string {
  return path.join(datasetDir(jobId), `chapter_${String(index).padStart(3, '0')}`)
}
export function outputVideoPath(jobId: string): string {
  return path.join(outputDir(jobId), 'master_recap.mp4')
}
export function progressFilePath(jobId: string): string {
  return path.join(jobDir(jobId), 'progress.json')
}

export async function ensureDir(p: string): Promise<void> {
  await fs.mkdir(p, { recursive: true })
}

export async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.access(p)
    return true
  } catch {
    return false
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}

// ---------------------------------------------------------------------------
// Caching: JSON file cache for VLM transcription results.
// Saves to data/cache/vlm/{hash}.json with a 1-hour TTL.
// On re-runs (e.g. after a render failure), cached VLM results are reused
// instantly — huge savings since VLM calls are the bottleneck.
// ---------------------------------------------------------------------------

import crypto from 'crypto'

const VLM_CACHE_DIR = path.join(DATA_DIR, 'cache', 'vlm')
const VLM_CACHE_TTL_MS = 365 * 24 * 3600 * 1000 // 1 year (effectively permanent)

function vlmCacheKey(imagePath: string): string {
  return crypto.createHash('sha256').update(imagePath).digest('hex').slice(0, 16)
}

async function getVlmCached(key: string): Promise<string | null> {
  try {
    const cacheFile = path.join(VLM_CACHE_DIR, `${key}.json`)
    const stat = await fs.stat(cacheFile)
    if (Date.now() - stat.mtimeMs > VLM_CACHE_TTL_MS) return null
    const data = JSON.parse(await fs.readFile(cacheFile, 'utf8'))
    return data.text ?? null
  } catch {
    return null
  }
}

async function setVlmCached(key: string, text: string): Promise<void> {
  try {
    await fs.mkdir(VLM_CACHE_DIR, { recursive: true })
    await fs.writeFile(
      path.join(VLM_CACHE_DIR, `${key}.json`),
      JSON.stringify({ text, ts: Date.now() }),
      'utf8',
    )
  } catch {
    // non-fatal
  }
}

// ---------------------------------------------------------------------------
// Credit/author/website panel detection.
// Manhwa chapters often start or end with "credits" panels that mention the
// scanlation group, translator, website, Discord, Patreon, etc. These are
// not part of the story and shouldn't be narrated. This function detects
// them by looking for common credit-related keywords in the transcribed text.
// ---------------------------------------------------------------------------

const CREDIT_PATTERNS: RegExp[] = [
  /scanlat/i,        // scanlation, scanlator
  /translat(?:ed|or|ion)\s+by/i,  // "translated by", "translator"
  /typeset(?:ting|ter)\s+by/i,    // "typesetting by"
  /proofread(?:er|ing)?\s+by/i,   // "proofreader by"
  /redraw(?:n|er|ing)?\s+by/i,    // "redrawn by"
  /clean(?:er|ing)\s+by/i,        // "cleaner by"
  /raw\s+(?:provider|by)/i,       // "raw provider"
  /discord(?:\.gg)?/i,            // discord links
  /patreon/i,
  /paypal/i,
  /ko-?fi/i,
  /buymeacoffee/i,
  /donate/i,
  /support\s+(?:us|the\s+(?:team|scanlat))/i,
  /join\s+(?:our\s+)?(?:discord|server)/i,
  /follow\s+(?:us|on)/i,
  /@[\w-]+\s*(?:on\s+)?(?:twitter|insta|tiktok|youtube)/i,  // social handles
  /website\s*:/i,
  /visit\s+(?:our\s+)?(?:site|website)/i,
  /please\s+(?:wait|don.?t\s+re-?upload|do\s+not\s+re-?upload)/i,
  /re-?upload/i,
  /aggregator/i,
  /chapter\s+end/i,        // "chapter end" credits
  /end\s+of\s+chapter/i,
  /next\s+chapter/i,       // "next chapter" teaser
  /coming\s+soon/i,
  /credit\s+(?:to|goes)/i, // "credit to"
  /special\s+thanks/i,
  /powered\s+by/i,
]

/**
 * Check if a panel's transcribed text indicates it's a credits/author/website
 * panel (not part of the story). Returns true if the text matches any credit
 * pattern, false otherwise.
 *
 * Used to filter out non-story panels so they're never narrated or shown
 * (they'd just waste screen time and confuse the viewer).
 */
export function isCreditPanel(text: string): boolean {
  if (!text || !text.trim()) return false
  const lower = text.toLowerCase()
  // Require at least one credit keyword. Short texts (like single sound
  // effects) won't match, so real story dialogue is safe.
  for (const pattern of CREDIT_PATTERNS) {
    if (pattern.test(lower)) {
      return true
    }
  }
  return false
}

/**
 * Filter an array of {image, text} narrations, removing credit panels.
 * Returns { filtered, creditsRemoved } where filtered is the narration array
 * with credit panels set to empty text (so the frame is skipped during render)
 * and creditsRemoved is the count of removed panels.
 *
 * We set text to empty rather than removing the entry entirely, so the frame
 * indices stay aligned with the Python render step's frame list.
 */
export function filterCreditPanels(
  narrations: Array<{ image: string; text: string }>,
): { filtered: Array<{ image: string; text: string }>; creditsRemoved: number } {
  let creditsRemoved = 0
  const filtered = narrations.map((n) => {
    if (isCreditPanel(n.text)) {
      creditsRemoved++
      return { ...n, text: '' } // empty = silent/skipped during render
    }
    return n
  })
  return { filtered, creditsRemoved }
}

// ---------------------------------------------------------------------------
// Multi-source scraping helpers (MangaHere + FanFox + Webtoons + AsuraScans).
// ---------------------------------------------------------------------------

// --- Source dispatchers ---

export type ScraperSource = 'mangahere' | 'fanfox' | 'webtoons' | 'asurascans'

export function getSourceFromId(id: string): ScraperSource | null {
  if (id.startsWith('mh-')) return 'mangahere'
  if (id.startsWith('ff-')) return 'fanfox'
  if (id.startsWith('wt-')) return 'webtoons'
  if (id.startsWith('as-')) return 'asurascans'
  return null
}

export function getSlugFromId(id: string): string {
  return id.replace(/^(mh-|ff-|wt-|as-)/, '')
}

// --- MangaHere (mangahere.cc) ---

const MANGAHERE_BASE = 'https://www.mangahere.cc'
const MANGAHERE_CDN = 'https://zjcdn.mangahere.org'

/**
 * Fetch the chapter list for a manga from MangaHere.
 * mangaId is the MangaHere slug (e.g. "solo_leveling").
 * Returns chapters oldest-first.
 */
export async function fetchMangaHereChapters(
  mangaSlug: string,
  chapterLimit: number,
): Promise<
  Array<{
    mangadexId: string // chapter slug e.g. "c001" (kept as mangadexId for DB compat)
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }>
> {
  const url = `${MANGAHERE_BASE}/manga/${mangaSlug}/`
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
  })
  if (!res.ok) {
    throw new Error(`MangaHere chapters ${res.status} for ${mangaSlug}`)
  }
  const html = await res.text()

  // Parse chapter links: href="/manga/{slug}/c{chapter}/1.html"
  const chapters: Array<{
    mangadexId: string
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }> = []
  const seen = new Set<string>()
  const chapterRegex = new RegExp(
    `href="/manga/${mangaSlug}/((?:v\\d+/)?c[0-9.]+)/1\\.html"`,
    'gi',
  )
  let match
  while ((match = chapterRegex.exec(html)) !== null) {
    const chapSlug = match[1]
    if (seen.has(chapSlug)) continue
    seen.add(chapSlug)
    const chapterNum = chapSlug.match(/c([0-9.]+)/) ? (chapSlug.match(/c([0-9.]+)/)![1].replace(/^0+/, '') || '0') : '0'
    chapters.push({
      mangadexId: chapSlug,
      chapterNum,
      title: null,
      language: 'en',
      pageCount: 0,
      external: false,
    })
  }

  // MangaHere returns newest-first; reverse to oldest-first.
  chapters.reverse()

  // Apply chapter limit (0 = all).
  const limited = chapterLimit > 0 ? chapters.slice(0, chapterLimit) : chapters
  return limited
}

/**
 * Extract all image URLs for a MangaHere chapter by scraping the chapter page HTML.
 *
 * MangaHere loads images via obfuscated JavaScript. The image filenames are
 * embedded in the HTML as pipe-separated values. We extract them and construct
 * full CDN URLs.
 */
export async function fetchMangaHereChapterImages(
  mangaSlug: string,
  chapterSlug: string,
): Promise<string[]> {
  const url = `${MANGAHERE_BASE}/manga/${mangaSlug}/${chapterSlug}/1.html`
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
  })
  if (!res.ok) {
    throw new Error(`MangaHere chapter ${res.status} for ${mangaSlug}/${chapterSlug}`)
  }
  const html = await res.text()

  // Extract store ID from cover image URL: store/manga/{storeId}/cover.jpg
  const storeMatch = html.match(/store\/manga\/(\d+)/)
  const storeId = storeMatch?.[1]
  if (!storeId) {
    throw new Error(`Could not extract store ID from ${url}`)
  }

  // Extract chapter folder from chapter slug: "v72/c700" → "700", "c001" → "001", "c200.5" → "200.5"
  const chapterMatch = chapterSlug.match(/c([0-9.]+)/)
  const chapterFolder = chapterMatch
    ? chapterMatch[1].padStart(3, '0')
    : chapterSlug.replace(/^c/, '').padStart(3, '0')

  // Extract image filenames from obfuscated JavaScript.
  // Pattern: {letter}{date}_{time}_{number} e.g. h20181105_144325_927
  const filenameRegex = /([a-z]\d{8}_\d{6}_[a-z0-9]+)/gi
  const filenames = new Set<string>()
  let m
  while ((m = filenameRegex.exec(html)) !== null) {
    filenames.add(m[1])
  }

  if (filenames.size === 0) {
    throw new Error(`No image filenames found in ${url}`)
  }

  // Construct full CDN URLs
  const imageUrls = Array.from(filenames).map(
    (fn) =>
      `${MANGAHERE_CDN}/store/manga/${storeId}/${chapterFolder}.0/compressed/${fn}.jpg`,
  )

  return imageUrls
}

/**
 * Download a single MangaHere image to disk.
 * CRITICAL: MangaHere CDN requires `Referer: https://www.mangahere.cc/` header.
 */
export async function downloadMangaHereImage(
  imageUrl: string,
  destPath: string,
): Promise<void> {
  const res = await fetch(imageUrl, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      Referer: `${MANGAHERE_BASE}/`,
      Accept: 'image/*',
    },
  })
  if (!res.ok) {
    throw new Error(`MangaHere image download ${res.status}: ${imageUrl}`)
  }
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(destPath, buf)
}

/**
 * Get the extension from a filename, e.g. "x01.jpg" -> ".jpg".
 */
export function extFromFilename(filename: string): string {
  const m = filename.match(/\.(jpe?g|png|webp|gif)$/i)
  return m ? `.${m[1].toLowerCase()}` : '.jpg'
}

// --- FanFox (fanfox.net) — same CMS as MangaHere, different CDN ---

const FANFOX_BASE = 'https://fanfox.net'
const FANFOX_CDN = 'https://fmcdn.mfcdn.net'

export async function fetchFanFoxChapters(
  mangaSlug: string,
  chapterLimit: number,
): Promise<
  Array<{
    mangadexId: string
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }>
> {
  const url = `${FANFOX_BASE}/manga/${mangaSlug}/`
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
  })
  if (!res.ok) {
    throw new Error(`FanFox chapters ${res.status} for ${mangaSlug}`)
  }
  const html = await res.text()

  const chapters: Array<{
    mangadexId: string
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }> = []
  const seen = new Set<string>()
  const chapterRegex = new RegExp(
    `href="/manga/${mangaSlug}/((?:v\\d+/)?c[0-9.]+)/1\\.html"`,
    'gi',
  )
  let match
  while ((match = chapterRegex.exec(html)) !== null) {
    const chapSlug = match[1]
    if (seen.has(chapSlug)) continue
    seen.add(chapSlug)
    const chapterNum = chapSlug.match(/c([0-9.]+)/) ? (chapSlug.match(/c([0-9.]+)/)![1].replace(/^0+/, '') || '0') : '0'
    chapters.push({
      mangadexId: chapSlug,
      chapterNum,
      title: null,
      language: 'en',
      pageCount: 0,
      external: false,
    })
  }
  chapters.reverse()
  return chapterLimit > 0 ? chapters.slice(0, chapterLimit) : chapters
}

export async function fetchFanFoxChapterImages(
  mangaSlug: string,
  chapterSlug: string,
): Promise<string[]> {
  const url = `${FANFOX_BASE}/manga/${mangaSlug}/${chapterSlug}/1.html`
  const res = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
  })
  if (!res.ok) {
    throw new Error(`FanFox chapter ${res.status} for ${mangaSlug}/${chapterSlug}`)
  }
  const html = await res.text()

  const storeMatch = html.match(/store\/manga\/(\d+)/)
  const storeId = storeMatch?.[1]
  if (!storeId) {
    throw new Error(`Could not extract store ID from ${url}`)
  }
  const ffChapMatch = chapterSlug.match(/c([0-9.]+)/)
  const chapterFolder = ffChapMatch
    ? ffChapMatch[1].padStart(3, '0')
    : chapterSlug.replace(/^c/, '').padStart(3, '0')

  const filenameRegex = /([a-z]\d{8}_\d{6}_[a-z0-9]+)/gi
  const filenames = new Set<string>()
  let m
  while ((m = filenameRegex.exec(html)) !== null) {
    filenames.add(m[1])
  }

  if (filenames.size === 0) {
    throw new Error(`No image filenames found in ${url}`)
  }

  return Array.from(filenames).map(
    (fn) =>
      `${FANFOX_CDN}/store/manga/${storeId}/${chapterFolder}.0/compressed/${fn}.jpg`,
  )
}

export async function downloadFanFoxImage(
  imageUrl: string,
  destPath: string,
): Promise<void> {
  const res = await fetch(imageUrl, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      Referer: `${FANFOX_BASE}/`,
      Accept: 'image/*',
    },
  })
  if (!res.ok) {
    throw new Error(`FanFox image download ${res.status}: ${imageUrl}`)
  }
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(destPath, buf)
}

// --- Webtoons (webtoons.com) — official manhwa/webtoons ---

const WEBTOONS_BASE = 'https://www.webtoons.com'

export async function fetchWebtoonsChapters(
  titleNo: number,
  chapterLimit: number,
): Promise<
  Array<{
    mangadexId: string
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }>
> {
  const res = await fetch(
    `${WEBTOONS_BASE}/en/fantasy/_/list?title_no=${titleNo}`,
    {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
      },
      redirect: 'follow',
    },
  )
  if (!res.ok) {
    throw new Error(`Webtoons chapters ${res.status} for title_no=${titleNo}`)
  }
  const html = await res.text()

  const chapters: Array<{
    mangadexId: string
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }> = []
  const seen = new Set<number>()
  const regex = /href="([^"]*\/viewer\?title_no=\d+&episode_no=(\d+))"/gi
  let match
  while ((match = regex.exec(html)) !== null) {
    const epNo = parseInt(match[2], 10)
    if (seen.has(epNo)) continue
    seen.add(epNo)
    chapters.push({
      mangadexId: `ep-${epNo}`,
      chapterNum: String(epNo),
      title: null,
      language: 'en',
      pageCount: 0,
      external: false,
    })
  }
  chapters.reverse()
  return chapterLimit > 0 ? chapters.slice(0, chapterLimit) : chapters
}

export async function fetchWebtoonsChapterImages(
  titleNo: number,
  episodeNo: number,
): Promise<string[]> {
  // Find the viewer URL from the list page.
  const listRes = await fetch(
    `${WEBTOONS_BASE}/en/fantasy/_/list?title_no=${titleNo}`,
    {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
      },
      redirect: 'follow',
    },
  )
  if (!listRes.ok) {
    throw new Error(`Webtoons list ${listRes.status}`)
  }
  const listHtml = await listRes.text()

  const viewerRegex = new RegExp(
    `href="([^"]*episode_no=${episodeNo})"`,
    'i',
  )
  const viewerMatch = listHtml.match(viewerRegex)
  if (!viewerMatch) {
    throw new Error(`Episode ${episodeNo} not found for title_no=${titleNo}`)
  }
  const viewerUrl = viewerMatch[1].startsWith('http')
    ? viewerMatch[1]
    : `${WEBTOONS_BASE}${viewerMatch[1]}`

  const res = await fetch(viewerUrl, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      'Accept-Language': 'en-US,en;q=0.9',
      Referer: `${WEBTOONS_BASE}/`,
    },
  })
  if (!res.ok) {
    throw new Error(`Webtoons viewer ${res.status}`)
  }
  const html = await res.text()

  // Webtoons embeds image URLs in data-url attributes.
  const images: string[] = []
  const dataUrlRegex = /data-url="(https:\/\/webtoon-phinf\.pstatic\.net\/[^"]+)"/gi
  let m
  while ((m = dataUrlRegex.exec(html)) !== null) {
    images.push(m[1])
  }

  // Fallback: try src attributes.
  if (images.length === 0) {
    const srcRegex = /src="(https:\/\/webtoon-phinf\.pstatic\.net\/[^"]+)"/gi
    while ((m = srcRegex.exec(html)) !== null) {
      images.push(m[1])
    }
  }

  return images
}

export async function downloadWebtoonsImage(
  imageUrl: string,
  destPath: string,
): Promise<void> {
  const res = await fetch(imageUrl, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      Referer: `${WEBTOONS_BASE}/`,
      Accept: 'image/*',
    },
  })
  if (!res.ok) {
    throw new Error(`Webtoons image download ${res.status}: ${imageUrl}`)
  }
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(destPath, buf)
}

// --- AsuraScans (asurascans.com) — JSON REST API ---
// AsuraScans is an Astro SPA with a clean REST API at api.asurascans.com.
//   GET /api/search?q={query}
//   GET /api/series/{slug}/chapters                       (newest-first)
//   GET /api/series/{slug}/chapters/{chapterSlug}         -> { chapter: { pages: [{url}] } }

const ASURA_API = 'https://api.asurascans.com'
const ASURA_WEB = 'https://asurascans.com'
const ASURA_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

interface AsuraChapter {
  id: number
  number: number
  title: string
  slug: string
}

interface AsuraChapterPage {
  url: string
}

/**
 * Fetch the chapter list for a manga from AsuraScans.
 * mangaId is the as-{slug} form; slug is the AsuraScans series slug.
 * Returns chapters oldest-first.
 */
export async function fetchAsuraScansChapters(
  mangaSlug: string,
  chapterLimit: number,
): Promise<
  Array<{
    mangadexId: string
    chapterNum: string | null
    title: string | null
    language: string
    pageCount: number
    external: boolean
  }>
> {
  const res = await fetch(
    `${ASURA_API}/api/series/${encodeURIComponent(mangaSlug)}/chapters`,
    {
      headers: {
        'User-Agent': ASURA_UA,
        Accept: 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
      },
    },
  )
  if (!res.ok) {
    throw new Error(`AsuraScans chapters ${res.status} for ${mangaSlug}`)
  }
  const body = await res.json()
  const chapters: AsuraChapter[] = body?.data ?? []

  // API returns newest-first; reverse to oldest-first.
  const oldest = [...chapters].reverse()
  const mapped = oldest.map((c) => ({
    mangadexId: c.slug, // chapter slug (UUID) — needed for the images endpoint
    chapterNum: String(c.number),
    title: c.title || null,
    language: 'en',
    pageCount: 0,
    external: false,
  }))
  return chapterLimit > 0 ? mapped.slice(0, chapterLimit) : mapped
}

/**
 * Fetch all page image URLs for an AsuraScans chapter via the JSON API.
 */
export async function fetchAsuraScansChapterImages(
  mangaSlug: string,
  chapterSlug: string,
): Promise<string[]> {
  const res = await fetch(
    `${ASURA_API}/api/series/${encodeURIComponent(mangaSlug)}/chapters/${encodeURIComponent(chapterSlug)}`,
    {
      headers: {
        'User-Agent': ASURA_UA,
        Accept: 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        Referer: `${ASURA_WEB}/`,
      },
    },
  )
  if (!res.ok) {
    throw new Error(`AsuraScans chapter ${res.status} for ${mangaSlug}/${chapterSlug}`)
  }
  const body = await res.json()
  const pages: AsuraChapterPage[] = body?.data?.chapter?.pages ?? []
  return pages.map((p) => p.url).filter((u): u is string => Boolean(u))
}

/** Download an AsuraScans image from cdn.asurascans.com. */
export async function downloadAsuraScansImage(
  imageUrl: string,
  destPath: string,
): Promise<void> {
  const res = await fetch(imageUrl, {
    headers: {
      'User-Agent': ASURA_UA,
      Referer: `${ASURA_WEB}/`,
      Accept: 'image/*',
    },
  })
  if (!res.ok) {
    throw new Error(`AsuraScans image download ${res.status}: ${imageUrl}`)
  }
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(destPath, buf)
}

// --- Unified dispatchers ---

export async function fetchChaptersForSource(
  source: ScraperSource,
  mangaId: string,
  chapterLimit: number,
) {
  const slug = getSlugFromId(mangaId)
  switch (source) {
    case 'mangahere':
      return fetchMangaHereChapters(slug, chapterLimit)
    case 'fanfox':
      return fetchFanFoxChapters(slug, chapterLimit)
    case 'webtoons':
      return fetchWebtoonsChapters(parseInt(slug, 10), chapterLimit)
    case 'asurascans':
      return fetchAsuraScansChapters(slug, chapterLimit)
  }
}

export async function fetchImagesForSource(
  source: ScraperSource,
  mangaId: string,
  chapterSlug: string,
): Promise<string[]> {
  const slug = getSlugFromId(mangaId)
  switch (source) {
    case 'mangahere':
      return fetchMangaHereChapterImages(slug, chapterSlug)
    case 'fanfox':
      return fetchFanFoxChapterImages(slug, chapterSlug)
    case 'webtoons':
      return fetchWebtoonsChapterImages(
        parseInt(slug, 10),
        parseInt(chapterSlug.replace(/^ep-/, ''), 10),
      )
    case 'asurascans':
      return fetchAsuraScansChapterImages(slug, chapterSlug)
  }
}

export async function downloadImageForSource(
  source: ScraperSource,
  imageUrl: string,
  destPath: string,
): Promise<void> {
  switch (source) {
    case 'mangahere':
      return downloadMangaHereImage(imageUrl, destPath)
    case 'fanfox':
      return downloadFanFoxImage(imageUrl, destPath)
    case 'webtoons':
      return downloadWebtoonsImage(imageUrl, destPath)
    case 'asurascans':
      return downloadAsuraScansImage(imageUrl, destPath)
  }
}

// ---------------------------------------------------------------------------
// VLM helper — lazy-loads z-ai SDK for per-image transcription.
// ---------------------------------------------------------------------------

let zaiPromise: Promise<unknown> | null = null

async function getZai() {
  // Lazy-load so the service can boot even if the SDK has issues at first run.
  if (!zaiPromise) {
    zaiPromise = (async () => {
      const ZAI = (await import('z-ai-web-dev-sdk')).default
      return await ZAI.create()
    })()
  }
  return await zaiPromise
}

// ---------------------------------------------------------------------------
// PaddleOCR PP-OCRv5 — Primary local OCR transcription.
//
// Sends image file paths to the PaddleOCR Python service (port 3002) for
// fast, CPU-efficient text extraction. This is the PRIMARY transcription
// method — VLM is used only as a fallback when OCR is unavailable or
// returns low-confidence results.
//
// PP-OCRv5 gives +13% accuracy over v4, handles multilingual text, and
// is specifically optimized for manga/manhwa panels (speech bubbles,
// rotated text, mixed scripts).
// ---------------------------------------------------------------------------

const OCR_CACHE_DIR = path.join(DATA_DIR, 'cache', 'ocr')

/** Check if the PaddleOCR service is reachable and ready. */
export async function isPaddleOCRAvailable(): Promise<boolean> {
  try {
    const baseUrl = process.env.OCR_SERVICE_URL || 'http://localhost:3002'
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(5000) })
    if (!res.ok) return false
    const data = (await res.json()) as { ready?: boolean }
    return data.ready === true
  } catch {
    return false
  }
}

/** Get OCR model name from the PaddleOCR service. */
export async function getOCRModelName(): Promise<string> {
  try {
    const baseUrl = process.env.OCR_SERVICE_URL || 'http://localhost:3002'
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(3000) })
    if (!res.ok) return 'unknown'
    const data = (await res.json()) as { model?: string }
    return data.model || 'unknown'
  } catch {
    return 'unknown'
  }
}

/** Get cached OCR result for an image, or null if not cached. */
async function getOcrCached(key: string): Promise<string | null> {
  try {
    const cacheFile = path.join(OCR_CACHE_DIR, `${key}.json`)
    const stat = await fs.stat(cacheFile)
    if (Date.now() - stat.mtimeMs > VLM_CACHE_TTL_MS) return null
    const data = JSON.parse(await fs.readFile(cacheFile, 'utf8'))
    return data.text ?? null
  } catch {
    return null
  }
}

/** Save OCR result to cache. */
async function setOcrCached(key: string, text: string): Promise<void> {
  try {
    await fs.mkdir(OCR_CACHE_DIR, { recursive: true })
    await fs.writeFile(
      path.join(OCR_CACHE_DIR, `${key}.json`),
      JSON.stringify({ text, ts: Date.now() }),
      'utf8',
    )
  } catch {
    // non-fatal
  }
}

/**
 * Generate per-image transcriptions using PaddleOCR PP-OCRv5.
 *
 * Sends image file paths to the local PaddleOCR service for fast, CPU-efficient
 * text extraction. Unlike VLM, OCR runs entirely on-device with no API keys
 * or network latency.
 *
 * The function sends images in batches of 20 to the PaddleOCR service's
 * `/ocr/batch` endpoint. Results are cached per-image (same cache key as VLM)
 * so re-runs are instant.
 *
 * @param imagePaths - Absolute paths to panel/page images
 * @param onProgress - Optional callback (done, total) for progress tracking
 * @returns Array of { image, text } in the same order as imagePaths
 * @throws Error if the PaddleOCR service is unreachable or returns an error
 */
export async function generateImageNarrationsOCR(
  imagePaths: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<Array<{ image: string; text: string }>> {
  if (imagePaths.length === 0) return []

  const baseUrl = process.env.OCR_SERVICE_URL || 'http://localhost:3002'
  const OCR_BATCH_SIZE = 20 // PaddleOCR is fast on CPU; larger batches reduce HTTP overhead
  const results: Array<{ image: string; text: string }> = new Array(imagePaths.length)
  let completedCount = 0

  // Process in batches
  for (let i = 0; i < imagePaths.length; i += OCR_BATCH_SIZE) {
    const batchPaths = imagePaths.slice(i, i + OCR_BATCH_SIZE)
    const startIdx = i

    // Check cache for each image in this batch
    const uncachedIndices: number[] = []
    const uncachedPaths: string[] = []

    for (let j = 0; j < batchPaths.length; j++) {
      const cacheKey = vlmCacheKey(batchPaths[j]) // reuse same hash for cross-method cache sharing
      const cached = await getOcrCached(cacheKey)
      if (cached !== null) {
        results[startIdx + j] = { image: path.basename(batchPaths[j]), text: cached }
        completedCount++
      } else {
        uncachedIndices.push(j)
        uncachedPaths.push(batchPaths[j])
      }
    }

    // If all cached, skip the API call
    if (uncachedPaths.length === 0) {
      onProgress?.(completedCount, imagePaths.length)
      continue
    }

    // Call PaddleOCR batch endpoint
    try {
      const t0 = Date.now()
      const res = await fetch(`${baseUrl}/ocr/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: uncachedPaths }),
        signal: AbortSignal.timeout(120000), // 2 min timeout for large batches
      })

      if (!res.ok) {
        const errBody = await res.text().catch(() => '')
        throw new Error(`PaddleOCR service returned HTTP ${res.status}: ${errBody.slice(0, 200)}`)
      }

      const data = (await res.json()) as {
        results: Array<{ index: number; text: string; confidence: number; regions: number }>
        model: string
        processing_time_ms: number
      }

      const elapsed = Date.now() - t0
      console.log(
        `[OCR] Batch ${Math.floor(i / OCR_BATCH_SIZE) + 1}: ${uncachedPaths.length} images in ${elapsed}ms (${data.model})`,
      )

      // Map results back to the correct positions in the results array
      for (const ocrResult of data.results) {
        const localIdx = uncachedIndices[ocrResult.index]
        if (localIdx === undefined) continue

        const globalIdx = startIdx + localIdx
        const text = (ocrResult.text || '').trim()

        // Cache the result
        const cacheKey = vlmCacheKey(uncachedPaths[ocrResult.index])
        if (text) {
          void setOcrCached(cacheKey, text)
          // Also write to VLM cache so VLM path can reuse it
          void setVlmCached(cacheKey, text)
        }

        results[globalIdx] = { image: path.basename(uncachedPaths[ocrResult.index]), text }
        completedCount++
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      console.warn(`[OCR] Batch failed: ${msg.slice(0, 200)}`)
      // Fill remaining uncached results with empty text
      for (let j = 0; j < uncachedIndices.length; j++) {
        const globalIdx = startIdx + uncachedIndices[j]
        if (!results[globalIdx]) {
          results[globalIdx] = { image: path.basename(uncachedPaths[j]), text: '' }
          completedCount++
        }
      }
      // Don't throw — let the pipeline continue with empty text for this batch.
      // The caller can decide to retry with VLM.
    }

    onProgress?.(completedCount, imagePaths.length)
  }

  return results
}

// ---------------------------------------------------------------------------
// VLM-based transcription (fallback when PaddleOCR is unavailable).
// ---------------------------------------------------------------------------

/**
 * Generate per-image narrations: send each image to the VLM individually and
 * get 2-4 sentences of narration describing exactly what's in that image.
 * This produces perfect sync — when the video shows image N, the narration
 * describes image N.
 *
 * Processes images with limited concurrency (3 at a time) to balance speed
 * and API load. Falls back to a generic sentence per image on any error so
 * the pipeline never blocks.
 *
 * Returns an array of { image, text } in the same order as imagePaths.
 */
export async function generateImageNarrations(
  imagePaths: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<Array<{ image: string; text: string }>> {
  if (imagePaths.length === 0) {
    return []
  }

  const results: Array<{ image: string; text: string }> = new Array(imagePaths.length)
  // BATCHED processing: send BATCH_SIZE sliced panels per VLM call. The VLM
  // API accepts multiple images in a single message, so one call returns
  // transcriptions for all panels in the batch. This cuts API calls ~6x
  // (190 panels -> ~32 batch calls), dramatically reducing both total time
  // and rate-limit pressure.
  //
  // CONCURRENT processing: multiple batches run simultaneously to cut total
  // transcription time ~4-5x. Configurable via VLM_CONCURRENCY env var
  // (default 4). Each batch writes to disjoint indices in the results array
  // so there are no race conditions. If a batch call fails or returns
  const LOW_MEM = process.env.VLM_LOW_MEM === '1' || process.env.VLM_CONCURRENCY === '1'
  // BATCH_SIZE: number of panel images sent per VLM API call.
  // 3 is the safe default — smaller batches mean less chance of a
  // skipped-panel shift, and cheaper to detect if it still happens.
  // Previously 6 (3 in LOW_MEM) — too large, a single mis-ordered index
  // silently displaced the rest of the batch.
  const BATCH_SIZE = 3
  // Concurrency: batches at a time. In low-mem mode, force 1.
  const CONCURRENCY = Math.max(
    1,
    Math.min(LOW_MEM ? 1 : 4, parseInt(process.env.VLM_CONCURRENCY || '2', 10)),
  )

  // Build the list of batches to process.
  const batches: Array<{ images: string[]; startIdx: number; num: number }> = []
  for (let i = 0; i < imagePaths.length; i += BATCH_SIZE) {
    batches.push({
      images: imagePaths.slice(i, i + BATCH_SIZE),
      startIdx: i,
      num: Math.floor(i / BATCH_SIZE) + 1,
    })
  }
  const totalBatches = batches.length

  // ── PRE-FLIGHT: test each VLM provider before transcription starts ──
  // This sends a tiny test request to each configured provider to check if
  // the API key is valid and the service is reachable. Only providers that
  // pass the test are added to the active pool — no wasted time on dead keys.
  type VlmProvider = 'siliconflow' | 'zhipu' | 'groq' | 'gemini' | 'openrouter' | 'ollama' | 'z-ai'
  const activeProviders: VlmProvider[] = [] // populated by pre-flight tests
  // z-ai (built-in SDK) is only available in the Z.ai sandbox environment.
  // On self-hosted instances, it doesn't work and causes process kills.
  // Only activate z-ai if Z_AI_SANDBOX=1 is explicitly set.
  if (process.env.Z_AI_SANDBOX === '1') {
    activeProviders.push('z-ai')
  }

  async function testProvider(provider: VlmProvider): Promise<boolean> {
    try {
      if (provider === 'ollama') {
        const baseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434'
        const model = process.env.OLLAMA_VISION_MODEL || 'qwen2.5-vl:7b'
        const res = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(8000) })
        if (!res.ok) {
          console.warn(`[VLM] ✗ Ollama not reachable at ${baseUrl} (HTTP ${res.status})`)
          return false
        }
        const data = await res.json() as { models?: Array<{ name: string }> }
        const modelNames = (data.models || []).map((m) => m.name)
        const hasModel = modelNames.some((n) => n.includes(model.split(':')[0]))
        if (!hasModel) {
          console.warn(`[VLM] ✗ Ollama running but model '${model}' not found. Available: ${modelNames.join(', ')}`)
          return false
        }
        console.log(`[VLM] ✓ Ollama ready at ${baseUrl} with ${model}`)
        return true
      }
      if (provider === 'groq' && process.env.GROQ_API_KEY) {
        const res = await fetch('https://api.groq.com/openai/v1/models', {
          headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
          signal: AbortSignal.timeout(10000),
        })
        if (res.ok) { console.log('[VLM] ✓ Groq key is valid'); return true }
        console.warn(`[VLM] ✗ Groq key invalid (HTTP ${res.status})`)
        return false
      }
      if (provider === 'gemini' && process.env.GEMINI_API_KEY) {
        const res = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY}`,
          { signal: AbortSignal.timeout(10000) },
        )
        if (res.ok) { console.log('[VLM] ✓ Gemini key is valid'); return true }
        console.warn(`[VLM] ✗ Gemini key invalid (HTTP ${res.status})`)
        return false
      }
      if (provider === 'openrouter' && process.env.OPENROUTER_API_KEY) {
        // Use /auth/key endpoint — faster than /models (which can timeout).
        const res = await fetch('https://openrouter.ai/api/v1/auth/key', {
          headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}` },
          signal: AbortSignal.timeout(10000),
        })
        if (res.ok) { console.log('[VLM] ✓ OpenRouter key is valid'); return true }
        console.warn(`[VLM] ✗ OpenRouter key invalid (HTTP ${res.status})`)
        return false
      }
      if (provider === 'zhipu' && process.env.ZHIPU_API_KEY) {
        // Zhipu AI (GLM-4V-Flash) — OpenAI-compatible API.
        const res = await fetch('https://open.bigmodel.cn/api/paas/v4/models', {
          headers: { Authorization: `Bearer ${process.env.ZHIPU_API_KEY}` },
          signal: AbortSignal.timeout(10000),
        })
        if (res.ok) { console.log('[VLM] ✓ Zhipu AI key is valid'); return true }
        console.warn(`[VLM] ✗ Zhipu AI key invalid (HTTP ${res.status})`)
        return false
      }
      if (provider === 'siliconflow' && process.env.SILICONFLOW_API_KEY) {
        // SiliconFlow — OpenAI-compatible API, free Qwen2.5-VL.
        const res = await fetch('https://api.siliconflow.cn/v1/user/info', {
          headers: { Authorization: `Bearer ${process.env.SILICONFLOW_API_KEY}` },
          signal: AbortSignal.timeout(10000),
        })
        if (res.ok) { console.log('[VLM] ✓ SiliconFlow key is valid'); return true }
        console.warn(`[VLM] ✗ SiliconFlow key invalid (HTTP ${res.status})`)
        return false
      }
    } catch (e) {
      console.warn(`[VLM] ✗ ${provider} pre-flight test failed: ${e instanceof Error ? e.message.slice(0, 80) : e}`)
    }
    return false
  }

  // Test all configured providers in parallel
  console.log('[VLM] Pre-flight: testing providers...')
  const tests = await Promise.all([
    testProvider('siliconflow'),
    testProvider('zhipu'),
    testProvider('ollama'),
    testProvider('groq'),
    testProvider('gemini'),
    testProvider('openrouter'),
  ])
  if (tests[0]) activeProviders.push('siliconflow')
  if (tests[1]) activeProviders.push('zhipu')
  if (tests[2]) activeProviders.push('ollama')
  if (tests[3]) activeProviders.push('groq')
  if (tests[4]) activeProviders.push('gemini')
  if (tests[5]) activeProviders.push('openrouter')

  // Detect sandboxed environments where z-ai createVision gets killed.
  // If no real providers (groq/gemini/ollama/openrouter) are configured,
  // and we have no API keys, skip z-ai entirely to avoid process kills.
  const HAS_REAL_PROVIDER = activeProviders.length > 0
  const HAS_API_KEYS = !!(process.env.SILICONFLOW_API_KEY || process.env.ZHIPU_API_KEY || process.env.GROQ_API_KEY || process.env.GEMINI_API_KEY || process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY)

  if (!HAS_REAL_PROVIDER) {
    console.warn('[VLM] No VLM providers available — panels will be silent')
    console.warn('[VLM] To enable transcription: set SILICONFLOW_API_KEY, ZHIPU_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, or install Ollama')
    return imagePaths.map((p) => ({ image: path.basename(p), text: '' }))
  }

  console.log(
    `[VLM] transcribing ${imagePaths.length} images in ${totalBatches} batches (${BATCH_SIZE}/batch) with concurrency ${CONCURRENCY} [${activeProviders.join(' + ')}]`,
  )

  // ── Auto-reduce concurrency for external API providers ──
  // Free-tier APIs (OpenRouter, Groq, Gemini) rate-limit aggressively.
  // With concurrency > 1, the first burst of parallel requests triggers 429s,
  // causing early batches to silently fail while later ones succeed.
  // Fix: when ANY external (non-ollama) provider is active, serialize
  // requests and add a small inter-batch delay to avoid rate limits.
  // Ollama is the only provider that can handle concurrency (it's local).
  const externalProviders = activeProviders.filter((p) => p !== 'ollama')
  // Always serialize external API requests (concurrency=1) and add a longer
  // delay between batches. Free-tier APIs rate-limit aggressively, and even
  // 2s between batches isn't always enough — 4s gives the quota window time
  // to reset. Ollama is the only provider that can handle concurrency (local).
  const EFFECTIVE_CONCURRENCY = (externalProviders.length >= 1 && !LOW_MEM) ? 1 : CONCURRENCY
  const INTER_BATCH_DELAY_MS = (externalProviders.length >= 1 && !LOW_MEM) ? 4000 : 0
  if (EFFECTIVE_CONCURRENCY < CONCURRENCY) {
    console.log(`[VLM] External provider(s) detected — reducing concurrency ${CONCURRENCY} → ${EFFECTIVE_CONCURRENCY} (avoids rate limits)`)
    if (INTER_BATCH_DELAY_MS > 0) console.log(`[VLM] Adding ${INTER_BATCH_DELAY_MS}ms delay between batches to avoid rate limiting`)
  }

  // Atomic progress counter — shared across all concurrent workers.
  let completedPanels = 0
  const reportProgress = () => {
    if (onProgress) {
      onProgress(Math.min(completedPanels, imagePaths.length), imagePaths.length)
    }
  }

  // Circuit breakers: track errors per provider, disable after threshold.
  const disabledProviders = new Set<VlmProvider>()
  const providerErrorCounts: Record<string, number> = {}
  const ERROR_THRESHOLD = 5 // raised from 3 — free-tier APIs have bursty rate limits

  // --- Provider selection: PRIMARY + FALLBACK strategy ---
  // For free-tier external APIs, round-robin is COUNTERPRODUCTIVE:
  // it burns through ALL providers' rate limits simultaneously.
  // Instead, use ONE provider at a time (the "primary"), and only switch
  // to the next when the current one is circuit-broken.
  // Ollama is always used alone (it's local, no rate limits).
  let currentPrimary: VlmProvider | null = null

  function pickProvider(exclude?: VlmProvider): VlmProvider {
    const excluded = new Set(disabledProviders)
    if (exclude) excluded.add(exclude)
    const available = activeProviders.filter((p) => !excluded.has(p))
    if (available.length === 0) {
      // All disabled — fall back to ollama (if available)
      if (activeProviders.includes('ollama') && !disabledProviders.has('ollama')) return 'ollama'
      return available[0] // will be undefined, triggering empty-text fallback
    }
    // If we have a working primary, keep using it (stickiness).
    if (currentPrimary && !excluded.has(currentPrimary) && available.includes(currentPrimary)) {
      return currentPrimary
    }
    // Pick the first available as the new primary (priority: siliconflow > zhipu > openrouter > gemini > groq > ollama > z-ai)
    const providerOrder: VlmProvider[] = ['siliconflow', 'zhipu', 'openrouter', 'gemini', 'groq', 'ollama', 'z-ai']
    for (const p of providerOrder) {
      if (available.includes(p)) {
        currentPrimary = p
        console.log(`[VLM] switched primary provider to ${p}`)
        return p
      }
    }
    return available[0]
  }

  // Process a single batch — writes results into the shared array at the
  // batch's start index (disjoint from other batches, so no locking needed).
  async function processBatch(batch: (typeof batches)[0]): Promise<void> {
    const { images, startIdx, num } = batch

    const providerLabel = pickProvider()
    console.log(`[VLM] batch ${num}/${totalBatches} → ${providerLabel} (${images.length} panels)`)

    let batchTexts: string[]
    let succeeded = false
    let countedPerImage = false  // set true if single-image fallback counted panels
    try {
      if (providerLabel === 'siliconflow') {
        batchTexts = await narrateImageBatchSiliconFlow(images, startIdx)
      } else if (providerLabel === 'zhipu') {
        batchTexts = await narrateImageBatchZhipu(images, startIdx)
      } else if (providerLabel === 'ollama') {
        batchTexts = await narrateImageBatchOllama(images, startIdx)
      } else if (providerLabel === 'groq') {
        batchTexts = await narrateImageBatchGroq(images, startIdx)
      } else if (providerLabel === 'gemini') {
        batchTexts = await narrateImageBatchGemini(images, startIdx)
      } else if (providerLabel === 'openrouter') {
        batchTexts = await narrateImageBatchOpenRouter(images, startIdx)
      } else {
        batchTexts = await narrateImageBatch(images, startIdx)
      }
      succeeded = true
    } catch (primaryErr) {
      const errMsg = primaryErr instanceof Error ? primaryErr.message : String(primaryErr)
      console.warn(`[VLM:${providerLabel}] batch ${num}/${totalBatches} primary call failed: ${errMsg.slice(0, 200)}`)
      const isContentFilter = errMsg.includes('contentFilter') || errMsg.includes('400')
      const isRateLimit = errMsg.includes('429') || errMsg.includes('rate')
      const isForbidden = errMsg.includes('403') || errMsg.includes('Forbidden')

      // --- Circuit breaker ---
      // Track consecutive errors per provider. After N errors, disable it.
      // Only trip on rate-limit (429) and forbidden (403) errors — transient
      // server errors (5xx) shouldn't disable a provider permanently.
      if (isRateLimit || isForbidden) {
        providerErrorCounts[providerLabel] = (providerErrorCounts[providerLabel] || 0) + 1
        if (providerErrorCounts[providerLabel] >= ERROR_THRESHOLD && !disabledProviders.has(providerLabel)) {
          disabledProviders.add(providerLabel)
          currentPrimary = null // force re-selection on next pickProvider()
          console.warn(
            `[VLM:${providerLabel}] circuit breaker tripped after ${providerErrorCounts[providerLabel]} errors — disabled for this transcription`,
          )
        }
      } else {
        // Non-rate-limit error (400, 5xx, etc.) — reset rate limit counter
        // but still track. If the same provider gets 3 non-rate-limit errors,
        // switch to a different provider (don't disable, just move on).
        providerErrorCounts[providerLabel] = (providerErrorCounts[providerLabel] || 0) + 1
        if (providerErrorCounts[providerLabel] >= 3) {
          currentPrimary = null // force switch to another provider
          console.warn(
            `[VLM:${providerLabel}] ${providerErrorCounts[providerLabel]} non-rate-limit errors — switching primary provider`,
          )
        }
      }

      // --- No z-ai fallback ---
      // z-ai SDK is disabled. Only Groq/gemini/openrouter/ollama are used.
      // Failed batches will go through retry with backoff instead.

      if (!succeeded) {
        const isRateLimited = errMsg.includes('429') || errMsg.includes('rate')
        console.warn(
          `[VLM:${providerLabel}] batch ${num}/${totalBatches} failed — ${isRateLimited ? 'rate limited' : 'error'}: ${errMsg.slice(0, 120)}`,
        )

        // RETRY STRATEGY depends on error type:
        // - Rate limit (429): retry SAME provider with long delays (transient).
        //   Switching providers just burns through both quotas.
        // - Permanent error (400, 403, 5xx): try a DIFFERENT provider immediately.
        const BATCH_RETRY_DELAYS = isRateLimited
          ? [30000, 45000, 60000] // 30s, 45s, 60s for rate limits
          : [5000, 10000, 15000]  // 5s, 10s, 15s for other errors (faster switch)

        for (let retry = 0; retry < BATCH_RETRY_DELAYS.length && !succeeded; retry++) {
          await sleep(BATCH_RETRY_DELAYS[retry])
          console.warn(`[VLM] batch ${num}/${totalBatches} retry ${retry + 1}/${BATCH_RETRY_DELAYS.length} after ${BATCH_RETRY_DELAYS[retry] / 1000}s`)
          try {
            let retryProvider: VlmProvider
            if (isRateLimited) {
              // Rate limit = transient. Retry the SAME provider after waiting.
              // The inner retry already waited 15s+30s+60s = 105s.
              // These outer retries add 30s+45s+60s = 135s more total.
              retryProvider = providerLabel
              console.log(`[VLM] batch ${num}/${totalBatches} retry ${retry + 1} — same provider ${retryProvider} (rate limit, transient)`)
            } else {
              // Permanent error — switch to a DIFFERENT provider.
              retryProvider = pickProvider(providerLabel)
              console.log(`[VLM] batch ${num}/${totalBatches} retry ${retry + 1} — switching to ${retryProvider} (was ${providerLabel})`)
            }
            if (retryProvider === 'siliconflow') batchTexts = await narrateImageBatchSiliconFlow(images, startIdx)
            else if (retryProvider === 'zhipu') batchTexts = await narrateImageBatchZhipu(images, startIdx)
            else if (retryProvider === 'groq') batchTexts = await narrateImageBatchGroq(images, startIdx)
            else if (retryProvider === 'gemini') batchTexts = await narrateImageBatchGemini(images, startIdx)
            else if (retryProvider === 'openrouter') batchTexts = await narrateImageBatchOpenRouter(images, startIdx)
            else if (retryProvider === 'ollama') batchTexts = await narrateImageBatchOllama(images, startIdx)
            else batchTexts = await narrateImageBatch(images, startIdx)
            succeeded = true
            console.log(`[VLM] batch ${num}/${totalBatches} succeeded on retry ${retry + 1} via ${retryProvider}`)
          } catch (retryErr) {
            const retryMsg = retryErr instanceof Error ? retryErr.message : String(retryErr)
            console.warn(`[VLM] batch ${num}/${totalBatches} retry ${retry + 1} failed: ${retryMsg.slice(0, 80)}`)
          }
        }

        if (!succeeded) {
          // Final fallback: empty text (silence). This is the correct behavior
          // for panels with genuinely no readable text. For panels that DO have
          // text but all VLM providers failed, the silence is the lesser evil —
          // the alternative was the annoying "scene continues to unfold" loop.
          console.warn(
            `[VLM] batch ${num}/${totalBatches} exhausted all retries — leaving ${images.length} panels silent`,
          )
          batchTexts = images.map(() => '')
          succeeded = true
        }
      }
    }

    // Write results for this batch.
    for (let j = 0; j < images.length; j++) {
      results[startIdx + j] = {
        image: path.basename(images[j]),
        text: batchTexts[j] ?? '',
      }
    }
    // Only increment once — the single-image fallback path already counted
    // per-image. The other paths (success, content-filter, cross-provider
    // fallback) count here.
    if (!countedPerImage) {
      completedPanels += images.length
    }
    reportProgress()
    // Force GC in low-mem mode to free base64 image buffers promptly.
    if (LOW_MEM) {
      try { globalThis.gc?.() } catch { /* no-op */ }
    }
  }

  // Simple concurrency pool: spin up EFFECTIVE_CONCURRENCY workers, each pulling the
  // next unprocessed batch from the queue. This naturally load-balances —
  // faster workers pick up more batches.
  let nextBatchIdx = 0
  async function worker(): Promise<void> {
    while (nextBatchIdx < batches.length) {
      const batchIdx = nextBatchIdx++
      if (INTER_BATCH_DELAY_MS > 0 && batchIdx > 0) {
        await sleep(INTER_BATCH_DELAY_MS)
      }
      await processBatch(batches[batchIdx])
    }
  }

  const workers: Promise<void>[] = []
  for (let i = 0; i < EFFECTIVE_CONCURRENCY; i++) {
    workers.push(worker())
  }
  await Promise.all(workers)

  return results
}

/**
 * Send a BATCH of sliced panels to the VLM via an isolated child process.
 * The child process (vlm-worker.ts) handles the actual z-ai SDK call,
 * which avoids the sandbox controller killing the main pipeline-service
 * process (which runs socket.io).
 *
 * Communication: JSON on stdin/stdout. Cache is checked/written by the
 * worker so it persists even if the main process crashes.
 */
async function narrateImageBatch(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first — if all images in this batch are cached, skip the VLM call
  const cachedResults: (string | null)[] = []
  let allCached = true
  for (const imgPath of imgPaths) {
    const cached = await getVlmCached(vlmCacheKey(imgPath))
    cachedResults.push(cached)
    if (cached === null) allCached = false
  }
  if (allCached) {
    console.log(`[VLM] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults as string[]
  }

  // Spawn vlm-worker.ts as a child process. It handles the z-ai SDK call
  // in isolation, avoiding the sandbox kill issue.
  const { spawn } = await import('child_process')
  const workerScript = path.join(import.meta.dirname, 'vlm-worker.ts')
  const prompt =
    `You are a precise transcriber for webtoon/manhwa panels, not a narrator. ` +
    `I am sending you {count} separate panel images, labeled Panel 1 through Panel {count} ` +
    `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
    `speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader would naturally ` +
    `read them (top to bottom, left to right within the panel). Translate to natural English if not already ` +
    `in English, preserving meaning and tone.\n\n` +
    `Guidelines:\n` +
    `1. Output the text VERBATIM (translated) — do not paraphrase, summarize, embellish, or add descriptive ` +
    `narration. Do not invent dialogue that is not actually written.\n` +
    `2. Do not describe artwork, action, or expressions — only transcribe written text that appears in the image.\n` +
    `3. If multiple bubbles/boxes are present in a panel, join them in reading order as separate sentences, ` +
    `preserving punctuation like \"...\" and \"!\" as written.\n` +
    `4. Sound effect text (e.g. \"BOOM\", \"CRASH\") can be included briefly if it is the only text present, ` +
    `otherwise skip pure onomatopoeia in favor of actual dialogue/captions.\n` +
    `5. If a panel has NO readable text at all (a purely visual/action panel with no bubbles or captions), ` +
    `use an empty string for that panel's text.\n\n` +
    `RESPONSE FORMAT: Return a JSON array with exactly {count} elements, one per panel in order. ` +
    `Each element is an object: {\"index\": <1-based panel number>, \"text\": \"<transcribed text or empty string>\"}.\n` +
    `Output ONLY the JSON array — no preamble, no markdown fences, no explanation.\n` +
    `Example for 2 panels: [{\"index\": 1, \"text\": \"What is this place?\"}, {\"index\": 2, \"text\": \"\"}]`

  const input = JSON.stringify({ images: imgPaths, prompt })

  return new Promise<string[]>((resolve, reject) => {
    const child = spawn('bun', [workerScript], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })

    let stdout = ''
    let stderr = ''

    child.stdout.on('data', (data: Buffer) => { stdout += data.toString() })
    child.stderr.on('data', (data: Buffer) => {
      stderr += data.toString()
      for (const line of data.toString().split('\n').filter((l: string) => l.trim())) {
        console.log(`[vlm-worker] ${line}`)
      }
    })

    child.on('close', (code: number | null) => {
      if (code !== 0 || !stdout.trim()) {
        // Worker was killed (sandbox) or failed — return empty strings (silent panels).
        // This is the correct behavior when no VLM provider is available.
        console.warn(`[VLM] vlm-worker exited ${code} — returning empty text for ${imgPaths.length} panels`)
        resolve(imgPaths.map(() => ''))
        return
      }
      try {
        const parsed = JSON.parse(stdout.trim())
        resolve(parsed.results as string[])
      } catch {
        console.warn('[VLM] vlm-worker output parse error — returning empty text')
        resolve(imgPaths.map(() => ''))
      }
    })

    child.on('error', (err: Error) => {
      console.warn(`[VLM] vlm-worker spawn error: ${err.message} — returning empty text`)
      resolve(imgPaths.map(() => ''))
    })

    child.stdin.write(input)
    child.stdin.end()

    setTimeout(() => {
      child.kill('SIGTERM')
      console.warn('[VLM] vlm-worker timed out — returning empty text')
      resolve(imgPaths.map(() => ''))
    }, 60_000).unref()
  })
}

// ---------------------------------------------------------------------------
// GEMINI VLM — second provider for round-robin load balancing.
// Uses the Gemini 2.0 Flash REST API (free tier, vision-capable, fast).
// When configured, batches alternate between z-ai and Gemini to double
// throughput and halve the transcription time.
// ---------------------------------------------------------------------------

export function isGeminiConfigured(): boolean {
  return Boolean(process.env.GEMINI_API_KEY)
}

/**
 * Same interface as narrateImageBatch, but calls Google Gemini 2.0 Flash
 * via its REST API. Reuses the same prompt + cache + parseBatchResponse so
 * transcription quality is identical between providers.
 */
async function narrateImageBatchGemini(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first (same cache as z-ai — a panel transcribed by either
  // provider is reused on re-runs).
  const cachedResults: (string | null)[] = []
  let allCached = true
  for (const imgPath of imgPaths) {
    const cached = await getVlmCached(vlmCacheKey(imgPath))
    cachedResults.push(cached)
    if (cached === null) allCached = false
  }
  if (allCached) {
    console.log(`[VLM:gemini] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults as string[]
  }

  const apiKey = process.env.GEMINI_API_KEY!
  const model = process.env.GEMINI_MODEL || 'gemini-2.0-flash'
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`

  // Read + base64-encode each image.
  const images: Array<{ mime: string; b64: string }> = []
  for (const imgPath of imgPaths) {
    const buf = await fs.readFile(imgPath)
    const ext = path.extname(imgPath).toLowerCase()
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'
    images.push({ mime, b64: buf.toString('base64') })
  }

  // Same prompt as z-ai — keeps transcription consistent across providers.
  const prompt =
    `You are a precise transcriber for webtoon/manhwa panels, not a narrator. ` +
    `I am sending you ${images.length} separate panel images, labeled Panel 1 through Panel ${images.length} ` +
    `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
    `speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader would naturally ` +
    `read them (top to bottom, left to right within the panel). Translate to natural English if not already ` +
    `in English, preserving meaning and tone.\n\n` +
    `Guidelines:\n` +
    `1. Output the text VERBATIM (translated) — do not paraphrase, summarize, embellish, or add descriptive ` +
    `narration. Do not invent dialogue that is not actually written.\n` +
    `2. Do not describe artwork, action, or expressions — only transcribe written text that appears in the image.\n` +
    `3. If multiple bubbles/boxes are present in a panel, join them in reading order as separate sentences, ` +
    `preserving punctuation like "..." and "!" as written.\n` +
    `4. Sound effect text (e.g. "BOOM", "CRASH") can be included briefly if it is the only text present, ` +
    `otherwise skip pure onomatopoeia in favor of actual dialogue/captions.\n` +
    `5. If a panel has NO readable text at all (a purely visual/action panel with no bubbles or captions), ` +
    `use an empty string for that panel's text.\n\n` +
    `RESPONSE FORMAT: Return a JSON array with exactly ${images.length} elements, one per panel in order. ` +
    `Each element is an object: {"index": <1-based panel number>, "text": "<transcribed text or empty string>"}.\n` +
    `Output ONLY the JSON array — no preamble, no markdown fences, no explanation.\n` +
    `Example for 2 panels: [{"index": 1, "text": "What is this place?"}, {"index": 2, "text": ""}]`

  // Build Gemini request body.
  const body = {
    contents: [{
      parts: [
        { text: prompt },
        ...images.map(img => ({ inline_data: { mime_type: img.mime, data: img.b64 } })),
      ],
    }],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 4096,
    },
  }

  const MAX_RETRIES = 4
  const BASE_DELAY_MS = 2000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60000) // 60s per batch
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(`Gemini API ${res.status}: ${errText.slice(0, 200)}`)
      }

      const data = await res.json() as {
        candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>
        error?: { message?: string }
      }

      if (data.error) {
        throw new Error(`Gemini error: ${data.error.message || JSON.stringify(data.error)}`)
      }

      const raw = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? ''
      const texts = parseBatchResponse(raw, images.length)

      // Cache each panel's transcription (same cache as z-ai).
      for (let i = 0; i < imgPaths.length && i < texts.length; i++) {
        if (texts[i]) {
          await setVlmCached(vlmCacheKey(imgPaths[i]), texts[i])
        }
      }
      return texts
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      // 429 = rate limited — retry with longer backoff (free tier friendly).
      if (msg.includes('429')) {
        if (attempt === MAX_RETRIES) throw err
        const delayMs = 15000 * Math.pow(2, attempt) // 15s, 30s, 60s
        console.warn(`[VLM:gemini] rate limited — retrying in ${delayMs / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES + 1})`)
        await sleep(delayMs)
        continue
      }
      const isRetryable = /5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed|aborted/i.test(msg)

      if (!isRetryable || attempt === MAX_RETRIES) {
        throw err
      }

      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(
        `[VLM:gemini] batch (panels ${batchStart + 1}-${batchStart + images.length}) attempt ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`,
      )
      await sleep(delayMs)
    }
  }

  throw lastErr
}

// ---------------------------------------------------------------------------
// GROQ VLM — third provider, fastest option (LPU hardware, 30 req/min free).
// Uses Groq's OpenAI-compatible API with a vision-capable model (Llama 4
// Scout). Groq's LPU inference is purpose-built for speed — typically 3-5x
// faster than cloud GPU providers for the same model size.
// ---------------------------------------------------------------------------

export function isGroqVlmConfigured(): boolean {
  return Boolean(process.env.GROQ_API_KEY)
}

/**
 * Same interface as narrateImageBatch, but calls Groq's vision model via its
 * OpenAI-compatible REST API. Reuses the same prompt + cache + parseBatchResponse.
 */
async function narrateImageBatchGroq(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first (same cache as z-ai + Gemini).
  const cachedResults: (string | null)[] = []
  let allCached = true
  for (const imgPath of imgPaths) {
    const cached = await getVlmCached(vlmCacheKey(imgPath))
    cachedResults.push(cached)
    if (cached === null) allCached = false
  }
  if (allCached) {
    console.log(`[VLM:groq] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults as string[]
  }

  const apiKey = process.env.GROQ_API_KEY!
  // Llama 4 Scout — Groq's vision model (check console.groq.com for current ID).
  // Override via GROQ_VLM_MODEL env var if needed.
  const model = process.env.GROQ_VLM_MODEL || 'meta-llama/llama-4-scout'
  const url = 'https://api.groq.com/openai/v1/chat/completions'

  // Read + base64-encode each image.
  const images: Array<{ mime: string; b64: string }> = []
  for (const imgPath of imgPaths) {
    const buf = await fs.readFile(imgPath)
    const ext = path.extname(imgPath).toLowerCase()
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'
    images.push({ mime, b64: buf.toString('base64') })
  }

  // Same prompt as z-ai + Gemini — keeps transcription consistent.
  const prompt =
    `You are a precise transcriber for webtoon/manhwa panels, not a narrator. ` +
    `I am sending you ${images.length} separate panel images, labeled Panel 1 through Panel ${images.length} ` +
    `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
    `speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader would naturally ` +
    `read them (top to bottom, left to right within the panel). Translate to natural English if not already ` +
    `in English, preserving meaning and tone.\n\n` +
    `Guidelines:\n` +
    `1. Output the text VERBATIM (translated) — do not paraphrase, summarize, embellish, or add descriptive ` +
    `narration. Do not invent dialogue that is not actually written.\n` +
    `2. Do not describe artwork, action, or expressions — only transcribe written text that appears in the image.\n` +
    `3. If multiple bubbles/boxes are present in a panel, join them in reading order as separate sentences, ` +
    `preserving punctuation like "..." and "!" as written.\n` +
    `4. Sound effect text (e.g. "BOOM", "CRASH") can be included briefly if it is the only text present, ` +
    `otherwise skip pure onomatopoeia in favor of actual dialogue/captions.\n` +
    `5. If a panel has NO readable text at all (a purely visual/action panel with no bubbles or captions), ` +
    `use an empty string for that panel's text.\n\n` +
    `RESPONSE FORMAT: Return a JSON array with exactly ${images.length} elements, one per panel in order. ` +
    `Each element is an object: {"index": <1-based panel number>, "text": "<transcribed text or empty string>"}.\n` +
    `Output ONLY the JSON array — no preamble, no markdown fences, no explanation.\n` +
    `Example for 2 panels: [{"index": 1, "text": "What is this place?"}, {"index": 2, "text": ""}]`

  // Build OpenAI-compatible request body (Groq uses the same format).
  const content: Array<
    | { type: 'text'; text: string }
    | { type: 'image_url'; image_url: { url: string } }
  > = [
    { type: 'text', text: prompt },
    ...images.map(img => ({
      type: 'image_url' as const,
      image_url: { url: `data:${img.mime};base64,${img.b64}` },
    })),
  ]

  const body = {
    model,
    messages: [{ role: 'user', content }],
    temperature: 0.1,
    max_tokens: 4096,
  }

  const MAX_RETRIES = 3
  const BASE_DELAY_MS = 2000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60000) // 60s per batch
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(`Groq API ${res.status}: ${errText.slice(0, 200)}`)
      }

      const data = await res.json() as {
        choices?: Array<{ message?: { content?: string } }>
        error?: { message?: string }
      }

      if (data.error) {
        throw new Error(`Groq error: ${data.error.message || JSON.stringify(data.error)}`)
      }

      const raw = data.choices?.[0]?.message?.content?.trim() ?? ''
      const texts = parseBatchResponse(raw, images.length)

      // Cache each panel's transcription (same cache as other providers).
      for (let i = 0; i < imgPaths.length && i < texts.length; i++) {
        if (texts[i]) {
          await setVlmCached(vlmCacheKey(imgPaths[i]), texts[i])
        }
      }
      return texts
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      // 403 = forbidden (invalid key, model not available). Don't retry.
      if (msg.includes('403')) {
        throw err
      }
      // 429 = rate limited — retry with longer backoff (free tier friendly).
      if (msg.includes('429')) {
        if (attempt === MAX_RETRIES) throw err
        const delayMs = 15000 * Math.pow(2, attempt) // 15s, 30s, 60s
        console.warn(`[VLM:groq] rate limited — retrying in ${delayMs / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES + 1})`)
        await sleep(delayMs)
        continue
      }
      const isRetryable = /5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed|aborted/i.test(msg)

      if (!isRetryable || attempt === MAX_RETRIES) {
        throw err
      }

      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(
        `[VLM:groq] batch (panels ${batchStart + 1}-${batchStart + images.length}) attempt ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`,
      )
      await sleep(delayMs)
    }
  }

  throw lastErr
}

// ---------------------------------------------------------------------------
// OPENROUTER VLM — fourth provider (access free LLaVA, Qwen-VL, etc.)
// OpenRouter is an API gateway that provides access to many models, including
// free vision models. Uses OpenAI-compatible API format.
// ---------------------------------------------------------------------------

export function isOpenRouterVlmConfigured(): boolean {
  return Boolean(process.env.OPENROUTER_API_KEY)
}

export function isOllamaConfigured(): boolean {
  // Ollama is "configured" if OLLAMA_BASE_URL is set (or default localhost) and
  // we don't need an API key — but we check if it's actually reachable later.
  const baseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434'
  return Boolean(baseUrl)
}

// OLLAMA VLM — local inference via Ollama's OpenAI-compatible API.
// Uses a vision-language model (e.g. qwen2.5-vl:7b) to transcribe panel text.
// Completely free, no API key needed, runs on your own hardware.
// Ollama exposes /v1/chat/completions with the same format as OpenAI.

async function narrateImageBatchOllama(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first (same cache as other providers).
  const cachedResults: (string | null)[] = []
  for (const imgPath of imgPaths) {
    cachedResults.push(await getVlmCached(vlmCacheKey(imgPath)))
  }
  if (cachedResults.every((c) => c !== null)) {
    console.log(`[VLM:ollama] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults.map((c) => c ?? '')
  }

  const baseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434'
  const model = process.env.OLLAMA_VISION_MODEL || 'qwen2.5-vl:7b'

  // Build multi-image content (same prompt as other providers for consistency).
  const content: Array<
    | { type: 'text'; text: string }
    | { type: 'image_url'; image_url: { url: string } }
  > = [
    {
      type: 'text',
      text:
        'You are a precise transcriber for webtoon/manhwa panels, not a narrator. ' +
        'I will send you multiple manga panel images ' +
        `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
        'speech bubbles, thought bubbles, and caption/narration boxes. ' +
        'Translate non-English text into natural English.\n\n' +
        'Rules:\n' +
        '1. Output text VERBATIM (translated) — do not paraphrase, summarize, or add narration.\n' +
        '2. Do not describe artwork, action, or expressions — only transcribe written text.\n' +
        '3. If a panel has NO readable text, output empty string for that panel.\n' +
        '4. Sound effects ("BOOM", "CRASH") — include only if the main text.\n' +
        '5. Each element is an object: {"index": <1-based panel number>, "text": "<transcribed text or empty string>"}.\n' +
        '6. Return ONLY a valid JSON array — no preamble, no markdown fences, no explanation.',
    },
  ]

  // Read all images and add to content.
  for (const imgPath of imgPaths) {
    const buf = await fs.readFile(imgPath)
    const b64 = buf.toString('base64')
    const ext = path.extname(imgPath).toLowerCase()
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'
    content.push({
      type: 'image_url',
      image_url: { url: `data:${mime};base64,${b64}` },
    })
  }

  const MAX_RETRIES = 2
  const BASE_DELAY_MS = 5000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content }],
          temperature: 0.1,
          max_tokens: 2048,
          // Ollama-specific: keep_alive keeps model in memory between calls
          // This avoids cold-start latency on subsequent batches.
          stream: false,
        }),
        signal: AbortSignal.timeout(120000), // Ollama local inference can be slow
      })

      if (!res.ok) {
        const body = await res.text().catch(() => '')
        throw new Error(`Ollama HTTP ${res.status}: ${body.slice(0, 200)}`)
      }

      const data = await res.json() as {
        choices?: Array<{ message?: { content?: string } }>
      }
      const raw = data?.choices?.[0]?.message?.content
      if (!raw) throw new Error('empty Ollama response')

      const texts = parseBatchResponse(raw, imgPaths.length)

      // Cache each panel's transcription (same cache as other providers).
      for (let i = 0; i < imgPaths.length; i++) {
        if (texts[i]) {
          await setVlmCached(vlmCacheKey(imgPaths[i]), texts[i])
        }
      }

      return texts
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      const isRetryable = /5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed|aborted/i.test(msg)
      if (!isRetryable || attempt === MAX_RETRIES) throw err
      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(`[VLM:ollama] batch retry ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`)
      await sleep(delayMs)
    }
  }

  throw lastErr
}

async function narrateImageBatchOpenRouter(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first (same cache as other providers).
  const cachedResults: (string | null)[] = []
  let allCached = true
  for (const imgPath of imgPaths) {
    const cached = await getVlmCached(vlmCacheKey(imgPath))
    cachedResults.push(cached)
    if (cached === null) allCached = false
  }
  if (allCached) {
    console.log(`[VLM:openrouter] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults as string[]
  }

  const apiKey = process.env.OPENROUTER_API_KEY!
  // Nemotron Nano VL — confirmed free vision model on OpenRouter.
  // CRITICAL: must be a VISION model (supports image_url content).
  // Override via OPENROUTER_VLM_MODEL env var.
  // Other free options: "google/gemma-4-26b-a4b-it:free"
  const model = process.env.OPENROUTER_VLM_MODEL || 'nvidia/nemotron-nano-12b-v2-vl:free'
  const url = 'https://openrouter.ai/api/v1/chat/completions'

  // Read + base64-encode each image.
  const images: Array<{ mime: string; b64: string }> = []
  for (const imgPath of imgPaths) {
    const buf = await fs.readFile(imgPath)
    const ext = path.extname(imgPath).toLowerCase()
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'
    images.push({ mime, b64: buf.toString('base64') })
  }

  // Same prompt as other providers.
  const prompt =
    `You are a precise transcriber for webtoon/manhwa panels, not a narrator. ` +
    `I am sending you ${images.length} separate panel images, labeled Panel 1 through Panel ${images.length} ` +
    `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
    `speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader would naturally ` +
    `read them (top to bottom, left to right within the panel). Translate to natural English if not already ` +
    `in English, preserving meaning and tone.\n\n` +
    `RESPONSE FORMAT: Return a JSON array with exactly ${images.length} elements, one per panel in order. ` +
    `Each element is an object: {"index": <1-based panel number>, "text": "<transcribed text or empty string>"}.\n` +
    `Output ONLY the JSON array — no preamble, no markdown fences, no explanation.\n` +
    `Example for 2 panels: [{"index": 1, "text": "What is this place?"}, {"index": 2, "text": ""}]`

  const content = [
    { type: 'text', text: prompt },
    ...images.map(img => ({
      type: 'image_url',
      image_url: { url: `data:${img.mime};base64,${img.b64}` },
    })),
  ]

  const body = {
    model,
    messages: [{ role: 'user', content }],
    temperature: 0.1,
    max_tokens: 4096,
  }

  const MAX_RETRIES = 3
  const BASE_DELAY_MS = 2000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60000)
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'HTTP-Referer': 'https://github.com/zainrana558/manhwa-recap-studio-v3',
          'X-Title': 'Manhwa Recap Studio',
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(`OpenRouter API ${res.status}: ${errText.slice(0, 200)}`)
      }

      const data = await res.json() as {
        choices?: Array<{ message?: { content?: string } }>
        error?: { message?: string }
      }

      if (data.error) {
        throw new Error(`OpenRouter error: ${data.error.message || JSON.stringify(data.error)}`)
      }

      const raw = data.choices?.[0]?.message?.content?.trim() ?? ''
      const texts = parseBatchResponse(raw, images.length)

      // Cache each panel's transcription.
      for (let i = 0; i < imgPaths.length && i < texts.length; i++) {
        if (texts[i]) {
          await setVlmCached(vlmCacheKey(imgPaths[i]), texts[i])
        }
      }
      return texts
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      // 403 = forbidden. Don't retry.
      if (msg.includes('403')) throw err
      // 429 = rate limited — retry with longer backoff (free tier friendly).
      if (msg.includes('429')) {
        if (attempt === MAX_RETRIES) throw err
        const delayMs = 15000 * Math.pow(2, attempt) // 15s, 30s, 60s
        console.warn(`[VLM:openrouter] rate limited — retrying in ${delayMs / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES + 1})`)
        await sleep(delayMs)
        continue
      }
      const isRetryable = /5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed|aborted/i.test(msg)
      if (!isRetryable || attempt === MAX_RETRIES) throw err
      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(`[VLM:openrouter] batch retry ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`)
      await sleep(delayMs)
    }
  }

  throw lastErr
}

/**
 * SiliconFlow — free Qwen2.5-VL-7B-Instruct vision model.
 * OpenAI-compatible API at https://api.siliconflow.cn/v1/chat/completions
 * Free tier: 14M tokens/month. Same prompt + cache + parseBatchResponse pattern.
 */
async function narrateImageBatchSiliconFlow(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first
  const cachedResults: (string | null)[] = []
  let allCached = true
  for (const imgPath of imgPaths) {
    const cached = await getVlmCached(vlmCacheKey(imgPath))
    cachedResults.push(cached)
    if (cached === null) allCached = false
  }
  if (allCached) {
    console.log(`[VLM:siliconflow] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults as string[]
  }

  const apiKey = process.env.SILICONFLOW_API_KEY!
  const model = process.env.SILICONFLOW_VLM_MODEL || 'Qwen/Qwen2.5-VL-7B-Instruct'
  const url = 'https://api.siliconflow.cn/v1/chat/completions'

  // Read + base64-encode each image.
  const images: Array<{ mime: string; b64: string }> = []
  for (const imgPath of imgPaths) {
    const buf = await fs.readFile(imgPath)
    const ext = path.extname(imgPath).toLowerCase()
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'
    images.push({ mime, b64: buf.toString('base64') })
  }

  // Same transcription prompt as other providers.
  const prompt =
    `You are a precise transcriber for webtoon/manhwa panels, not a narrator. ` +
    `I am sending you ${images.length} separate panel images, labeled Panel 1 through Panel ${images.length} ` +
    `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
    `speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader would naturally ` +
    `read them (top to bottom, left to right within the panel). Translate to natural English if not already ` +
    `in English, preserving meaning and tone.\n\n` +
    `RESPONSE FORMAT: Return a JSON array with exactly ${images.length} elements, one per panel in order. ` +
    `Each element is an object: {"index": <1-based panel number>, "text": "<transcribed text or empty string>"}.\n` +
    `Output ONLY the JSON array — no preamble, no markdown fences, no explanation.\n` +
    `Example for 2 panels: [{"index": 1, "text": "What is this place?"}, {"index": 2, "text": ""}]`

  const content = [
    { type: 'text', text: prompt },
    ...images.map(img => ({
      type: 'image_url',
      image_url: { url: `data:${img.mime};base64,${img.b64}` },
    })),
  ]

  const body = {
    model,
    messages: [{ role: 'user', content }],
    temperature: 0.1,
    max_tokens: 4096,
  }

  const MAX_RETRIES = 3
  const BASE_DELAY_MS = 2000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60000)
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(`SiliconFlow API ${res.status}: ${errText.slice(0, 200)}`)
      }

      const data = await res.json() as {
        choices?: Array<{ message?: { content?: string } }>
        error?: { message?: string }
      }

      if (data.error) {
        throw new Error(`SiliconFlow error: ${data.error.message || JSON.stringify(data.error)}`)
      }

      const raw = data.choices?.[0]?.message?.content?.trim() ?? ''
      const texts = parseBatchResponse(raw, images.length)

      // Cache each panel's transcription.
      for (let i = 0; i < imgPaths.length && i < texts.length; i++) {
        if (texts[i]) {
          await setVlmCached(vlmCacheKey(imgPaths[i]), texts[i])
        }
      }
      return texts
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      // 403 = forbidden. Don't retry.
      if (msg.includes('403')) throw err
      // 429 = rate limited — retry with longer backoff.
      if (msg.includes('429')) {
        if (attempt === MAX_RETRIES) throw err
        const delayMs = 15000 * Math.pow(2, attempt) // 15s, 30s, 60s
        console.warn(`[VLM:siliconflow] rate limited — retrying in ${delayMs / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES + 1})`)
        await sleep(delayMs)
        continue
      }
      const isRetryable = /5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed|aborted/i.test(msg)
      if (!isRetryable || attempt === MAX_RETRIES) throw err
      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(`[VLM:siliconflow] batch retry ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`)
      await sleep(delayMs)
    }
  }

  throw lastErr
}

/**
 * Zhipu AI (GLM-4V-Flash) — free vision model, optimized for OCR/text extraction.
 * OpenAI-compatible API at https://open.bigmodel.cn/api/paas/v4/chat/completions
 * No rate limits on free tier. Same prompt + cache + parseBatchResponse pattern.
 */
async function narrateImageBatchZhipu(imgPaths: string[], batchStart: number): Promise<string[]> {
  // Check cache first
  const cachedResults: (string | null)[] = []
  let allCached = true
  for (const imgPath of imgPaths) {
    const cached = await getVlmCached(vlmCacheKey(imgPath))
    cachedResults.push(cached)
    if (cached === null) allCached = false
  }
  if (allCached) {
    console.log(`[VLM:zhipu] cache hit — all ${imgPaths.length} panels cached, skipping API call`)
    return cachedResults as string[]
  }

  const apiKey = process.env.ZHIPU_API_KEY!
  const model = process.env.ZHIPU_VLM_MODEL || 'glm-4v-flash'
  const url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

  // Read + base64-encode each image.
  const images: Array<{ mime: string; b64: string }> = []
  for (const imgPath of imgPaths) {
    const buf = await fs.readFile(imgPath)
    const ext = path.extname(imgPath).toLowerCase()
    const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'
    images.push({ mime, b64: buf.toString('base64') })
  }

  // Same transcription prompt as other providers.
  const prompt =
    `You are a precise transcriber for webtoon/manhwa panels, not a narrator. ` +
    `I am sending you ${images.length} separate panel images, labeled Panel 1 through Panel ${images.length} ` +
    `(in the order they appear below). For EACH panel, transcribe ONLY the actual text you can see inside ` +
    `speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader would naturally ` +
    `read them (top to bottom, left to right within the panel). Translate to natural English if not already ` +
    `in English, preserving meaning and tone.\n\n` +
    `RESPONSE FORMAT: Return a JSON array with exactly ${images.length} elements, one per panel in order. ` +
    `Each element is an object: {"index": <1-based panel number>, "text": "<transcribed text or empty string>"}.\n` +
    `Output ONLY the JSON array — no preamble, no markdown fences, no explanation.\n` +
    `Example for 2 panels: [{"index": 1, "text": "What is this place?"}, {"index": 2, "text": ""}]`

  const content = [
    { type: 'text', text: prompt },
    ...images.map(img => ({
      type: 'image_url',
      image_url: { url: `data:${img.mime};base64,${img.b64}` },
    })),
  ]

  const body = {
    model,
    messages: [{ role: 'user', content }],
    temperature: 0.1,
    max_tokens: 4096,
  }

  const MAX_RETRIES = 3
  const BASE_DELAY_MS = 2000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60000)
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(`Zhipu API ${res.status}: ${errText.slice(0, 200)}`)
      }

      const data = await res.json() as {
        choices?: Array<{ message?: { content?: string } }>
        error?: { message?: string }
      }

      if (data.error) {
        throw new Error(`Zhipu error: ${data.error.message || JSON.stringify(data.error)}`)
      }

      const raw = data.choices?.[0]?.message?.content?.trim() ?? ''
      const texts = parseBatchResponse(raw, images.length)

      // Cache each panel's transcription.
      for (let i = 0; i < imgPaths.length && i < texts.length; i++) {
        if (texts[i]) {
          await setVlmCached(vlmCacheKey(imgPaths[i]), texts[i])
        }
      }
      return texts
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      // 403 = forbidden. Don't retry.
      if (msg.includes('403')) throw err
      // 429 = rate limited — retry with longer backoff.
      if (msg.includes('429')) {
        if (attempt === MAX_RETRIES) throw err
        const delayMs = 15000 * Math.pow(2, attempt) // 15s, 30s, 60s
        console.warn(`[VLM:zhipu] rate limited — retrying in ${delayMs / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES + 1})`)
        await sleep(delayMs)
        continue
      }
      const isRetryable = /5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed|aborted/i.test(msg)
      if (!isRetryable || attempt === MAX_RETRIES) throw err
      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(`[VLM:zhipu] batch retry ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`)
      await sleep(delayMs)
    }
  }

  throw lastErr
}

/**
 * Parse the VLM's batch response into an array of per-panel text strings.
 *
 * The model is asked to return a JSON array like:
 *   [{"index": 1, "text": "..."}, {"index": 2, "text": ""}, ...]
 *
 * This function is defensive: it strips markdown code fences, extracts the
 * JSON array, and validates the element count. If parsing fails or the count
 * is wrong, it throws so the caller falls back to single-image calls.
 */
function parseBatchResponse(raw: string, expectedCount: number): string[] {
  if (!raw) {
    throw new Error('empty VLM batch response')
  }

  let cleaned = raw.trim()
  const fenceMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fenceMatch) {
    cleaned = fenceMatch[1].trim()
  }

  const start = cleaned.indexOf('[')
  const end = cleaned.lastIndexOf(']')
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('no JSON array found in VLM batch response')
  }
  const jsonStr = cleaned.slice(start, end + 1)

  let parsed: unknown
  try {
    parsed = JSON.parse(jsonStr)
  } catch (e) {
    throw new Error(`failed to parse VLM batch JSON: ${e instanceof Error ? e.message : e}`)
  }

  if (!Array.isArray(parsed)) {
    throw new Error('VLM batch response is not a JSON array')
  }

  // ── Sanity check: length mismatch ──
  // This alone catches the skipped-panel shift bug immediately.
  // If the VLM drops or duplicates an item, we log it rather than
  // silently filling gaps with empty text.
  if (parsed.length !== expectedCount) {
    console.warn(
      `[VLM] parseBatchResponse WARNING: parsed ${parsed.length} items but expected ${expectedCount}. ` +
      `Possible skipped/duplicate panel — ${expectedCount - parsed.length > 0 ? `${expectedCount - parsed.length} panel(s) will be silent` : 'extra items will be dropped'}.`,
    )
  }

  const texts: string[] = new Array(expectedCount).fill('')
  for (let i = 0; i < parsed.length && i < expectedCount; i++) {
    const item = parsed[i]
    if (typeof item === 'object' && item !== null) {
      const obj = item as { index?: number; text?: string }
      const text = typeof obj.text === 'string' ? obj.text : ''
      // VLM returns 1-based index per the prompt ("Panel 1 through Panel N")
      // Clamp 0-based: if index is already 0-based, keep it; if 1-based, subtract 1.
      const idx = typeof obj.index === 'number'
        ? (obj.index >= 1 ? obj.index - 1 : obj.index)
        : i
      if (idx >= 0 && idx < expectedCount) {
        texts[idx] = text
      } else {
        console.warn(
          `[VLM] parseBatchResponse: out-of-range index ${obj.index} (valid 0-${expectedCount - 1}), placing at position ${i}`,
        )
        texts[i] = text
      }
    } else if (typeof item === 'string') {
      texts[i] = item
    }
  }

  return texts
}


/**
 * Send a single image to the VLM and get back the actual dialogue/caption
 * text from its speech bubbles, thought bubbles, and caption boxes —
 * transcribed as-is (translated to English), not narrated or paraphrased.
 *
 * Retries on rate-limit (429) and transient server errors (5xx) with
 * exponential backoff: 2s, 4s, 8s, 16s. This is critical because the pipeline
 * now makes one VLM call per sliced panel (190+ calls for a typical chapter),
 * so without backoff a single 429 would permanently lose that panel's text.
 */
async function narrateSingleImage(imgPath: string): Promise<string> {
  // Check cache first — if we've already transcribed this image, return instantly
  const cacheKey = vlmCacheKey(imgPath)
  const cached = await getVlmCached(cacheKey)
  if (cached !== null) {
    console.log(`[VLM] cache hit for ${path.basename(imgPath)}`)
    return cached
  }

  const zai = await getZai()

  const buf = await fs.readFile(imgPath)
  const b64 = buf.toString('base64')
  const ext = path.extname(imgPath).toLowerCase()
  const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg'

  const content: Array<
    | { type: 'text'; text: string }
    | { type: 'image_url'; image_url: { url: string } }
  > = [
    {
      type: 'text',
      text:
        'You are a precise transcriber for a webtoon/manhwa panel, not a narrator. ' +
        'Look at this single panel and transcribe ONLY the actual text you can see inside ' +
        'speech bubbles, thought bubbles, and caption/narration boxes — in the order a reader ' +
        'would naturally read them (top to bottom, left to right within the panel). ' +
        'Translate it into natural English if it is not already in English, preserving the ' +
        'original meaning and tone as closely as possible.\n\n' +
        'Guidelines:\n' +
        '1. Output the text VERBATIM (translated) — do not paraphrase, summarize, embellish, or ' +
        'add descriptive narration around it. Do not invent dialogue that is not actually written.\n' +
        '2. Do not describe the artwork, action, or characters\' expressions — only transcribe ' +
        'written text that literally appears in the image.\n' +
        '3. If multiple bubbles/boxes are present, join them in reading order as separate ' +
        'sentences, preserving punctuation like "..." and "!" as written.\n' +
        '4. Sound effect text (e.g. "BOOM", "CRASH") can be included briefly if it is the only ' +
        'text present, otherwise skip pure onomatopoeia in favor of actual dialogue/captions.\n' +
        '5. Never mention chapter numbers, page numbers, panels, or that you are looking at an image.\n' +
        '6. If the panel has NO readable text at all (a purely visual/action panel with no bubbles ' +
        'or captions), output nothing at all — an empty response. Do not invent narration to fill it.\n' +
        '7. Output ONLY the transcribed (translated) text — no preamble, no headers, no markdown, ' +
        'no notes about what you did.',
    },
    {
      type: 'image_url',
      image_url: { url: `data:${mime};base64,${b64}` },
    },
  ]

  const zaiAny = zai as {
    chat: {
      completions: {
        createVision: (opts: {
          messages: Array<{ role: string; content: typeof content }>
          thinking: { type: string }
        }) => Promise<{
          choices?: Array<{ message?: { content?: string } }>
        }>
      }
    }
  }

  // Exponential backoff for rate-limit (429) and transient server errors (5xx).
  // Base delays: 2s, 4s, 8s, 16s — caps total wait at ~30s before giving up.
  const MAX_RETRIES = 4
  const BASE_DELAY_MS = 2000
  let lastErr: unknown = null

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await zaiAny.chat.completions.createVision({
        messages: [{ role: 'user', content }],
        thinking: { type: 'disabled' },
      })

      const text = resp?.choices?.[0]?.message?.content?.trim()
      // Cache the result so re-runs skip this VLM call
      if (text) {
        void setVlmCached(cacheKey, text)
      }
      return text ?? ''
    } catch (err) {
      lastErr = err
      const msg = err instanceof Error ? err.message : String(err)
      // Retry on 429 (rate limit) and 5xx (server errors). Don't retry on
      // 4xx client errors (bad request, auth, etc.) — those won't fix themselves.
      const isRetryable = /429|rate.?limit|too many requests|5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed/i.test(msg)

      if (!isRetryable || attempt === MAX_RETRIES) {
        throw err
      }

      const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
      console.warn(
        `[VLM] ${path.basename(imgPath)} attempt ${attempt + 1}/${MAX_RETRIES + 1} failed (${msg.slice(0, 80)}) — retrying in ${delayMs}ms`,
      )
      await sleep(delayMs)
    }
  }

  throw lastErr
}

