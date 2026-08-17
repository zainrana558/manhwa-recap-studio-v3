/**
 * vlm-worker.ts — Standalone VLM transcription worker.
 *
 * This script is spawned by index.ts as a child process for each batch of images.
 * It handles the z-ai SDK call in isolation.
 *
 * Input:  JSON on stdin  { images: string[], prompt: string }
 * Output: JSON on stdout { results: string[] }
 */
import { config as loadDotenv } from 'dotenv'
import { resolve } from 'path'
loadDotenv({ path: resolve(import.meta.dirname, '../..', '.env'), override: true })

import { promises as fs } from 'fs'
import path from 'path'
import crypto from 'crypto'

const VLM_CACHE_DIR = path.resolve(import.meta.dirname, '../../data/cache/vlm')

function vlmCacheKey(imagePath: string): string {
  return crypto.createHash('sha256').update(imagePath).digest('hex').slice(0, 16)
}

async function getVlmCached(key: string): Promise<string | null> {
  try {
    const cacheFile = path.join(VLM_CACHE_DIR, `${key}.json`)
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
    // ignore cache write errors
  }
}

function parseBatchResponse(raw: string, expectedCount: number): string[] {
  const jsonMatch = raw.match(/\[[\s\S]*?\]/)
  if (!jsonMatch) {
    process.stderr.write(`[vlm-worker] no JSON array in response (expected ${expectedCount})\n`)
    return Array(expectedCount).fill('')
  }
  try {
    const parsed = JSON.parse(jsonMatch[0])
    if (!Array.isArray(parsed)) return Array(expectedCount).fill('')

    // ── Sanity check: length mismatch ──
    if (parsed.length !== expectedCount) {
      process.stderr.write(
        `[vlm-worker] WARNING: parsed ${parsed.length} items but expected ${expectedCount}. ` +
        `Possible skipped panel — filling gaps with empty text.\n`,
      )
    }

    // ── Index-aware placement ──
    // The VLM prompt requests {"index": 1, "text": "..."} (1-based).
    // Map by index when available, fall back to array position.
    // BUG FIX: previous version ignored index entirely, causing silent
    // panel shifts when the VLM returned items out of order.
    const texts: string[] = new Array(expectedCount).fill('')
    for (let i = 0; i < parsed.length && i < expectedCount; i++) {
      const item = parsed[i]
      if (typeof item === 'string') {
        texts[i] = item
      } else if (typeof item === 'object' && item !== null) {
        const text = typeof item.text === 'string' ? item.text : ''
        // VLM returns 1-based index per the prompt ("Panel 1 through Panel N")
        if (typeof item.index === 'number') {
          const idx = item.index >= 1 ? item.index - 1 : item.index
          if (idx >= 0 && idx < expectedCount) {
            texts[idx] = text
          } else {
            process.stderr.write(
              `[vlm-worker] WARNING: out-of-range index ${item.index} (valid: 0-${expectedCount - 1}), placing at position ${i}\n`,
            )
            texts[i] = text
          }
        } else {
          texts[i] = text
        }
      }
    }
    return texts
  } catch {
    return Array(expectedCount).fill('')
  }
}

async function main() {
  let input = ''
  for await (const chunk of process.stdin) {
    input += chunk
  }
  const { images, prompt } = JSON.parse(input) as { images: string[]; prompt: string }

  // Check cache for all images.
  const results: string[] = []
  const uncachedIndices: number[] = []
  const uncachedPaths: string[] = []

  for (let i = 0; i < images.length; i++) {
    const cached = await getVlmCached(vlmCacheKey(images[i]))
    if (cached !== null) {
      results[i] = cached
    } else {
      results[i] = ''
      uncachedIndices.push(i)
      uncachedPaths.push(images[i])
    }
  }

  if (uncachedPaths.length === 0) {
    process.stdout.write(JSON.stringify({ results }) + '\n')
    return
  }

  // Quick pre-flight: test if z-ai createVision actually works.
  // In some sandboxed environments, the SDK is present but calls get killed.
  try {
    const { default: ZAI } = await import('z-ai-web-dev-sdk')
    const zai = await ZAI.create()
    // Test with a tiny request first (1 small image).
    const testBuf = await fs.readFile(uncachedPaths[0])
    const testB64 = testBuf.toString('base64').slice(0, 100) + '...truncated'
    // We can't really do a tiny test — just try the real thing.
    // If it gets killed, the parent will see exit code != 0 and use empty text.

    const imgData: Array<{ name: string; b64: string }> = []
    for (const imgPath of uncachedPaths) {
      const buf = await fs.readFile(imgPath)
      imgData.push({ name: path.basename(imgPath), b64: buf.toString('base64') })
    }

    const content: Array<
      | { type: 'text'; text: string }
      | { type: 'image_url'; image_url: { url: string } }
    > = [
      { type: 'text', text: prompt.replace(/{count}/g, String(imgData.length)) },
      ...imgData.map((img) => ({
        type: 'image_url' as const,
        image_url: { url: `data:image/jpeg;base64,${img.b64}` },
      })),
    ]

    const MAX_RETRIES = 4
    const BASE_DELAY_MS = 2000
    let lastErr: unknown = null

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const zaiAny = zai as any
        const resp = await zaiAny.chat.completions.createVision({
          messages: [{ role: 'user', content }],
          thinking: { type: 'disabled' },
        })

        const raw = resp?.choices?.[0]?.message?.content?.trim() ?? ''
        const texts = parseBatchResponse(raw, imgData.length)

        for (let i = 0; i < uncachedIndices.length; i++) {
          results[uncachedIndices[i]] = texts[i] ?? ''
          if (texts[i]) {
            await setVlmCached(vlmCacheKey(uncachedPaths[i]), texts[i])
          }
        }

        process.stdout.write(JSON.stringify({ results }) + '\n')
        return
      } catch (err) {
        lastErr = err
        const msg = err instanceof Error ? err.message : String(err)
        const isRetryable = /429|rate.?limit|too many requests|5\d{2}|server error|timeout|econnreset|socket hang up|fetch failed/i.test(msg)
        if (!isRetryable || attempt === MAX_RETRIES) {
          break
        }
        const delayMs = BASE_DELAY_MS * Math.pow(2, attempt)
        process.stderr.write(`[vlm-worker] retry ${attempt + 1}/${MAX_RETRIES} after ${delayMs}ms: ${msg.slice(0, 80)}\n`)
        await new Promise((r) => setTimeout(r, delayMs))
      }
    }
  } catch (err) {
    process.stderr.write(`[vlm-worker] z-ai init/call failed: ${err instanceof Error ? err.message : String(err)}\n`)
  }

  // All retries exhausted or z-ai unavailable — return empty strings for uncached.
  process.stderr.write(`[vlm-worker] returning empty text for ${uncachedPaths.length} uncached panels\n`)
  process.stdout.write(JSON.stringify({ results }) + '\n')
}

main().catch((err) => {
  process.stderr.write(`[vlm-worker] fatal: ${err}\n`)
  process.exit(1)
})
