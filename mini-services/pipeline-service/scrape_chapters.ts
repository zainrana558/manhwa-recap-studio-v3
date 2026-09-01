/**
 * Standalone chapter scraper (training data / bulk pulls).
 *   bun scrape_chapters.ts --source mangadex   --id <mangaId>  --from 1 --to 5   --out DIR
 *   bun scrape_chapters.ts --source asurascans --id solo-leveling --count 10     --out DIR
 *   bun scrape_chapters.ts --source weebcentral --id 01J76XYCPSY3C4BNPBRY8JMCBE --from 1 --to 5 --out DIR
 *   bun scrape_chapters.ts --search "solo leveling" --source weebcentral   (just prints matches)
 *
 * --id is whatever that source's fetch* takes (MangaDex UUID, Asura/Comick slug,
 * WeebCentral series ULID). Prefixes (md-/as-/wc-/…) are stripped automatically.
 */
import * as fs from 'fs/promises'
import * as path from 'path'
import {
  type ScraperSource,
  fetchChaptersForSource,
  fetchImagesForSource,
  downloadImageForSource,
  searchWeebCentral,
} from './lib'

function arg(name: string, def = ''): string {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def
}

async function main() {
  const source = arg('source', 'mangadex') as ScraperSource
  const search = arg('search')
  if (search) {
    if (source !== 'weebcentral') throw new Error('--search only wired for weebcentral')
    for (const r of await searchWeebCentral(search, 10)) console.log(`  ${r.id}\t${r.title}`)
    return
  }

  const from = parseFloat(arg('from', '1'))
  const to = parseFloat(arg('to', '5'))
  const count = parseInt(arg('count', '0'), 10)
  const out = arg('out')
  const id = arg('id') || arg('slug')
  const delayMs = parseInt(arg('delay', '150'), 10)
  if (!out || !id) throw new Error('--out and --id required')
  await fs.mkdir(out, { recursive: true })

  const chapters = await fetchChaptersForSource(source, id, 0)
  console.log(`${chapters.length} chapters available`)

  const seen = new Set<number>()
  let want = chapters
    .map((c, idx) => ({ ...c, n: c.chapterNum ? parseFloat(c.chapterNum) : idx + 1 }))
    .filter((c) => (seen.has(c.n) ? false : (seen.add(c.n), true)))
  want = count > 0 ? want.slice(0, count) : want.filter((c) => c.n >= from && c.n <= to)
  console.log(`downloading ${want.length}`)

  for (const c of want) {
    const dir = path.join(out, `chapter_${String(Math.round(c.n)).padStart(3, '0')}`)
    await fs.mkdir(dir, { recursive: true })
    let urls: string[]
    try {
      urls = await fetchImagesForSource(source, id, c.mangadexId)
    } catch (e) {
      console.log(`  ch ${c.n}: list failed — ${e}`)
      continue
    }
    let ok = 0
    for (let i = 0; i < urls.length; i++) {
      const ext = (urls[i].split('?')[0].split('.').pop() || 'jpg').slice(0, 4)
      const dest = path.join(dir, `${String(i + 1).padStart(3, '0')}.${ext}`)
      try {
        await downloadImageForSource(source, urls[i], dest)
        ok++
      } catch (e) {
        console.log(`    page ${i + 1} failed: ${e}`)
      }
      await new Promise((r) => setTimeout(r, delayMs))
    }
    console.log(`  ch ${c.n}: ${ok}/${urls.length} pages -> ${dir}`)
  }
  console.log('done')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
