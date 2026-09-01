/**
 * Standalone chapter scraper for building training data.
 *   bun scrape_chapters.ts --source mangadex --id <mangaId> --from 1 --to 5 --out DIR
 *   bun scrape_chapters.ts --source asurascans --slug solo-leveling --from 1 --to 5 --out DIR
 */
import * as fs from 'fs/promises'
import * as path from 'path'
import {
  fetchMangaDexChapters,
  fetchMangaDexChapterImages,
  downloadMangaDexImage,
  fetchAsuraScansChapters,
  fetchAsuraScansChapterImages,
  downloadAsuraScansImage,
} from './lib'

function arg(name: string, def = ''): string {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def
}

async function main() {
  const source = arg('source', 'mangadex')
  const from = parseInt(arg('from', '1'), 10)
  const to = parseInt(arg('to', '5'), 10)
  const out = arg('out')
  const id = arg('id')
  const slug = arg('slug')
  if (!out) throw new Error('--out required')
  await fs.mkdir(out, { recursive: true })

  let chapters: Array<{ mangadexId: string; chapterNum: string | null }> = []
  if (source === 'mangadex') {
    chapters = await fetchMangaDexChapters(id, 0)
  } else if (source === 'asurascans') {
    chapters = (await fetchAsuraScansChapters(slug, 0)) as typeof chapters
  } else {
    throw new Error(`unsupported source ${source}`)
  }
  console.log(`${chapters.length} chapters available`)

  const count = parseInt(arg('count', '0'), 10) // if set: take first N available, ignore --from/--to
  const seen = new Set<number>()
  let want = chapters
    .map((c, idx) => ({ ...c, n: c.chapterNum ? parseFloat(c.chapterNum) : idx + 1 }))
    .filter((c) => (seen.has(c.n) ? false : (seen.add(c.n), true))) // first upload per ch number
  want = count > 0 ? want.slice(0, count) : want.filter((c) => c.n >= from && c.n <= to)
  console.log(`downloading ${want.length} (ch ${from}-${to})`)

  for (const c of want) {
    const dir = path.join(out, `chapter_${String(Math.round(c.n)).padStart(3, '0')}`)
    await fs.mkdir(dir, { recursive: true })
    let urls: string[]
    try {
      urls =
        source === 'mangadex'
          ? await fetchMangaDexChapterImages(id, c.mangadexId)
          : await fetchAsuraScansChapterImages(slug, c.mangadexId)
    } catch (e) {
      console.log(`  ch ${c.n}: list failed — ${e}`)
      continue
    }
    let ok = 0
    for (let i = 0; i < urls.length; i++) {
      const dest = path.join(dir, `${String(i + 1).padStart(3, '0')}.${urls[i].split('.').pop()!.split('?')[0]}`)
      try {
        if (source === 'mangadex') await downloadMangaDexImage(urls[i], dest)
        else await downloadAsuraScansImage(urls[i], dest)
        ok++
      } catch (e) {
        console.log(`    page ${i + 1} failed: ${e}`)
      }
      await new Promise((r) => setTimeout(r, 150))
    }
    console.log(`  ch ${c.n}: ${ok}/${urls.length} pages -> ${dir}`)
  }
  console.log('done')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
