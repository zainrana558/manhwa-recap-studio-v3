import { getPrivateExt } from "@/lib/private-loader";
import { NextRequest, NextResponse } from "next/server";
import { searchAllManga, searchSingleSource } from "@/lib/manga-search";

export const dynamic = "force-dynamic";

// Increase the max duration for this API route since it calls multiple external sources.
export const maxDuration = 30;

const VALID_SOURCES = ["asurascans", "thunderscans", "vortexscans", "rizzfables", "webtoons", "mgeko", "mangadex", "mangahere", "fanfox", "mangapill", "comick", "weebcentral", "mal", "anilist"] as const;
type ValidSource = (typeof VALID_SOURCES)[number];

function isSource(s: string): s is ValidSource {
  return (VALID_SOURCES as readonly string[]).includes(s);
}

function emptySources(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of VALID_SOURCES) out[s] = 0;
  return out;
}

/**
 * GET /api/search?q={query}&limit={limit}&source={mangahere|fanfox|webtoons|mal|anilist}
 *
 * - No `source`: queries all 5 sources in parallel, dedupes, returns sources counts.
 * - `source=...`: queries just that one source.
 *
 * Returns: { manga: MangadexManga[], total: number, sources: { mangahere, fanfox, webtoons, mal, anilist } }
 *
 * IMPORTANT: This route NEVER returns 502. If sources are unreachable, it returns
 * 200 with an empty manga array and a warning message, so the UI can gracefully
 * show "No results" instead of a hard error.
 */
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const q = (searchParams.get("q") ?? "").trim();
    const limitParam = searchParams.get("limit");
    const limit = limitParam ? Math.max(1, parseInt(limitParam, 10) || 12) : 12;
    const sourceParam = (searchParams.get("source") ?? "").trim().toLowerCase();
    const adult = searchParams.get("adult") === "1";
    const genre = (searchParams.get("genre") ?? "").trim();



    const ext = getPrivateExt();
    if (!adult && q && ext?.isUnlockCode(q)) {
      return NextResponse.json({ unlockPrivate: true, ...ext.getPrivateConfig() });
    }

    if (!q && !(adult && genre)) {
      return NextResponse.json(
        { error: "Query parameter 'q' is required." },
        { status: 400 }
      );
    }

    // Single-source mode.
    if (isSource(sourceParam)) {
      try {
        const rawSingle = await searchSingleSource(q, sourceParam, limit);
        const manga = filterSafeNormalResults(rawSingle, adult);
        const sources: Record<string, number> = {};
        for (const s of VALID_SOURCES) {
          sources[s] = s === sourceParam ? manga.length : 0;
        }
        return NextResponse.json({ manga, total: manga.length, sources });
      } catch {
        // Single source failed — return empty instead of 502
        return NextResponse.json({
          manga: [],
          total: 0,
          sources: emptySources(),
          warning: `${sourceParam} search is currently unavailable. Try again later.`,
        });
      }
    }

    // All-source mode — NEVER return 502, always return 200 with whatever results we got.
    try {
      const { manga: rawManga, sources: rawSources } = await searchAllManga(q, limit, { adult, genre });
      const noBrokenSources = rawManga.filter((m) => m.source !== "mangapill");
      const safeManga = filterSafeNormalResults(noBrokenSources, adult);
      const manga = enrichBrokenCovers(safeManga);
      const sources: Record<string, number> = { ...rawSources };
      if (!adult) {
        for (const k of Object.keys(sources)) sources[k] = 0;
        for (const item of manga) {
          const s = item.source ?? "mangahere";
          sources[s] = (sources[s] ?? 0) + 1;
        }
      }
      return NextResponse.json({
        manga,
        total: manga.length,
        sources,
      });
    } catch {
      // All sources failed — return empty instead of 502
      return NextResponse.json({
        manga: [],
        total: 0,
        sources: emptySources(),
        warning: "All search sources are currently unreachable. Try again later.",
      });
    }
  } catch {
    // Outermost catch — still return 200 with empty
    return NextResponse.json({
      manga: [],
      total: 0,
      sources: emptySources(),
      warning: "Search temporarily unavailable. Try again later.",
    });
  }
}


function enrichBrokenCovers<T extends { title: string; coverUrl?: string | null; lastChapter?: string | null; source?: string }>(items: T[]): T[] {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const isBlockedCover = (u?: string | null) => !u || u.includes("uploads.mangadex.org");
  const parseCh = (c?: string | null) => {
    if (!c) return 0;
    const n = parseFloat(String(c));
    return Number.isFinite(n) && n > 0 ? n : 0;
  };

  const goodByTitle = new Map<string, string>();
  const goodList: Array<{ key: string; url: string }> = [];
  const bestChByTitle = new Map<string, string>();

  for (const m of items) {
    const k = norm(m.title);
    if (!isBlockedCover(m.coverUrl) && !goodByTitle.has(k)) {
      goodByTitle.set(k, m.coverUrl!);
      goodList.push({ key: k, url: m.coverUrl! });
    }
    const chVal = parseCh(m.lastChapter);
    if (chVal > 0) {
      const prev = parseCh(bestChByTitle.get(k));
      if (chVal > prev) bestChByTitle.set(k, String(m.lastChapter));
    }
  }

  return items.map((m) => {
    const k = norm(m.title);
    let coverUrl = m.coverUrl;
    if (isBlockedCover(coverUrl)) {
      const exact = goodByTitle.get(k);
      if (exact) coverUrl = exact;
      else {
        const fuzzy = goodList.find((g) => g.key.includes(k) || k.includes(g.key));
        if (fuzzy) coverUrl = fuzzy.url;
      }
    }

    let lastChapter = parseCh(m.lastChapter) > 0 ? m.lastChapter : null;
    if (!lastChapter) {
      lastChapter = bestChByTitle.get(k) ?? null;
    }

    if (coverUrl !== m.coverUrl || lastChapter !== m.lastChapter) {
      return { ...m, coverUrl, lastChapter };
    }
    return m;
  });
}

const ADULT_RATINGS = new Set([
  "pornographic",
  "erotica",
  "adult",
  "nsfw",
  "18+",
  "rx",
  "hentai",
  "suggestive+",
]);

const ADULT_TAG_PATTERNS = [
  /^adult$/i,
  /^smut$/i,
  /^hentai$/i,
  /^erotica$/i,
  /^ecchi$/i,
  /^pornographic$/i,
  /^pornhwa$/i,
  /^nsfw$/i,
  /^18\+$/i,
  /^r-?18$/i,
  /^nc-?17$/i,
  /^doujinshi$/i,
  /^h-?manga$/i,
  /^netorare$/i,
  /^ntr$/i,
  /^incest$/i,
  /^milf$/i,
  /^bara$/i,
  /^yaoi$/i,
  /^yuri$/i,
  /sexual\s*violence/i,
  /explicit/i,
];

const ADULT_TEXT_REGEX = /(?:\b(?:hentai|pornhwa|porn|smut|nsfw|r-?18|nc-?17|erotica|erotic|pornographic|uncensored|adult\s+only|netorare|xxx|sex\s+friend|sex\s+study)\b|\(\s*18\+\s*\)|\[\s*18\+\s*\])/i;

function isAdultContent(m: {
  title?: string;
  description?: string | null;
  contentRating?: string | null;
  tags?: string[] | null;
  genres?: string[] | null;
}): boolean {
  const rating = (m.contentRating ?? "").toLowerCase().trim();
  if (ADULT_RATINGS.has(rating)) return true;

  const allTags = [...(m.tags ?? []), ...(m.genres ?? [])];
  for (const tag of allTags) {
    const t = String(tag ?? "").trim();
    if (ADULT_TAG_PATTERNS.some((rx) => rx.test(t))) return true;
  }

  if (m.title && ADULT_TEXT_REGEX.test(m.title)) return true;
  if (m.description && ADULT_TEXT_REGEX.test(m.description)) return true;

  return false;
}

function filterSafeNormalResults<T extends {
  title?: string;
  description?: string | null;
  contentRating?: string | null;
  tags?: string[] | null;
  genres?: string[] | null;
  source?: string;
}>(items: T[], adult: boolean): T[] {
  if (adult) return items;
  return items.filter((m) => !isAdultContent(m));
}
