/**
 * Multi-source manga scraper — 3 scraping APIs + 2 metadata APIs.
 *
 * Scraping sources (provide chapter images):
 *  1. MangaHere (mangahere.cc) — manga + manhwa, obfuscated JS image URLs
 *  2. FanFox (fanfox.net) — same CMS as MangaHere, different CDN (fmcdn.mfcdn.net)
 *  3. Webtoons (webtoons.com) — official manhwa/webtoons, direct img tags
 *
 * Metadata sources (search only, resolve to scraping sources by title):
 *  - Jikan (MyAnimeList)
 *  - AniList
 *
 * Each manga result is tagged with `source: "mangahere" | "fanfox" | "webtoons" | "mal" | "anilist"`.
 * The pipeline service uses the source to determine which scraper to use.
 */

import type { MangadexManga } from "@/types/pipeline";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const FETCH_TIMEOUT_MS = 5_000; // 5s timeout for all external fetches

async function fetchWithTimeout(
  url: string,
  init?: RequestInit,
  timeoutMs: number = FETCH_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface ScrapedChapter {
  id: string; // chapter slug/id used by the source
  chapterNum: string;
  title: string | null;
  /** Translated language of this chapter, when known (e.g. "en"). Defaults to "en" if omitted. */
  language?: string;
}

export interface ScrapedImage {
  url: string;
  referer: string; // required Referer header for this CDN
  /** Suggested local filename (with extension) when saving this page to disk. */
  filename?: string;
  /** Extra request headers beyond Referer, if the CDN needs them (e.g. User-Agent). */
  headers?: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Source 1: MangaHere (mangahere.cc)
// ---------------------------------------------------------------------------

const MANGAHERE_BASE = "https://www.mangahere.cc";
const MANGAHERE_CDN = "https://zjcdn.mangahere.org";

export async function searchMangaHere(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const res = await fetchWithTimeout(
    `${MANGAHERE_BASE}/search.php?name=${encodeURIComponent(query)}`,
    { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" } }
  );
  if (!res.ok) throw new Error(`MangaHere search ${res.status}`);
  const html = await res.text();

  const slugSet = new Set<string>();
  const matches = html.matchAll(/href="\/manga\/([a-z0-9_]+)\/"/gi);
  for (const m of matches) {
    if (!/^c\d/.test(m[1])) slugSet.add(m[1]);
  }

  return Array.from(slugSet).slice(0, limit).map((slug) => ({
    id: `mh-${slug}`,
    title: slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    description: "",
    coverUrl: null,
    status: null,
    year: null,
    originalLanguage: null,
    availableTranslatedLanguages: [],
    tags: [],
    contentRating: null,
    lastChapter: null,
    source: "mangahere" as const,
    externalUrl: `${MANGAHERE_BASE}/manga/${slug}/`,
  }));
}

export async function getMangaHereChapters(slug: string): Promise<ScrapedChapter[]> {
  const res = await fetchWithTimeout(`${MANGAHERE_BASE}/manga/${slug}/`, {
    headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" },
  });
  if (!res.ok) throw new Error(`MangaHere chapters ${res.status}`);
  const html = await res.text();

  const chapters: ScrapedChapter[] = [];
  const seen = new Set<string>();
  // URLs have optional volume prefix: /manga/{slug}/v01/c000/1.html or /manga/{slug}/c000/1.html
  // Capture the full path (volume + chapter) as the chapter ID for correct image URL construction
  const regex = new RegExp(`href="/manga/${slug}/((?:v\\d+/)?c[0-9.]+)/1\\.html"`, "gi");
  let match;
  while ((match = regex.exec(html)) !== null) {
    const fullPath = match[1]; // e.g. "v72/c700" or "c700"
    if (seen.has(fullPath)) continue;
    seen.add(fullPath);
    // Extract chapter number from the c{num} part
    const chapterMatch = fullPath.match(/c([0-9.]+)/);
    const chapterNum = chapterMatch ? (chapterMatch[1].replace(/^0+/, "") || "0") : "0";
    chapters.push({
      id: fullPath,
      chapterNum,
      title: null,
    });
  }
  chapters.reverse();
  return chapters;
}

export async function getMangaHereImages(
  slug: string,
  chapterSlug: string
): Promise<ScrapedImage[]> {
  const res = await fetchWithTimeout(
    `${MANGAHERE_BASE}/manga/${slug}/${chapterSlug}/1.html`,
    { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" } }
  );
  if (!res.ok) throw new Error(`MangaHere chapter ${res.status}`);
  const html = await res.text();

  const storeMatch = html.match(/store\/manga\/(\d+)/);
  if (!storeMatch) throw new Error("Could not extract store ID");
  const storeId = storeMatch[1];
  const chapterFolder = chapterSlug.match(/c([0-9.]+)/)
    ? chapterSlug.match(/c([0-9.]+)/)![1].padStart(3, "0")
    : chapterSlug.replace(/^c/, "").padStart(3, "0");

  const filenames = new Set<string>();
  let m;
  const re = /([a-z]\d{8}_\d{6}_[a-z0-9]+)/gi;
  while ((m = re.exec(html)) !== null) filenames.add(m[1]);

  return Array.from(filenames).map((fn) => ({
    url: `${MANGAHERE_CDN}/store/manga/${storeId}/${chapterFolder}.0/compressed/${fn}.jpg`,
    referer: `${MANGAHERE_BASE}/`,
  }));
}

// ---------------------------------------------------------------------------
// Source 2: FanFox (fanfox.net) — same CMS as MangaHere, different CDN
// ---------------------------------------------------------------------------

const FANFOX_BASE = "https://fanfox.net";
const FANFOX_CDN = "https://fmcdn.mfcdn.net";

export async function searchFanFox(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const res = await fetchWithTimeout(
    `${FANFOX_BASE}/search?name=${encodeURIComponent(query)}`,
    { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" } }
  );
  if (!res.ok) throw new Error(`FanFox search ${res.status}`);
  const html = await res.text();

  const slugSet = new Set<string>();
  const matches = html.matchAll(/href="\/manga\/([a-z0-9_]+)\/"/gi);
  for (const m of matches) {
    if (!/^c\d/.test(m[1])) slugSet.add(m[1]);
  }

  return Array.from(slugSet).slice(0, limit).map((slug) => ({
    id: `ff-${slug}`,
    title: slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    description: "",
    coverUrl: null,
    status: null,
    year: null,
    originalLanguage: null,
    availableTranslatedLanguages: [],
    tags: [],
    contentRating: null,
    lastChapter: null,
    source: "fanfox" as const,
    externalUrl: `${FANFOX_BASE}/manga/${slug}/`,
  }));
}

export async function getFanFoxChapters(slug: string): Promise<ScrapedChapter[]> {
  const res = await fetchWithTimeout(`${FANFOX_BASE}/manga/${slug}/`, {
    headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" },
  });
  if (!res.ok) throw new Error(`FanFox chapters ${res.status}`);
  const html = await res.text();

  const chapters: ScrapedChapter[] = [];
  const seen = new Set<string>();
  // URLs have optional volume prefix: /manga/{slug}/v72/c700/1.html or /manga/{slug}/c700/1.html
  // Capture the full path (volume + chapter) as the chapter ID
  const regex = new RegExp(`href="/manga/${slug}/((?:v\\d+/)?c[0-9.]+)/1\\.html"`, "gi");
  let match;
  while ((match = regex.exec(html)) !== null) {
    const fullPath = match[1]; // e.g. "v72/c700" or "c700"
    if (seen.has(fullPath)) continue;
    seen.add(fullPath);
    const chapterMatch = fullPath.match(/c([0-9.]+)/);
    const chapterNum = chapterMatch ? (chapterMatch[1].replace(/^0+/, "") || "0") : "0";
    chapters.push({
      id: fullPath,
      chapterNum,
      title: null,
    });
  }
  chapters.reverse();
  return chapters;
}

export async function getFanFoxImages(
  slug: string,
  chapterSlug: string
): Promise<ScrapedImage[]> {
  const res = await fetchWithTimeout(
    `${FANFOX_BASE}/manga/${slug}/${chapterSlug}/1.html`,
    { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" } }
  );
  if (!res.ok) throw new Error(`FanFox chapter ${res.status}`);
  const html = await res.text();

  const storeMatch = html.match(/store\/manga\/(\d+)/);
  if (!storeMatch) throw new Error("Could not extract store ID");
  const storeId = storeMatch[1];
  const chapterFolder = chapterSlug.match(/c([0-9.]+)/)
    ? chapterSlug.match(/c([0-9.]+)/)![1].padStart(3, "0")
    : chapterSlug.replace(/^c/, "").padStart(3, "0");

  const filenames = new Set<string>();
  let m;
  const re = /([a-z]\d{8}_\d{6}_[a-z0-9]+)/gi;
  while ((m = re.exec(html)) !== null) filenames.add(m[1]);

  return Array.from(filenames).map((fn) => ({
    url: `${FANFOX_CDN}/store/manga/${storeId}/${chapterFolder}.0/compressed/${fn}.jpg`,
    referer: `${FANFOX_BASE}/`,
  }));
}

// ---------------------------------------------------------------------------
// Source 3: Webtoons (webtoons.com) — official manhwa/webtoons
// ---------------------------------------------------------------------------

const WEBTOONS_BASE = "https://www.webtoons.com";

interface WebtoonsTitle {
  titleNo: number;
  title: string;
  genre: string;
  url: string;
}

export async function searchWebtoons(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  // Webtoons search requires a headless browser or their API.
  // We use the public search endpoint that returns HTML.
  const res = await fetchWithTimeout(
    `${WEBTOONS_BASE}/en/search?keyword=${encodeURIComponent(query)}&searchType=ALL`,
    {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
      },
    }
  );
  if (!res.ok) throw new Error(`Webtoons search ${res.status}`);
  const html = await res.text();

  // Parse search results: links like /en/{genre}/{title-slug}/list?title_no={n}
  // The title text is in nested elements, so we extract the title from the URL slug.
  const titles: WebtoonsTitle[] = [];
  const seen = new Set<number>();
  const regex = /href="([^"]*\/en\/([^"]+)\/list\?title_no=(\d+))"/gi;
  let match;
  while ((match = regex.exec(html)) !== null) {
    const titleNo = parseInt(match[3], 10);
    if (seen.has(titleNo)) continue;
    seen.add(titleNo);
    // Extract title from the URL path: /en/{genre}/{title-slug}/list?title_no=N
    // match[2] is the genre/title-slug part, e.g. "fantasy/tower-of-god"
    const pathParts = match[2].split("/");
    const titleSlug = pathParts[pathParts.length - 1]; // e.g. "tower-of-god"
    const title = titleSlug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    if (!title) continue;
    titles.push({
      titleNo,
      title,
      genre: pathParts[0] ?? "",
      url: match[1].startsWith("http") ? match[1] : `${WEBTOONS_BASE}${match[1]}`,
    });
  }

  return titles.slice(0, limit).map((t) => ({
    id: `wt-${t.titleNo}`,
    title: t.title,
    description: "",
    coverUrl: null,
    status: "Ongoing",
    year: null,
    originalLanguage: "ko",
    availableTranslatedLanguages: ["en"],
    tags: [],
    contentRating: "safe",
    lastChapter: null,
    source: "webtoons" as const,
    externalUrl: t.url,
  }));
}

export async function getWebtoonsChapters(
  titleNo: number
): Promise<ScrapedChapter[]> {
  // We need to find the manga's list page URL first.
  // Webtoons episode list is at /en/{genre}/{title}/list?title_no={n}
  // We try fetching the list page directly.
  const res = await fetchWithTimeout(
    `${WEBTOONS_BASE}/en/fantasy/_/list?title_no=${titleNo}`,
    {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
    }
  );
  if (!res.ok) throw new Error(`Webtoons chapters ${res.status}`);
  const html = await res.text();

  // Parse episode links: /en/{genre}/{title}/{episode-slug}/viewer?title_no={n}&episode_no={m}
  const chapters: ScrapedChapter[] = [];
  const seen = new Set<number>();
  const regex = /href="([^"]*\/viewer\?title_no=\d+&episode_no=(\d+))"/gi;
  let match;
  while ((match = regex.exec(html)) !== null) {
    const epNo = parseInt(match[2], 10);
    if (seen.has(epNo)) continue;
    seen.add(epNo);
    chapters.push({
      id: `ep-${epNo}`,
      chapterNum: String(epNo),
      title: null,
    });
  }
  // Webtoons lists newest first; reverse to oldest first.
  chapters.reverse();
  return chapters;
}

export async function getWebtoonsImages(
  titleNo: number,
  episodeNo: number
): Promise<ScrapedImage[]> {
  // We need the viewer URL. We try the most common pattern.
  const listRes = await fetchWithTimeout(
    `${WEBTOONS_BASE}/en/fantasy/_/list?title_no=${titleNo}`,
    {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
    }
  );
  if (!listRes.ok) throw new Error(`Webtoons list ${listRes.status}`);
  const listHtml = await listRes.text();

  // Find the viewer URL for this episode.
  const viewerRegex = new RegExp(
    `href="([^"]*episode_no=${episodeNo})"[^>]*>(?:[^<]*<[^>]*>)*[^<]*`,
    "i"
  );
  const viewerMatch = listHtml.match(viewerRegex);
  if (!viewerMatch) throw new Error(`Episode ${episodeNo} not found`);
  const viewerUrl = viewerMatch[1].startsWith("http")
    ? viewerMatch[1]
    : `${WEBTOONS_BASE}${viewerMatch[1]}`;

  const res = await fetchWithTimeout(viewerUrl, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Accept-Language": "en-US,en;q=0.9",
      Referer: `${WEBTOONS_BASE}/`,
    },
  });
  if (!res.ok) throw new Error(`Webtoons viewer ${res.status}`);
  const html = await res.text();

  // Webtoons embeds image URLs in <img> tags with URLs like
  // https://webtoon-phinf.pstatic.net/...
  const images: ScrapedImage[] = [];
  const imgRegex = /data-url="(https:\/\/webtoon-phinf\.pstatic\.net\/[^"]+)"/gi;
  let m;
  while ((m = imgRegex.exec(html)) !== null) {
    images.push({ url: m[1], referer: `${WEBTOONS_BASE}/` });
  }

  // Fallback: try src attributes
  if (images.length === 0) {
    const srcRegex = /src="(https:\/\/webtoon-phinf\.pstatic\.net\/[^"]+)"/gi;
    while ((m = srcRegex.exec(html)) !== null) {
      images.push({ url: m[1], referer: `${WEBTOONS_BASE}/` });
    }
  }

  return images;
}

// ---------------------------------------------------------------------------
// Source 4: AsuraScans (asurascans.com) — JSON REST API
// AsuraScans is an Astro SPA backed by a clean REST API at api.asurascans.com.
// It returns rich metadata: covers, descriptions, genres, status, chapter counts.
// Endpoints:
//   GET /api/search?q={query}                         -> { data: [comic], meta }
//   GET /api/series/{slug}/chapters                   -> { data: [chapter] }   (newest-first)
//   GET /api/series/{slug}/chapters/{chapterSlug}     -> { data: { chapter: { pages: [{url}] } } }
// ---------------------------------------------------------------------------

const ASURA_API = "https://api.asurascans.com";
const ASURA_WEB = "https://asurascans.com";
const ASURA_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

function stripHtml(s: string): string {
  return s
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&#039;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

interface AsuraGenre {
  id: number;
  name: string;
  slug: string;
}

interface AsuraSearchComic {
  id: number;
  slug: string;
  title: string;
  description?: string;
  cover?: string;
  status?: string;
  type?: string;
  chapter_count?: number;
  public_url?: string;
  genres?: AsuraGenre[];
}

interface AsuraChapter {
  id: number;
  number: number;
  title: string;
  slug: string;
}

interface AsuraChapterPage {
  url: string;
}

export async function searchAsuraScans(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const res = await fetchWithTimeout(
    `${ASURA_API}/api/search?q=${encodeURIComponent(query)}`,
    {
      headers: {
        "User-Agent": ASURA_UA,
        Accept: "application/json",
        "Accept-Language": "en-US,en;q=0.9",
      },
    }
  );
  if (!res.ok) throw new Error(`AsuraScans search ${res.status}`);
  const body = await res.json();
  const comics: AsuraSearchComic[] = body?.data ?? [];

  return comics.slice(0, limit).map((c) => {
    const type = (c.type ?? "").toLowerCase();
    const origLang =
      type === "manhwa" ? "ko" : type === "manhua" ? "zh" : type === "manga" ? "ja" : null;
    const status = c.status
      ? c.status.charAt(0).toUpperCase() + c.status.slice(1)
      : null;
    return {
      id: `as-${c.slug}`,
      title: c.title || c.slug,
      description: c.description ? stripHtml(c.description) : "",
      coverUrl: c.cover || null,
      status,
      year: null,
      originalLanguage: origLang,
      availableTranslatedLanguages: ["en"],
      tags: (c.genres ?? []).map((g) => g.name).filter(Boolean),
      contentRating: "safe",
      lastChapter: c.chapter_count ? String(c.chapter_count) : null,
      source: "asurascans" as const,
      externalUrl: c.public_url
        ? `${ASURA_WEB}${c.public_url}`
        : `${ASURA_WEB}/comics/${c.slug}`,
    };
  });
}

export async function getAsuraScansChapters(
  slug: string
): Promise<ScrapedChapter[]> {
  const res = await fetchWithTimeout(
    `${ASURA_API}/api/series/${encodeURIComponent(slug)}/chapters`,
    {
      headers: {
        "User-Agent": ASURA_UA,
        Accept: "application/json",
        "Accept-Language": "en-US,en;q=0.9",
      },
    }
  );
  if (!res.ok) throw new Error(`AsuraScans chapters ${res.status} for ${slug}`);
  const body = await res.json();
  const chapters: AsuraChapter[] = body?.data ?? [];

  // API returns newest-first; reverse to oldest-first.
  const oldest = [...chapters].reverse();
  return oldest.map((c) => ({
    id: c.slug, // chapter slug (UUID) — needed for the images endpoint
    chapterNum: String(c.number),
    title: c.title || null,
  }));
}

export async function getAsuraScansImages(
  slug: string,
  chapterSlug: string
): Promise<ScrapedImage[]> {
  const res = await fetchWithTimeout(
    `${ASURA_API}/api/series/${encodeURIComponent(slug)}/chapters/${encodeURIComponent(chapterSlug)}`,
    {
      headers: {
        "User-Agent": ASURA_UA,
        Accept: "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        Referer: `${ASURA_WEB}/`,
      },
    }
  );
  if (!res.ok)
    throw new Error(`AsuraScans chapter ${res.status} for ${slug}/${chapterSlug}`);
  const body = await res.json();
  const pages: AsuraChapterPage[] = body?.data?.chapter?.pages ?? [];

  return pages
    .map((p) => p.url)
    .filter((u): u is string => Boolean(u))
    .map((u) => ({ url: u, referer: `${ASURA_WEB}/` }));
}

// ---------------------------------------------------------------------------
// SEARCH: MangaDex, MangaPill, Toonily
// ---------------------------------------------------------------------------

export async function searchMangaDex(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const url = `https://api.mangadex.org/manga?title=${encodeURIComponent(query)}&limit=${limit}&includes[]=cover_art&contentRating[]=safe&contentRating[]=suggestive&order[relevance]=desc`;
  const res = await fetchWithTimeout(url, {}, 15000);
  if (!res.ok) throw new Error(`MangaDex search ${res.status}`);
  const data = await res.json() as {
    data: Array<{
      id: string;
      attributes: {
        title?: Record<string, string>;
        description?: Record<string, string>;
        year?: number | null;
        status?: string;
        originalLanguage?: string;
        contentRating?: string;
        tags?: Array<{ attributes: { name?: Record<string, string> } }>;
      };
      relationships?: Array<{ type: string; attributes?: { fileName?: string } }>;
    }>;
  };

  return data.data.map((m) => {
    const title = m.attributes.title?.en || Object.values(m.attributes.title || {})[0] || m.id;
    const coverRel = m.relationships?.find((r) => r.type === "cover_art");
    const coverUrl = coverRel?.attributes?.fileName
      ? `https://uploads.mangadex.org/covers/${m.id}/${coverRel.attributes.fileName}.256.jpg`
      : null;
    const desc = m.attributes.description?.en || "";
    return {
      id: `md-${m.id}`,
      title,
      description: desc,
      coverUrl,
      status: m.attributes.status
        ? m.attributes.status.charAt(0).toUpperCase() + m.attributes.status.slice(1)
        : null,
      year: m.attributes.year ?? null,
      originalLanguage: m.attributes.originalLanguage ?? null,
      availableTranslatedLanguages: ["en"],
      tags: (m.attributes.tags || [])
        .map((t) => t.attributes?.name?.en)
        .filter(Boolean) as string[],
      contentRating: (m.attributes.contentRating || "safe") as "safe" | "suggestive" | "erotica",
      lastChapter: null,
      source: "mangadex" as const,
      externalUrl: `https://mangadex.org/title/${m.id}`,
    };
  });
}

export async function searchMangaPill(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const url = `https://mangapill.com/search?q=${encodeURIComponent(query)}`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: "https://mangapill.com/" },
  }, 15000);
  if (!res.ok) throw new Error(`MangaPill search ${res.status}`);
  const html = await res.text();

  const results: MangadexManga[] = [];
  // MangaPill search results: <a href="/manga/{id}-{slug}" class="...">
  const regex = /href="\/manga\/([^"]+)"[^>]*>[\s\S]*?<h3[^>]*>([^<]+)<\/h3>[\s\S]*?(?:src="([^"]+)")?/g;
  let match;
  while ((match = regex.exec(html)) !== null && results.length < limit) {
    const slug = match[1];
    const title = match[2].trim();
    const cover = match[3] || null;
    results.push({
      id: `mp-${slug}`,
      title,
      description: "",
      coverUrl: cover,
      status: null,
      year: null,
      originalLanguage: null,
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter: null,
      source: "mangapill" as const,
      externalUrl: `https://mangapill.com/manga/${slug}`,
    });
  }

  return results;
}

export async function searchToonily(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const url = `https://toonily.com/?s=${encodeURIComponent(query)}&post_type=wp-manga`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: "https://toonily.com/" },
  }, 15000);
  if (!res.ok) throw new Error(`Toonily search ${res.status}`);
  const html = await res.text();

  const results: MangadexManga[] = [];
  // Madara theme: results are <div class="row c-tabs-item__content"> with
  // <a href="https://toonily.com/serie/{slug}/"> and <img src="{cover}">
  const regex = /href="https:\/\/toonily\.com\/serie\/([^/]+)\/?"[^>]*>[\s\S]*?(?:src="([^"]+)")?[\s\S]*?<h3[^>]*>\s*<a[^>]*>([^<]+)<\/a>/g;
  let match;
  while ((match = regex.exec(html)) !== null && results.length < limit) {
    const slug = match[1];
    const cover = match[2] || null;
    const title = match[3].trim();
    results.push({
      id: `tl-${slug}`,
      title,
      description: "",
      coverUrl: cover,
      status: null,
      year: null,
      originalLanguage: "ko", // Toonily is manhwa-focused
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter: null,
      source: "toonily" as const,
      externalUrl: `https://toonily.com/serie/${slug}/`,
    });
  }

  return results;
}

// ---------------------------------------------------------------------------
// Unified dispatcher: given a manga ID, determine the source and delegate.
// ---------------------------------------------------------------------------

export type ScraperSource =
  | "mangahere"
  | "fanfox"
  | "webtoons"
  | "asurascans"
  | "mangadex"
  | "mangapill"
  | "toonily"
  | "comick"
  | "weebcentral";

export function getSourceFromId(id: string): ScraperSource | null {
  if (id.startsWith("mh-")) return "mangahere";
  if (id.startsWith("ff-")) return "fanfox";
  if (id.startsWith("wt-")) return "webtoons";
  if (id.startsWith("as-")) return "asurascans";
  if (id.startsWith("md-")) return "mangadex";
  if (id.startsWith("mp-")) return "mangapill";
  if (id.startsWith("tl-")) return "toonily";
  if (id.startsWith("cm-")) return "comick";
  if (id.startsWith("wc-")) return "weebcentral";
  return null;
}

export function getSlugFromId(id: string): string {
  return id.replace(/^(mh-|ff-|wt-|as-|md-|mp-|tl-|cm-|wc-)/, "");
}

export async function getChaptersForSource(
  source: ScraperSource,
  slug: string
): Promise<ScrapedChapter[]> {
  switch (source) {
    case "mangahere":
      return getMangaHereChapters(slug);
    case "fanfox":
      return getFanFoxChapters(slug);
    case "webtoons":
      return getWebtoonsChapters(parseInt(slug, 10));
    case "asurascans":
      return getAsuraScansChapters(slug);
    case "mangadex":
      return getMangaDexChapters(slug);
    case "mangapill":
      return getMangaPillChapters(slug);
    case "toonily":
      return getToonilyChapters(slug);
    case "comick":
      return getComickChapters(slug);
    case "weebcentral":
      return getWeebCentralChapters(slug);
  }
}

export async function getImagesForSource(
  source: ScraperSource,
  slug: string,
  chapterId: string
): Promise<ScrapedImage[]> {
  switch (source) {
    case "mangahere":
      return getMangaHereImages(slug, chapterId);
    case "fanfox":
      return getFanFoxImages(slug, chapterId);
    case "webtoons":
      return getWebtoonsImages(parseInt(slug, 10), parseInt(chapterId.replace(/^ep-/, ""), 10));
    case "asurascans":
      return getAsuraScansImages(slug, chapterId);
    case "mangadex":
      return getMangaDexImages(slug, chapterId);
    case "mangapill":
      return getMangaPillImages(slug, chapterId);
    case "toonily":
      return getToonilyImages(slug, chapterId);
    case "comick":
      return getComickImages(slug, chapterId);
    case "weebcentral":
      return getWeebCentralImages(slug, chapterId);
  }
}

// ---------------------------------------------------------------------------
// COMICK (comick.io) — JSON API aggregator. Large manhwa/manhua catalogue,
// no API key. `cm-{slug}` id form. See mini-services/pipeline-service/lib.ts
// for the matching job-time scraper (this file drives search + the chapter
// list shown at job creation).
// ---------------------------------------------------------------------------

// The API host moved api.comick.fun -> api.comick.dev (the .fun host now 000s
// from many networks). Allow COMICK_API to override.
const COMICK_API = process.env.COMICK_API || "https://api.comick.dev";
const COMICK_IMG = "https://meo.comick.pictures";
const COMICK_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";
const comickHeaders = () => ({
  "User-Agent": COMICK_UA,
  Accept: "application/json",
  Referer: "https://comick.io/",
});

export async function searchComick(query: string, limit = 10): Promise<MangadexManga[]> {
  const url = `${COMICK_API}/v1.0/search?q=${encodeURIComponent(query)}&type=comic&limit=${limit}&page=1`;
  const res = await fetchWithTimeout(url, { headers: comickHeaders() }, 15000);
  if (!res.ok) throw new Error(`Comick search ${res.status}`);
  const data = (await res.json()) as Array<{
    hid: string;
    slug: string;
    title: string;
    desc?: string;
    country?: string;
    status?: number;
    year?: number | null;
    md_covers?: Array<{ b2key?: string }>;
    last_chapter?: number | string | null;
  }>;
  return (Array.isArray(data) ? data : []).map((m) => ({
    id: `cm-${m.slug}`,
    title: m.title || m.slug,
    description: m.desc || "",
    coverUrl: m.md_covers?.[0]?.b2key ? `${COMICK_IMG}/${m.md_covers[0].b2key}` : null,
    status:
      m.status === 2 ? "Completed" : m.status === 1 ? "Ongoing" : m.status === 3 ? "Cancelled" : null,
    year: m.year ?? null,
    originalLanguage: m.country ?? null,
    availableTranslatedLanguages: ["en"],
    tags: [],
    contentRating: "safe" as const,
    lastChapter: m.last_chapter != null ? String(m.last_chapter) : null,
    source: "comick" as unknown as "mangadex",
    externalUrl: `https://comick.io/comic/${m.slug}`,
  }));
}

export async function getComickChapters(slug: string): Promise<ScrapedChapter[]> {
  const out: ScrapedChapter[] = [];
  for (let page = 1; page <= 40; page++) {
    const url = `${COMICK_API}/comic/${encodeURIComponent(slug)}/chapters?lang=en&page=${page}&limit=100&chap-order=1`;
    const res = await fetchWithTimeout(url, { headers: comickHeaders() }, 15000);
    if (!res.ok) throw new Error(`Comick chapters ${res.status} for ${slug}`);
    const body = (await res.json()) as {
      chapters?: Array<{ hid: string; chap?: string | null; title?: string | null; lang?: string }>;
      total?: number;
    };
    const chaps = body.chapters ?? [];
    if (chaps.length === 0) break;
    for (const c of chaps) {
      if (!c.hid) continue;
      out.push({
        id: c.hid,
        chapterNum: c.chap ?? String(out.length + 1),
        title: c.title ?? null,
        language: c.lang ?? "en",
      });
    }
    if (body.total && out.length >= body.total) break;
  }
  const seen = new Set<string>();
  const deduped = out.filter((c) => (seen.has(c.chapterNum) ? false : (seen.add(c.chapterNum), true)));
  deduped.sort((a, b) => (parseFloat(a.chapterNum) || 0) - (parseFloat(b.chapterNum) || 0));
  if (deduped.length === 0) throw new Error(`Comick: no English chapters for ${slug}`);
  return deduped;
}

export async function getComickImages(_slug: string, chapterHid: string): Promise<ScrapedImage[]> {
  const url = `${COMICK_API}/chapter/${encodeURIComponent(chapterHid)}?tachiyomi=true`;
  const res = await fetchWithTimeout(url, { headers: comickHeaders() }, 15000);
  if (!res.ok) throw new Error(`Comick images ${res.status} for chapter ${chapterHid}`);
  const body = (await res.json()) as { chapter?: { md_images?: Array<{ b2key?: string }> } };
  const imgs = (body.chapter?.md_images ?? [])
    .map((m) => m.b2key)
    .filter((k): k is string => Boolean(k))
    .map((k, i) => ({
      url: `${COMICK_IMG}/${k}`,
      referer: "https://comick.io/",
      filename: `${String(i + 1).padStart(3, "0")}.${k.split(".").pop() || "jpg"}`,
    }));
  if (imgs.length === 0) throw new Error(`Comick returned no images for chapter ${chapterHid}`);
  return imgs;
}

// ---------------------------------------------------------------------------
// 5. MANGADEX — official REST API, pure JSON, no HTML parsing needed.
// Source: api.mangadex.org — the gold standard for manga APIs.
// Has English-translated content from scanlation groups.
// ---------------------------------------------------------------------------

export async function getMangaDexChapters(mangaId: string): Promise<ScrapedChapter[]> {
  // MangaDex caps `limit` at 500 per request. A single unpaginated call
  // silently truncated any manga with more than 500 English chapters
  // (long-running series aren't rare) with no error and no indication
  // chapters were missing. Paginate via `offset` until every chapter is
  // fetched, using the response envelope's `total` field, with a hard
  // iteration cap as a sanity bound against a malformed/unbounded loop.
  const allData: Array<{
    id: string;
    attributes: {
      chapter?: string;
      title?: string | null;
      translatedLanguage?: string;
      externalUrl?: string | null;
      pages?: number;
    };
  }> = [];
  let offset = 0;
  const pageLimit = 500;
  const maxPages = 20; // 10,000 chapters — far beyond any real manga
  for (let page = 0; page < maxPages; page++) {
    const url = `https://api.mangadex.org/manga/${mangaId}/feed?translatedLanguage[]=en&order[chapter]=asc&limit=${pageLimit}&offset=${offset}&contentRating[]=safe&contentRating[]=suggestive`;
    const res = await fetchWithTimeout(url, {}, 15000);
    if (!res.ok) throw new Error(`MangaDex chapters ${res.status}`);
    const page_data = await res.json() as {
      data: Array<{
        id: string;
        attributes: {
          chapter?: string;
          title?: string | null;
          translatedLanguage?: string;
          externalUrl?: string | null;
          pages?: number;
        };
      }>;
      total?: number;
    };
    allData.push(...page_data.data);
    const total = page_data.total ?? allData.length;
    offset += page_data.data.length;
    if (page_data.data.length === 0 || offset >= total) break;
  }
  const data = { data: allData };

  const chapters: ScrapedChapter[] = [];
  for (const ch of data.data) {
    // Skip chapters that only have external URLs (DMCA'd/licensed)
    if (ch.attributes.externalUrl) continue;
    chapters.push({
      id: ch.id,
      chapterNum: ch.attributes.chapter ?? String(chapters.length + 1),
      title: ch.attributes.title ?? null,
      language: ch.attributes.translatedLanguage ?? "en",
    });
  }

  if (chapters.length === 0) {
    throw new Error("No readable chapters (all are external/licensed). Try a different source.");
  }

  return chapters;
}

export async function getMangaDexImages(
  _slug: string,
  chapterId: string
): Promise<ScrapedImage[]> {
  // MangaDex at-home server API: returns baseUrl + file list
  const url = `https://api.mangadex.org/at-home/server/${chapterId}`;
  const res = await fetchWithTimeout(url, {}, 15000);
  if (!res.ok) throw new Error(`MangaDex images ${res.status}`);

  const data = await res.json() as {
    baseUrl: string;
    chapter: {
      hash: string;
      data: string[];
    };
  };

  const images: ScrapedImage[] = [];
  for (const file of data.chapter.data) {
    const imgUrl = `${data.baseUrl}/data/${data.chapter.hash}/${file}`;
    images.push({
      url: imgUrl,
      referer: "https://mangadex.org/",
      filename: file,
      // MangaDex CDN requires Referer header
      headers: { Referer: "https://mangadex.org/" },
    });
  }

  if (images.length === 0) {
    throw new Error("MangaDex returned no images for this chapter");
  }

  return images;
}

// ---------------------------------------------------------------------------
// 6. MANGAPILL — scanlation aggregator, predictable image URLs.
// Source: mangapill.com — English manga/manhwa with CDN-hosted images.
// Image URLs are deterministic: cdn.readdetectiveconan.com/file/mangap/{id}/{offset}/{page}.jpg
// ---------------------------------------------------------------------------

export async function getMangaPillChapters(slug: string): Promise<ScrapedChapter[]> {
  const url = `https://mangapill.com/manga/${slug}`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: "https://mangapill.com/" },
  }, 15000);
  if (!res.ok) throw new Error(`MangaPill chapters ${res.status}`);

  const html = await res.text();
  const chapters: ScrapedChapter[] = [];

  // Parse chapter links from HTML
  const chapterRegex = /href="\/chapters\/[^"]*"/g;
  const matches = html.match(chapterRegex) || [];

  for (const match of matches) {
    const href = match.replace(/href="|"/g, "");
    const chapterId = href.split("/").pop() || "";
    // Extract chapter number from URL like /chapters/123-45678-chapter-5
    const numMatch = chapterId.match(/chapter-(\d+(?:\.\d+)?)/);
    const chapterNum = numMatch ? numMatch[1] : String(chapters.length + 1);
    chapters.push({
      id: chapterId,
      chapterNum,
      title: null,
      language: "en",
    });
  }

  if (chapters.length === 0) {
    throw new Error("MangaPill returned no chapters");
  }

  return chapters;
}

export async function getMangaPillImages(
  slug: string,
  chapterId: string
): Promise<ScrapedImage[]> {
  const url = `https://mangapill.com/chapters/${chapterId}`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: "https://mangapill.com/" },
  }, 15000);
  if (!res.ok) throw new Error(`MangaPill images ${res.status}`);

  const html = await res.text();
  const images: ScrapedImage[] = [];

  // MangaPill uses data-src or src on img tags with cdn.readdetectiveconan.com
  const imgRegex = /(?:data-src|src)="(https:\/\/cdn\.readdetectiveconan\.com\/file\/mangap\/[^"]+)"/g;
  let match;
  let page = 1;
  while ((match = imgRegex.exec(html)) !== null) {
    images.push({
      url: match[1],
      referer: "https://mangapill.com/",
      filename: `${String(page).padStart(3, "0")}.jpg`,
      headers: { Referer: "https://mangapill.com/" },
    });
    page++;
  }

  if (images.length === 0) {
    throw new Error("MangaPill returned no images for this chapter");
  }

  return images;
}

// ---------------------------------------------------------------------------
// 7. TOONILY — manhwa-focused scanlation aggregator (WordPress Madara theme).
// Source: toonily.com — lots of English manhwa, Madara theme pattern.
// The Madara scraper pattern is reusable for dozens of sister sites.
// ---------------------------------------------------------------------------

export async function getToonilyChapters(slug: string): Promise<ScrapedChapter[]> {
  const url = `https://toonily.com/serie/${slug}/`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: "https://toonily.com/" },
  }, 15000);
  if (!res.ok) throw new Error(`Toonily chapters ${res.status}`);

  const html = await res.text();
  const chapters: ScrapedChapter[] = [];

  // Madara theme: chapters are in <li> with <a href="...chapter-N/">
  const chapterRegex = /href="(https:\/\/toonily\.com\/[^"]+\/chapter-(\d+(?:\.\d+)?)\/?)"/g;
  const seen = new Set<string>();
  let match;
  while ((match = chapterRegex.exec(html)) !== null) {
    const chapterUrl = match[1];
    const chapterNum = match[2];
    if (seen.has(chapterNum)) continue;
    seen.add(chapterNum);
    // Extract chapter ID from URL
    const idMatch = chapterUrl.match(/\/([^/]+)\/chapter-/);
    const chapterId = idMatch ? `${idMatch[1]}-ch-${chapterNum}` : chapterNum;
    chapters.push({
      id: chapterId,
      chapterNum,
      title: null,
      language: "en",
    });
  }

  if (chapters.length === 0) {
    throw new Error("Toonily returned no chapters");
  }

  // Sort by chapter number (ascending)
  chapters.sort((a, b) => parseFloat(a.chapterNum) - parseFloat(b.chapterNum));

  return chapters;
}

export async function getToonilyImages(
  slug: string,
  chapterId: string
): Promise<ScrapedImage[]> {
  // chapterId format: {slug}-ch-{num}
  const numMatch = chapterId.match(/ch-(\d+(?:\.\d+)?)/);
  if (!numMatch) throw new Error(`Invalid Toonily chapter ID: ${chapterId}`);
  const chapterNum = numMatch[1];

  const url = `https://toonily.com/serie/${slug}/chapter-${chapterNum}/`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": "Mozilla/5.0", Referer: "https://toonily.com/" },
  }, 15000);
  if (!res.ok) throw new Error(`Toonily images ${res.status}`);

  const html = await res.text();
  const images: ScrapedImage[] = [];

  // Madara theme: images are in <div class="reading-content"> with <img data-src="...">
  const imgRegex = /data-src="(https:\/\/[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"/gi;
  let match;
  let page = 1;
  while ((match = imgRegex.exec(html)) !== null) {
    const imgUrl = match[1].trim();
    // Skip non-content images (logos, icons)
    if (imgUrl.includes("logo") || imgUrl.includes("icon") || imgUrl.includes("avatar")) continue;
    const ext = imgUrl.match(/\.(jpg|jpeg|png|webp)/i)?.[1] || "jpg";
    images.push({
      url: imgUrl,
      referer: "https://toonily.com/",
      filename: `${String(page).padStart(3, "0")}.${ext}`,
      headers: { Referer: "https://toonily.com/" },
    });
    page++;
  }

  if (images.length === 0) {
    throw new Error("Toonily returned no images for this chapter");
  }

  return images;
}

// ---------------------------------------------------------------------------
// WEEBCENTRAL (weebcentral.com) — large manga/manhwa catalogue, HTMX site.
// All fragment endpoints respond to plain GETs with an `HX-Request: true`
// header. `wc-{ULID}` id form. Series + chapter ids are 26-char ULIDs.
// See mini-services/pipeline-service/lib.ts for the matching job-time scraper.
// ---------------------------------------------------------------------------

const WEEBCENTRAL_BASE = "https://weebcentral.com";
const WEEBCENTRAL_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
const weebHeaders = (referer?: string): Record<string, string> => ({
  "User-Agent": WEEBCENTRAL_UA,
  "HX-Request": "true",
  Accept: "text/html",
  ...(referer ? { Referer: referer } : {}),
});

export async function searchWeebCentral(query: string, limit = 10): Promise<MangadexManga[]> {
  const url =
    `${WEEBCENTRAL_BASE}/search/data?text=${encodeURIComponent(query)}` +
    `&sort=Best%20Match&order=Descending&official=Any&anime=Any&adult=Any&display_mode=Minimal%20Display`;
  const res = await fetchWithTimeout(url, { headers: weebHeaders() }, 15000);
  if (!res.ok) throw new Error(`WeebCentral search ${res.status}`);
  const html = await res.text();
  const out: MangadexManga[] = [];
  const seen = new Set<string>();
  const re = /href="https:\/\/weebcentral\.com\/series\/([0-9A-Z]{26})\/([^"]+)"/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) && out.length < limit) {
    if (seen.has(m[1])) continue;
    seen.add(m[1]);
    const slug = m[2];
    out.push({
      id: `wc-${m[1]}`,
      title: decodeURIComponent(slug).replace(/-/g, " "),
      description: "",
      coverUrl: `https://temp.compsci88.com/cover/normal/${m[1]}.webp`,
      status: null,
      year: null,
      originalLanguage: null,
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter: null,
      source: "weebcentral",
      externalUrl: `${WEEBCENTRAL_BASE}/series/${m[1]}/${slug}`,
    });
  }
  return out;
}

/** Series ULID -> human title (the search slug is the ULID, not a readable name). */
export async function getWeebCentralSeriesTitle(seriesId: string): Promise<string | null> {
  try {
    const res = await fetchWithTimeout(
      `${WEEBCENTRAL_BASE}/series/${encodeURIComponent(seriesId)}/x`,
      { headers: { "User-Agent": WEEBCENTRAL_UA } },
      10000
    );
    if (!res.ok) return null;
    const html = await res.text();
    const m = html.match(/<title>([^<]+?)\s*\|\s*Weeb Central<\/title>/i);
    return m ? m[1].trim() : null;
  } catch {
    return null;
  }
}

export async function getWeebCentralChapters(seriesId: string): Promise<ScrapedChapter[]> {
  const url = `${WEEBCENTRAL_BASE}/series/${encodeURIComponent(seriesId)}/full-chapter-list`;
  const res = await fetchWithTimeout(
    url,
    { headers: weebHeaders(`${WEEBCENTRAL_BASE}/series/${seriesId}`) },
    15000
  );
  if (!res.ok) throw new Error(`WeebCentral chapters ${res.status} for ${seriesId}`);
  const html = (await res.text()).replace(/\s+/g, " ");
  const out: ScrapedChapter[] = [];
  const seen = new Set<string>();
  const re =
    /href="(?:https:\/\/weebcentral\.com)?\/chapters\/([0-9A-Z]{26})"[\s\S]*?<span[^>]*>\s*(?:Chapter|Episode)\s+([\d.]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    if (seen.has(m[1])) continue;
    seen.add(m[1]);
    out.push({ id: m[1], chapterNum: m[2], title: null, language: "en" });
  }
  if (out.length === 0) throw new Error(`WeebCentral returned no chapters for ${seriesId}`);
  out.reverse(); // full-chapter-list is newest-first
  return out;
}

export async function getWeebCentralImages(
  _seriesId: string,
  chapterId: string
): Promise<ScrapedImage[]> {
  const url = `${WEEBCENTRAL_BASE}/chapters/${encodeURIComponent(chapterId)}/images?is_prev=False&current_page=1&reading_style=long_strip`;
  const res = await fetchWithTimeout(
    url,
    { headers: weebHeaders(`${WEEBCENTRAL_BASE}/chapters/${chapterId}`) },
    15000
  );
  if (!res.ok) throw new Error(`WeebCentral images ${res.status} for chapter ${chapterId}`);
  const html = await res.text();
  const images: ScrapedImage[] = [];
  const re = /<img[^>]+src="(https:\/\/[^"]+?\.(?:jpg|jpeg|png|webp))"/gi;
  let m: RegExpExecArray | null;
  let page = 1;
  while ((m = re.exec(html)) !== null) {
    const ext = m[1].match(/\.(jpg|jpeg|png|webp)/i)?.[1] || "jpg";
    images.push({
      url: m[1],
      referer: `${WEEBCENTRAL_BASE}/`,
      filename: `${String(page).padStart(3, "0")}.${ext}`,
      headers: { "User-Agent": WEEBCENTRAL_UA, Referer: `${WEEBCENTRAL_BASE}/` },
    });
    page++;
  }
  if (images.length === 0) throw new Error(`WeebCentral returned no images for chapter ${chapterId}`);
  return images;
}
