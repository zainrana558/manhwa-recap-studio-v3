import { getPrivateExt } from "@/lib/private-loader";
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

// Cloudflare-fronted hosts that 403 a plain datacenter fetch on TLS/JA3
// fingerprint (toonily / weebcentral). The pipeline-service side
// (mini-services/pipeline-service/lib.ts) routes these through cf_fetch.py
// (curl_cffi). Here — the Next.js search path — we can't spawn a subprocess
// from a Turbopack-bundled route without the bundler choking on the venv
// symlink, so the search request goes out via CF_PROXY_URL when configured
// (a plain HTTP proxy / cf-worker), else a normal fetch (which just yields
// fewer results for that one source — Promise.allSettled tolerates it).
const CF_IMPERSONATE_HOSTS = /(^|\.)(toonily\.(com|me)|weebcentral\.com|comick\.io)$/i;
const CF_PROXY_URL = process.env.CF_PROXY_URL || "";

async function fetchWithTimeout(
  url: string,
  init?: RequestInit,
  timeoutMs: number = FETCH_TIMEOUT_MS
): Promise<Response> {
  let target = url;
  try {
    if (CF_PROXY_URL && CF_IMPERSONATE_HOSTS.test(new URL(url).hostname)) {
      target = CF_PROXY_URL.replace(/\/$/, "") + "?url=" + encodeURIComponent(url);
    }
  } catch {
    /* ignore */
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(target, { ...init, signal: controller.signal });
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
  coverUrl: string | null;
}

export async function searchWebtoons(
  query: string,
  limit = 10
): Promise<MangadexManga[]> {
  const res = await fetchWithTimeout(
    `${WEBTOONS_BASE}/en/search?keyword=${encodeURIComponent(query)}&searchType=ALL`,
    {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
      },
    }
  );
  if (!res.ok) throw new Error(`Webtoons search ${res.status}`);
  const html = await res.text();

  const titles: WebtoonsTitle[] = [];
  const seen = new Set<number>();

  // Match full <a> card blocks so we capture the real poster <img src="https://webtoon-phinf.pstatic.net/..."> and <strong class="title">
  const cardRe = /<a\s+[^>]*href="([^"]*\/en\/([^"]+)\/list\?title_no=(\d+))"[^>]*>([\s\S]*?)<\/a>/gi;
  let match;
  while ((match = cardRe.exec(html)) !== null) {
    const titleNo = parseInt(match[3], 10);
    if (seen.has(titleNo)) continue;
    seen.add(titleNo);
    const pathParts = match[2].split("/");
    const titleSlug = pathParts[pathParts.length - 1];
    const inner = match[4] ?? "";
    const titleM = inner.match(/<strong[^>]*class="[^"]*title[^"]*"[^>]*>([\s\S]*?)<\/strong>/i);
    const rawTitle = titleM
      ? titleM[1].replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&#39;|&apos;/g, "'").replace(/&quot;/g, '"').trim()
      : titleSlug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    if (!rawTitle) continue;
    const imgM = inner.match(/<img[^>]+src="([^"]+)"/i);
    const coverUrl = imgM && /^https?:\/\//i.test(imgM[1]) ? imgM[1] : null;
    titles.push({
      titleNo,
      title: rawTitle,
      genre: pathParts[0] ?? "",
      url: match[1].startsWith("http") ? match[1] : `${WEBTOONS_BASE}${match[1]}`,
      coverUrl,
    });
  }

  return titles.slice(0, limit).map((t) => ({
    id: `wt-${t.titleNo}`,
    title: t.title,
    description: "",
    coverUrl: t.coverUrl,
    status: "Ongoing",
    year: null,
    originalLanguage: "ko",
    availableTranslatedLanguages: ["en"],
    tags: t.genre ? [t.genre.charAt(0).toUpperCase() + t.genre.slice(1)] : [],
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

let _mdTagCache: Record<string, string> | null = null;
async function getMangaDexTagMap(): Promise<Record<string, string>> {
  if (_mdTagCache) return _mdTagCache;
  try {
    const res = await fetchWithTimeout("https://api.mangadex.org/manga/tag", {}, 10000);
    if (!res.ok) return {};
    const j = await res.json() as { data?: Array<{ id: string; attributes?: { name?: Record<string, string> } }> };
    const map: Record<string, string> = {};
    for (const t of j.data || []) {
      const nm = (t.attributes?.name?.en || "").toLowerCase().trim();
      if (nm) map[nm] = t.id;
    }
    _mdTagCache = map;
    return map;
  } catch {
    return {};
  }
}

export async function searchMangaDex(
  query: string,
  limit = 10,
  adult = false,
  genre = ""
): Promise<MangadexManga[]> {
  const ratings = adult
    ? "contentRating[]=erotica&contentRating[]=pornographic&contentRating[]=suggestive"
    : "contentRating[]=safe&contentRating[]=suggestive";
  let tagParam = "";
  let effectiveTitle = query.trim();
  const gNorm = genre.trim().toLowerCase();
  if (gNorm) {
    const tagMap = await getMangaDexTagMap();
    const aliases: Record<string, string[]> = {
      milf: ["milf", "office workers"],
      smut: ["smut", "sexual violence"],
    };
    const candidates = [gNorm, ...(aliases[gNorm] || [])];
    const matchedIds = candidates.map((c) => tagMap[c]).filter(Boolean);
    if (matchedIds.length > 0) {
      tagParam = `&includedTags[]=${encodeURIComponent(matchedIds[0])}`;
    } else if (!effectiveTitle) {
      effectiveTitle = genre.trim();
    }
  }
  const titleParam = effectiveTitle ? `title=${encodeURIComponent(effectiveTitle)}&` : "";
  const orderParam = effectiveTitle ? "order[relevance]=desc" : "order[followedCount]=desc";
  const url = `https://api.mangadex.org/manga?${titleParam}limit=${limit}&includes[]=cover_art&${ratings}${tagParam}&${orderParam}`;
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
  // MangaPill search cards (2026 markup):
  //   <a href="/manga/{numId}/{slug}" class="mb-2"> … </a>
  //   <div class="mt-3 font-black leading-tight line-clamp-2">{Title}</div>
  const regex =
    /<a href="\/manga\/(\d+\/[a-z0-9-]+)"[^>]*class="mb-2"[\s\S]{0,400}?<div class="mt-3 font-black[^"]*">\s*([^<]+?)\s*<\/div>/g;
  const seen = new Set<string>();
  let match;
  while ((match = regex.exec(html)) !== null && results.length < limit) {
    const slug = match[1];
    if (seen.has(slug)) continue;
    seen.add(slug);
    const title = match[2].trim();
    const cover: string | null = null;
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

// ---------------------------------------------------------------------------
// Unified dispatcher: given a manga ID, determine the source and delegate.
// ---------------------------------------------------------------------------

export type ScraperSource =
  | "asurascans"
  | "thunderscans"
  | "vortexscans"
  | "rizzfables"
  | "webtoons"
  | "mgeko"
  | "mangadex"
  | "mangahere"
  | "fanfox"
  | "mangapill"
  | "comick"
  | "weebcentral"
  | (string & {});

export function getSourceFromId(id: string): ScraperSource | null {
  if (id.startsWith("mh-")) return "mangahere";
  if (id.startsWith("ff-")) return "fanfox";
  if (id.startsWith("wt-")) return "webtoons";
  if (id.startsWith("as-")) return "asurascans";
  if (id.startsWith("ts-")) return "thunderscans";
  if (id.startsWith("vs-")) return "vortexscans";
  if (id.startsWith("rf-")) return "rizzfables";
  if (id.startsWith("md-")) return "mangadex";
  if (id.startsWith("mp-")) return "mangapill";
  if (id.startsWith("cm-")) return "comick";
  if (id.startsWith("wc-")) return "weebcentral";
  if (id.startsWith("mg-")) return "mgeko";
  const ext = getPrivateExt();
  if (ext?.isPrivateId(id)) return ext.getSourceFromId(id);
  return null;
}

export function getSlugFromId(id: string): string {
  const ext = getPrivateExt();
  if (ext?.isPrivateId(id)) return ext.getSlugFromId(id) ?? id;
  return id.replace(/^(mh-|ff-|wt-|as-|ts-|vs-|rf-|md-|mp-|cm-|wc-|mg-)/, "");
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
    case "thunderscans":
      return getThunderScansChapters(slug);
    case "vortexscans":
      return getVortexScansChapters(slug);
    case "rizzfables":
      return getRizzFablesChapters(slug);
    case "mangadex":
      return getMangaDexChapters(slug);
    case "mangapill":
      return getMangaPillChapters(slug);
    case "comick":
      return getComickChapters(slug);
    case "weebcentral":
      return getWeebCentralChapters(slug);
    case "mgeko":
      return getMgekoChapters(slug);
    default: {
      const ext = getPrivateExt();
      const ch = await ext?.getChapters(source, slug);
      if (ch) return ch;
      throw new Error(`Unsupported source: ${source}`);
    }
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
    case "thunderscans":
      return getThunderScansImages(slug, chapterId);
    case "vortexscans":
      return getVortexScansImages(slug, chapterId);
    case "rizzfables":
      return getRizzFablesImages(slug, chapterId);
    case "mangadex":
      return getMangaDexImages(slug, chapterId);
    case "mangapill":
      return getMangaPillImages(slug, chapterId);
    case "comick":
      return getComickImages(slug, chapterId);
    case "weebcentral":
      return getWeebCentralImages(slug, chapterId);
    case "mgeko":
      return getMgekoImages(slug, chapterId);
    default: {
      const ext = getPrivateExt();
      const imgs = await ext?.getImages(source, slug, chapterId);
      if (imgs) return imgs;
      throw new Error(`Unsupported source: ${source}`);
    }
  }
}

// ---------------------------------------------------------------------------
// MGEKO / MangaGeko (mgeko.cc) — large manhwa + manhua + manga aggregator,
// plain server-rendered HTML, no Cloudflare challenge (works from a
// datacenter IP where toonily / comick / weebcentral do not). `mg-{slug}`
// id form. Structure:
//   search : /search/?search={q}            -> <a href="/manga/{slug}/" title="{Title}">
//   chaps  : /manga/{slug}/all-chapters/     -> <a href="/reader/en/{chapSlug}/">  (newest first)
//   images : /reader/en/{chapSlug}/          -> <img src="https://imgsrv5.com/.../NN.jpg">
// ---------------------------------------------------------------------------
const MGEKO_BASE = "https://www.mgeko.cc";
const MGEKO_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

export async function searchMgeko(query: string, limit = 10): Promise<MangadexManga[]> {
  const res = await fetchWithTimeout(
    `${MGEKO_BASE}/search/?search=${encodeURIComponent(query)}`,
    { headers: { "User-Agent": MGEKO_UA } },
    15000,
  );
  if (!res.ok) throw new Error(`Mgeko search ${res.status}`);
  const html = await res.text();
  const out: MangadexManga[] = [];
  const seen = new Set<string>();
  // <a href="/manga/{slug}/" title="{Title}"> … <img class="lazy" data-src='{cover}'>
  const re =
    /<a href="\/manga\/([^"/]+)\/"\s+title="([^"]+)"(?:[\s\S]{0,400}?(?:data-src|data-original|src)=['"]([^'"]+\.(?:png|jpe?g|webp))['"])?/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null && out.length < limit) {
    if (seen.has(m[1])) continue;
    seen.add(m[1]);
    let cover: string | null =
      m[3] && !/loading\.gif|placeholder|no-?cover|logo/i.test(m[3]) ? m[3] : null;
    if (cover && cover.startsWith("/")) cover = `${MGEKO_BASE}${cover}`;
    out.push({
      id: `mg-${m[1]}`,
      title: m[2].trim(),
      description: "",
      coverUrl: cover,
      status: null,
      year: null,
      originalLanguage: null,
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter: null,
      source: "mgeko" as const,
      externalUrl: `${MGEKO_BASE}/manga/${m[1]}/`,
    });
  }
  return out;
}

export async function getMgekoChapters(slug: string): Promise<ScrapedChapter[]> {
  const res = await fetchWithTimeout(
    `${MGEKO_BASE}/manga/${slug}/all-chapters/`,
    { headers: { "User-Agent": MGEKO_UA, Referer: `${MGEKO_BASE}/manga/${slug}/` } },
    15000,
  );
  if (!res.ok) throw new Error(`Mgeko chapters ${res.status} for ${slug}`);
  const html = await res.text();
  const out: ScrapedChapter[] = [];
  const seen = new Set<string>();
  const re = /href="\/reader\/en\/([^"]+?)\/"/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    const chapSlug = m[1];
    if (seen.has(chapSlug)) continue;
    seen.add(chapSlug);
    const numM = chapSlug.match(/chapter-([0-9.]+)/i);
    out.push({ id: chapSlug, chapterNum: numM ? numM[1] : String(out.length + 1), title: null, language: "en" });
  }
  out.reverse(); // page lists newest first
  return out;
}

export async function getMgekoImages(_slug: string, chapterId: string): Promise<ScrapedImage[]> {
  const res = await fetchWithTimeout(
    `${MGEKO_BASE}/reader/en/${chapterId}/`,
    { headers: { "User-Agent": MGEKO_UA, Referer: `${MGEKO_BASE}/` } },
    15000,
  );
  if (!res.ok) throw new Error(`Mgeko images ${res.status} for ${chapterId}`);
  const html = await res.text();
  const urls: string[] = [];
  const seen = new Set<string>();
  const re = /<img[^>]+(?:src|data-src)="(https?:\/\/[^"]+?\.(?:jpg|jpeg|png|webp))"/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    if (/logo|banner|avatar|icon|credits-mgeko|radioads|radioonlinehd|\/ads\/|vline\.png/i.test(m[1]) || seen.has(m[1])) continue;
    seen.add(m[1]);
    urls.push(m[1]);
  }
  return urls.map((url, i) => ({
    url,
    referer: `${MGEKO_BASE}/`,
    filename: `${String(i + 1).padStart(3, "0")}.jpg`,
    headers: { "User-Agent": MGEKO_UA },
  }));
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

// Human-readable series title for a MangaDex id (search gives it, but
// /api/manga/[id] loads by id and would otherwise show the raw UUID).
export async function getMangaDexTitle(mangaId: string): Promise<string | null> {
  try {
    const res = await fetchWithTimeout(
      `https://api.mangadex.org/manga/${mangaId}`,
      {},
      10000
    );
    if (!res.ok) return null;
    const j = (await res.json()) as {
      data?: { attributes?: { title?: Record<string, string>; altTitles?: Array<Record<string, string>> } };
    };
    const t = j.data?.attributes?.title ?? {};
    const alts = (j.data?.attributes?.altTitles ?? []).reduce(
      (acc, o) => ({ ...acc, ...o }),
      {} as Record<string, string>
    );
    return (
      t.en ||
      alts.en ||
      t["ja-ro"] ||
      alts["ja-ro"] ||
      Object.values(t)[0] ||
      Object.values(alts)[0] ||
      null
    );
  } catch {
    return null;
  }
}

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
  // Row label lives in `<span class="">…</span>`. It is NOT always "Chapter N":
  // long webtoons use season prefixes ("S4 - Episode 178", "S3 - Chapter 235").
  // Grab the label, then pull the trailing number; fall back to list position.
  const re =
    /\/chapters\/([0-9A-Z]{26})"[\s\S]*?<span class="">\s*([^<]+?)\s*<\/span>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    if (seen.has(m[1])) continue;
    seen.add(m[1]);
    const label = m[2];
    const numM =
      label.match(/(?:Chapter|Episode)\s+([\d.]+)/i) || label.match(/([\d.]+)\s*$/);
    out.push({ id: m[1], chapterNum: numM ? numM[1] : "", title: label, language: "en" });
  }
  if (out.length === 0) throw new Error(`WeebCentral returned no chapters for ${seriesId}`);
  out.reverse(); // full-chapter-list is newest-first
  out.forEach((c, i) => {
    if (!c.chapterNum) c.chapterNum = String(i + 1);
  });
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


// ---------------------------------------------------------------------------
// ThunderScans (en-thunderscans.com) — WordPress MangaStream scanlator with
// ultra-clean uniform 800x13000+ vertical webtoon strips, no Cloudflare.
// `ts-{slug}` id form.
// ---------------------------------------------------------------------------
const THUNDER_WEB = "https://en-thunderscans.com";
const SCAN_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

export async function searchThunderScans(query: string, limit = 12): Promise<MangadexManga[]> {
  const q = query.trim();
  if (!q) return [];
  const res = await fetchWithTimeout(`${THUNDER_WEB}/?s=${encodeURIComponent(q)}`, {
    headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`ThunderScans search HTTP ${res.status}`);
  const html = (await res.text()).replace(/<script[\s\S]*?<\/script>/gi, "");
  const cards = html.match(/<div class="bsx">[\s\S]*?<\/a>\s*<\/div>/g) ?? html.match(/<div class="bsx">[\s\S]*?<\/div>\s*<\/div>/g) ?? [];
  const seen = new Set<string>();
  const out: MangadexManga[] = [];
  for (const card of cards) {
    const hrefM = card.match(/href="https:\/\/en-thunderscans\.com\/comics\/([^"\/?]+)\/?"/i);
    if (!hrefM) continue;
    const slug = hrefM[1];
    if (seen.has(slug) || slug.includes("+")) continue;
    seen.add(slug);
    const titleM =
      card.match(/<div class="tt">\s*([^<]+?)\s*<\/div>/i) ||
      card.match(/title="([^"]+)"/i);
    const title = (titleM ? titleM[1] : slug.replace(/-/g, " "))
      .replace(/&#8217;/g, "'")
      .replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"')
      .trim();
    if (!title || title.includes("value.title") || title.includes("'+")) continue;
    const imgM = card.match(/<img[^>]+src="([^"]+)"/i);
    const coverUrl = imgM ? imgM[1] : null;
    const chM = card.match(/class="epxs"[^>]*>[^0-9]*([0-9]+(?:\.[0-9]+)?)/i) || card.match(/Chapter\s+([0-9]+(?:\.[0-9]+)?)/i);
    const lastChapter = chM ? chM[1] : null;
    out.push({
      id: `ts-${slug}`,
      title,
      description: "",
      coverUrl,
      status: null,
      year: null,
      originalLanguage: "ko",
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter,
      source: "thunderscans" as const,
      externalUrl: `${THUNDER_WEB}/comics/${slug}/`,
    });
    if (out.length >= limit) break;
  }
  return out;
}

export async function getThunderScansChapters(slug: string): Promise<ScrapedChapter[]> {
  const res = await fetchWithTimeout(`${THUNDER_WEB}/comics/${encodeURIComponent(slug)}/`, {
    headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`ThunderScans series HTTP ${res.status}`);
  const html = await res.text();
  const re = /<li[^>]*data-num="([^"]+)"[^>]*>[\s\S]*?href="https:\/\/en-thunderscans\.com\/([^"\/]+)\/?"/gi;
  const seen = new Set<string>();
  const out: ScrapedChapter[] = [];
  for (const m of html.matchAll(re)) {
    const rawNum = m[1].trim();
    const chSlug = m[2].trim();
    if (!rawNum || rawNum.includes("{{") || seen.has(chSlug)) continue;
    seen.add(chSlug);
    out.push({
      id: chSlug,
      chapterNum: rawNum,
      title: null,
      language: "en",
      pageCount: 0,
      external: false,
    });
  }
  out.sort((a, b) => (parseFloat(a.chapterNum || "0") || 0) - (parseFloat(b.chapterNum || "0") || 0));
  return out;
}

export async function getThunderScansImages(_slug: string, chapterSlug: string): Promise<ScrapedImage[]> {
  const res = await fetchWithTimeout(`${THUNDER_WEB}/${encodeURIComponent(chapterSlug)}/`, {
    headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`ThunderScans chapter HTTP ${res.status}`);
  const html = await res.text();
  const m = html.match(/"images"\s*:\s*\[([^\]]+)\]/);
  const seen = new Set<string>();
  const out: ScrapedImage[] = [];
  if (m) {
    for (const im of m[1].matchAll(/"(https?:[^"]+)"/g)) {
      const u = im[1].replace(/\\\//g, "/");
      if (!seen.has(u) && !/logo|icon|banner|discord/i.test(u)) {
        seen.add(u);
        out.push({ url: u, referer: `${THUNDER_WEB}/` });
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// VortexScans (vortexscans.org) â€” Iken JSON REST API + 800x5000 WebP strips
// (same uniform vertical strip format as AsuraScans). `vs-{postId}--{slug}`.
// ---------------------------------------------------------------------------
const VORTEX_API = "https://api.vortexscans.org";
const VORTEX_WEB = "https://vortexscans.org";

function parseVortexSlug(raw: string): { postId: string; slug: string } {
  const idx = raw.indexOf("--");
  if (idx > 0) {
    return { postId: raw.slice(0, idx), slug: raw.slice(idx + 2) };
  }
  return { postId: "", slug: raw };
}

export async function searchVortexScans(query: string, limit = 12): Promise<MangadexManga[]> {
  const q = query.trim();
  if (!q) return [];
  const url = `${VORTEX_API}/api/query?searchTerm=${encodeURIComponent(q)}&perPage=${Math.min(Math.max(limit, 1), 25)}&page=1`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": SCAN_UA, Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`VortexScans search HTTP ${res.status}`);
  const body = (await res.json()) as {
    posts?: Array<{
      id: number;
      slug: string;
      postTitle: string;
      postContent?: string | null;
      featuredImage?: string | null;
      seriesStatus?: string | null;
      genres?: Array<{ name?: string }>;
      _count?: { chapters?: number };
      chapters?: Array<{ number?: number | string }>;
      chapterCount?: number;
    }>;
  };
  return (body.posts ?? [])
    .filter((p) => p?.id && p?.slug && p?.postTitle && !/\[novel\]/i.test(p.postTitle))
    .slice(0, limit)
    .map((p) => ({
      id: `vs-${p.id}--${p.slug}`,
      title: p.postTitle,
      description: (p.postContent ?? "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim(),
      coverUrl: p.featuredImage || null,
      status: p.seriesStatus ? p.seriesStatus.charAt(0) + p.seriesStatus.slice(1).toLowerCase() : null,
      year: null,
      originalLanguage: "ko",
      availableTranslatedLanguages: ["en"],
      tags: (p.genres ?? []).map((g) => g.name).filter((n): n is string => Boolean(n)),
      contentRating: "safe",
      lastChapter:
        p.chapters?.[0]?.number != null
          ? String(p.chapters[0].number)
          : (p._count?.chapters ?? p.chapterCount)
            ? String(p._count?.chapters ?? p.chapterCount)
            : null,
      source: "vortexscans" as const,
      externalUrl: `${VORTEX_WEB}/series/${p.slug}`,
    }));
}

export async function getVortexScansChapters(compoundSlug: string): Promise<ScrapedChapter[]> {
  const { postId } = parseVortexSlug(compoundSlug);
  if (!postId) return [];
  const url = `${VORTEX_API}/api/chapters?postId=${encodeURIComponent(postId)}&take=500&order=asc`;
  const res = await fetchWithTimeout(url, {
    headers: { "User-Agent": SCAN_UA, Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`VortexScans chapters HTTP ${res.status}`);
  const body = (await res.json()) as {
    post?: {
      chapters?: Array<{
        slug: string;
        number?: number | string | null;
        title?: string | null;
        isLocked?: boolean;
      }>;
    };
  };
  const raw = (body.post?.chapters ?? []).filter((c) => c?.slug && !c.isLocked);
  const out: ScrapedChapter[] = raw.map((c) => ({
    id: c.slug,
    chapterNum: c.number != null ? String(c.number) : null,
    title: c.title || null,
    language: "en",
    pageCount: 0,
    external: false,
  }));
  out.sort((a, b) => (parseFloat(a.chapterNum || "0") || 0) - (parseFloat(b.chapterNum || "0") || 0));
  return out;
}

export async function getVortexScansImages(compoundSlug: string, chapterSlug: string): Promise<ScrapedImage[]> {
  const { slug } = parseVortexSlug(compoundSlug);
  const res = await fetchWithTimeout(
    `${VORTEX_WEB}/series/${encodeURIComponent(slug)}/${encodeURIComponent(chapterSlug)}`,
    {
      headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
      cache: "no-store",
    }
  );
  if (!res.ok) throw new Error(`VortexScans chapter HTTP ${res.status}`);
  const html = await res.text();
  const re = /https:\/\/storage\.vortexscans\.org\/upload\/series\/[^"'\\<>\s]+\.(?:webp|jpg|jpeg|png)/gi;
  const seen = new Set<string>();
  const out: ScrapedImage[] = [];
  for (const m of html.matchAll(re)) {
    const u = m[0];
    if (u.includes("/featured/") || seen.has(u)) continue;
    seen.add(u);
    out.push({ url: u, referer: `${VORTEX_WEB}/` });
  }
  return out;
}

// ---------------------------------------------------------------------------
// RizzFables (rizzfables.com) â€” WordPress MangaStream scanlator with uniform
// 900px-wide tall vertical WebP strips, no Cloudflare challenge. `rf-{slug}`.
// ---------------------------------------------------------------------------
const RIZZ_WEB = "https://rizzfables.com";

export async function searchRizzFables(query: string, limit = 12): Promise<MangadexManga[]> {
  const q = query.trim();
  if (!q) return [];
  const res = await fetchWithTimeout(`${RIZZ_WEB}/?s=${encodeURIComponent(q)}`, {
    headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`RizzFables search HTTP ${res.status}`);
  const html = (await res.text()).replace(/<script[\s\S]*?<\/script>/gi, "");
  const cards = html.match(/<div class="bsx">[\s\S]*?<\/a>\s*<\/div>/g) ?? html.match(/<div class="bsx">[\s\S]*?<\/div>\s*<\/div>/g) ?? [];
  const seen = new Set<string>();
  const out: MangadexManga[] = [];
  for (const card of cards) {
    const hrefM = card.match(/href="https:\/\/rizzfables\.com\/series\/([^"\/]+)\/?"/i);
    if (!hrefM) continue;
    const slug = hrefM[1];
    if (seen.has(slug) || slug.includes("+")) continue;
    seen.add(slug);
    const titleM =
      card.match(/<div class="tt">\s*([^<]+?)\s*<\/div>/i) ||
      card.match(/title="([^"]+)"/i);
    const title = (titleM ? titleM[1] : slug.replace(/-/g, " "))
      .replace(/&#8217;/g, "'")
      .replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"')
      .trim();
    if (!title || title.includes("value.title") || title.includes("'+")) continue;
    const imgM = card.match(/<img[^>]+src="([^"]+)"/i);
    let coverUrl = imgM ? imgM[1] : null;
    if (coverUrl && coverUrl.startsWith("/")) {
      coverUrl = `${RIZZ_WEB}${coverUrl}`;
    }
    const chM = card.match(/class="epxs"[^>]*>[^0-9]*([0-9]+(?:\.[0-9]+)?)/i) || card.match(/Chapter\s+([0-9]+(?:\.[0-9]+)?)/i);
    const lastChapter = chM ? chM[1] : null;
    out.push({
      id: `rf-${slug}`,
      title,
      description: "",
      coverUrl,
      status: null,
      year: null,
      originalLanguage: "ko",
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter,
      source: "rizzfables" as const,
      externalUrl: `${RIZZ_WEB}/series/${slug}`,
    });
    if (out.length >= limit) break;
  }
  return out;
}

export async function getRizzFablesChapters(slug: string): Promise<ScrapedChapter[]> {
  const res = await fetchWithTimeout(`${RIZZ_WEB}/series/${encodeURIComponent(slug)}`, {
    headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`RizzFables series HTTP ${res.status}`);
  const html = await res.text();
  const re = /<li[^>]*data-num="([^"]+)"[^>]*>[\s\S]*?href="https:\/\/rizzfables\.com\/chapter\/([^"\/]+)\/?"/gi;
  const seen = new Set<string>();
  const out: ScrapedChapter[] = [];
  for (const m of html.matchAll(re)) {
    const rawNum = m[1].trim();
    const chSlug = m[2].trim();
    if (!rawNum || rawNum.includes("{{") || seen.has(chSlug)) continue;
    seen.add(chSlug);
    out.push({
      id: chSlug,
      chapterNum: rawNum,
      title: null,
      language: "en",
      pageCount: 0,
      external: false,
    });
  }
  // Chapter list is newest-first on the page â€” sort ascending by chapter number.
  out.sort((a, b) => (parseFloat(a.chapterNum || "0") || 0) - (parseFloat(b.chapterNum || "0") || 0));
  return out;
}

export async function getRizzFablesImages(_slug: string, chapterSlug: string): Promise<ScrapedImage[]> {
  const res = await fetchWithTimeout(`${RIZZ_WEB}/chapter/${encodeURIComponent(chapterSlug)}`, {
    headers: { "User-Agent": SCAN_UA, Accept: "text/html,*/*" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`RizzFables chapter HTTP ${res.status}`);
  const html = await res.text();
  const areaM = html.match(/<div[^>]+id="readerarea"[^>]*>([\s\S]*?)<\/div>/i);
  const scope = areaM ? areaM[1] : html;
  const re = /src="(https?:\/\/[^"]+\.(?:webp|jpg|jpeg|png)(?:\?[^"]*)?)"/gi;
  const seen = new Set<string>();
  const out: ScrapedImage[] = [];
  for (const m of scope.matchAll(re)) {
    const u = m[1];
    if (seen.has(u) || /logo|icon|avatar|banner|discord/i.test(u)) continue;
    seen.add(u);
    out.push({ url: u, referer: `${RIZZ_WEB}/` });
  }
  return out;
}
