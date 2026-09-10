import type { MangadexManga } from "@/types/pipeline";
import {
  searchMangaHere,
  searchFanFox,
  searchWebtoons,
  searchAsuraScans,
  searchMangaDex,
  searchMangaPill,
  searchToonily,
  searchComick,
  searchWeebCentral,
  searchMgeko,
} from "./scrapers";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const FETCH_TIMEOUT_MS = 5_000;

async function fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Unified multi-source manga search.
 *
 * Sources:
 *  - MangaHere (scrapeable — has chapter images via HTML scraping)
 *  - Jikan (MyAnimeList) — free REST, no auth, ~3 req/sec
 *  - AniList — free GraphQL, no auth
 *
 * Non-MangaHere results are tagged with `source: "mal" | "anilist"` and an
 * `externalUrl` pointing at the original MAL/AniList page. The frontend
 * resolves them to a MangaHere manga by re-searching MangaHere by title
 * when the user selects one (MangaHere has scrapeable chapter images).
 */

// ---------------------------------------------------------------------------
// Jikan (MyAnimeList) — https://docs.api.jikan.moe/
// ---------------------------------------------------------------------------

interface JikanImageSet {
  large_image_url?: string;
  image_url?: string;
}

interface JikanManga {
  mal_id: number;
  url?: string;
  title?: string;
  title_english?: string | null;
  title_japanese?: string | null;
  images?: { jpg?: JikanImageSet; webp?: JikanImageSet };
  synopsis?: string | null;
  status?: string | null;
  year?: number | null;
  rating?: string | null;
  type?: string | null;
  genres?: Array<{ name?: string }>;
  themes?: Array<{ name?: string }>;
  demographics?: Array<{ name?: string }>;
  authors?: Array<{ name?: string }>;
}

interface JikanResponse {
  data?: JikanManga[];
}

/** Free, no-auth Jikan search. ~3 req/sec rate limit — we only fire 1 per query. */
async function searchJikan(query: string, limit = 12): Promise<MangadexManga[]> {
  const url = new URL("https://api.jikan.moe/v4/manga");
  url.searchParams.set("q", query);
  url.searchParams.set("limit", String(Math.min(Math.max(limit, 1), 25)));
  url.searchParams.set("sfw", "true");
  url.searchParams.set("order_by", "relevance");
  url.searchParams.set("sort", "desc");

  let res: Response;
  try {
    res = await fetchWithTimeout(url.toString(), {
      headers: { accept: "application/json" },
      cache: "no-store",
    });
  } catch (err) {
    throw new Error(
      `Jikan network error: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    throw new Error(`Jikan API ${res.status} ${res.statusText}`);
  }

  const body = (await res.json()) as JikanResponse;
  const items = body.data ?? [];

  return items.map((m) => {
    const tags: string[] = [];
    for (const g of m.genres ?? []) if (g?.name) tags.push(g.name);
    for (const t of m.themes ?? []) if (t?.name) tags.push(t.name);
    for (const d of m.demographics ?? []) if (d?.name) tags.push(d.name);

    // Best-effort content rating mapping.
    const rating = (m.rating ?? "").toLowerCase();
    let contentRating: string | null = null;
    if (rating.includes("erotica")) contentRating = "erotica";
    else if (rating.includes("hentai")) contentRating = "pornographic";
    else if (rating.includes("mature") || rating.includes("17"))
      contentRating = "suggestive";
    else contentRating = "safe";

    return {
      id: `mal-${m.mal_id}`,
      title: m.title_english || m.title || `MAL ${m.mal_id}`,
      description: m.synopsis ?? "",
      coverUrl:
        m.images?.webp?.large_image_url ||
        m.images?.jpg?.large_image_url ||
        null,
      status: m.status ?? null,
      year: m.year ?? null,
      originalLanguage: null, // Jikan doesn't expose this reliably
      availableTranslatedLanguages: [],
      tags,
      contentRating,
      lastChapter: null,
      source: "mal" as const,
      externalUrl: m.url ?? `https://myanimelist.net/manga/${m.mal_id}`,
    };
  });
}

// ---------------------------------------------------------------------------
// AniList — https://docs.anilist.co/ (GraphQL)
// ---------------------------------------------------------------------------

interface AniListTitle {
  romaji?: string | null;
  english?: string | null;
  native?: string | null;
}

interface AniListMedia {
  id: number;
  idMal?: number | null;
  title?: AniListTitle;
  coverImage?: { large?: string; extraLarge?: string };
  description?: string | null;
  status?: string | null;
  startDate?: { year?: number | null };
  countryOfOrigin?: string | null;
  genres?: string[];
  tags?: Array<{ name?: string }>;
  siteUrl?: string | null;
}

interface AniListResponse {
  data?: { Page?: { media?: AniListMedia[] } };
}

/** Map AniList country-of-origin (JP/KR/CN/TW) to a language code. */
function mapAniListCountry(code: string | null | undefined): string | null {
  if (!code) return null;
  const c = code.toUpperCase();
  switch (c) {
    case "JP":
      return "ja";
    case "KR":
      return "ko";
    case "CN":
      return "zh";
    case "TW":
      return "zh";
    default:
      return c.toLowerCase();
  }
}

const ANILIST_QUERY = `
  query ($search: String, $perPage: Int) {
    Page(perPage: $perPage) {
      media(type: MANGA, search: $search, sort: SEARCH_MATCH) {
        id
        idMal
        title { romaji english native }
        coverImage { large extraLarge }
        description
        status
        startDate { year }
        countryOfOrigin
        genres
        tags { name }
        siteUrl
      }
    }
  }
`;

/** Free, no-auth GraphQL search. Returns MangaDex-shaped results. */
async function searchAniList(query: string, limit = 12): Promise<MangadexManga[]> {
  const variables = {
    search: query,
    perPage: Math.min(Math.max(limit, 1), 25),
  };

  let res: Response;
  try {
    res = await fetchWithTimeout("https://graphql.anilist.co", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify({ query: ANILIST_QUERY, variables }),
      cache: "no-store",
    });
  } catch (err) {
    throw new Error(
      `AniList network error: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (!res.ok) {
    throw new Error(`AniList API ${res.status} ${res.statusText}`);
  }

  const body = (await res.json()) as AniListResponse;
  const items = body.data?.Page?.media ?? [];

  return items.map((m) => {
    const tags: string[] = [];
    for (const g of m.genres ?? []) if (g) tags.push(g);
    for (const t of m.tags ?? []) if (t?.name) tags.push(t.name);

    // AniList descriptions contain <br> tags — strip HTML.
    const description = (m.description ?? "")
      .replace(/<br\s*\/?>/gi, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    // AniList statuses are uppercase (FINISHING, RELEASING, FINISHED, CANCELLED, HIATUS).
    const status = m.status
      ? m.status.charAt(0) + m.status.slice(1).toLowerCase()
      : null;

    return {
      id: `anilist-${m.id}`,
      title: m.title?.english || m.title?.romaji || m.title?.native || `AniList ${m.id}`,
      description,
      coverUrl: m.coverImage?.extraLarge || m.coverImage?.large || null,
      status,
      year: m.startDate?.year ?? null,
      originalLanguage: mapAniListCountry(m.countryOfOrigin),
      availableTranslatedLanguages: [],
      tags,
      contentRating: "safe",
      lastChapter: null,
      source: "anilist" as const,
      externalUrl: m.siteUrl ?? null,
    };
  });
}

// ---------------------------------------------------------------------------
// Unified search + dedupe
// ---------------------------------------------------------------------------

/** Normalize a title for dedupe comparisons. */
function normalizeTitle(title: string): string {
  return title
    .toLowerCase()
    .normalize("NFKD") // strip accents
    .replace(/[\u0300-\u036f]/g, "")
    // collapse ALL non-alphanumerics (incl. spaces): "Nanhao And Shangfeng" and
    // "nan hao and shang feng" both -> "nanhaoandshangfeng" so tokenisation
    // differences between sources don't defeat exact/substring matching.
    .replace(/[^a-z0-9]+/g, "")
    .trim();
}

// words of a title/query, accent- and punctuation-stripped, for token-set matching
function titleWords(s: string): string[] {
  return s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

export interface MangaSearchSources {
  mangahere: number;
  fanfox: number;
  webtoons: number;
  mal: number;
  anilist: number;
  asurascans: number;
  mangadex: number;
  mangapill: number;
  toonily: number;
  comick: number;
  weebcentral: number;
  mgeko: number;
}

export interface UnifiedSearchResult {
  manga: MangadexManga[];
  sources: MangaSearchSources;
}

/**
 * Query all 5 sources in parallel (3 scraping + 2 metadata), dedupe by
 * normalized title, and sort so scrapeable sources come first.
 *
 * If a source errors, results from the others are still returned.
 */
export async function searchAllManga(
  query: string,
  limit = 12
): Promise<UnifiedSearchResult> {
  const safeLimit = Math.min(Math.max(limit, 1), 25);

  const [mhRes, ffRes, wtRes, asRes, mdRes, mpRes, tlRes, cmRes, wcRes, mgRes, malRes, alRes] = await Promise.allSettled([
    searchMangaHere(query, safeLimit),
    searchFanFox(query, safeLimit),
    searchWebtoons(query, safeLimit),
    searchAsuraScans(query, safeLimit),
    searchMangaDex(query, safeLimit),
    searchMangaPill(query, safeLimit),
    searchToonily(query, safeLimit),
    searchComick(query, safeLimit),
    searchWeebCentral(query, safeLimit),
    searchMgeko(query, safeLimit),
    searchJikan(query, safeLimit),
    searchAniList(query, safeLimit),
  ]);

  const mangahere = mhRes.status === "fulfilled" ? mhRes.value : [];
  const fanfox = ffRes.status === "fulfilled" ? ffRes.value : [];
  const webtoons = wtRes.status === "fulfilled" ? wtRes.value : [];
  const asurascans = asRes.status === "fulfilled" ? asRes.value : [];
  const mangadex = mdRes.status === "fulfilled" ? mdRes.value : [];
  const mangapill = mpRes.status === "fulfilled" ? mpRes.value : [];
  const toonily = tlRes.status === "fulfilled" ? tlRes.value : [];
  const comick = cmRes.status === "fulfilled" ? cmRes.value : [];
  const weebcentral = wcRes.status === "fulfilled" ? wcRes.value : [];
  const mgeko = mgRes.status === "fulfilled" ? mgRes.value : [];
  const mal = malRes.status === "fulfilled" ? malRes.value : [];
  const anilist = alRes.status === "fulfilled" ? alRes.value : [];

  // NO cross-source hiding. Every scrapeable source that has the series gets
  // its OWN card — the same title on mgeko (tiles), AsuraScans (tall strips)
  // and Webtoons all show, because they scrape differently and the user picks
  // which one to recap. We only collapse:
  //   1. literal duplicates: the same (source, id) returned twice
  //   2. same source + same normalized title (a source listing a series twice)
  //   3. a MAL/AniList metadata entry whose title a real scrapeable source
  //      already has — that card can't be scraped and adds nothing the
  //      scrapeable card doesn't (MAL/AniList with NO scrapeable twin are
  //      kept: handleSelect re-resolves them to a scrapeable source on click)
  const scrapeableTitles = new Set<string>();
  for (const m of [...mgeko, ...mangadex, ...webtoons, ...asurascans, ...mangapill,
                   ...comick, ...weebcentral, ...toonily, ...mangahere, ...fanfox]) {
    const k = normalizeTitle(m.title);
    if (k) scrapeableTitles.add(k);
  }
  const seen = new Map<string, MangadexManga>();
  for (const m of [
    ...mgeko,
    ...mangadex,
    ...webtoons,
    ...asurascans,
    ...mangapill,
    ...comick,
    ...weebcentral,
    ...toonily,
    ...mangahere,
    ...fanfox,
    ...mal,
    ...anilist,
  ]) {
    const key = normalizeTitle(m.title);
    if (!key) continue;
    const src = m.source ?? "mangahere";
    const isMeta = src === "mal" || src === "anilist";
    if (isMeta && scrapeableTitles.has(key)) continue;   // redundant metadata card
    const dedupeKey = `${src}::${m.id || key}`;
    const perSourceTitleKey = `${src}::title::${key}`;
    if (seen.has(dedupeKey) || seen.has(perSourceTitleKey)) {
      const existing = seen.get(dedupeKey) ?? seen.get(perSourceTitleKey)!;
      if (!existing.coverUrl && m.coverUrl) existing.coverUrl = m.coverUrl;
      if (!existing.description && m.description) existing.description = m.description;
      continue;
    }
    seen.set(dedupeKey, m);
    seen.set(perSourceTitleKey, m);
  }
  const deduped: MangadexManga[] = [...new Set(seen.values())];

  // Sort: by relevance to the query first (exact title match > starts-with >
  // contains > no match), then by source priority (scrapeable first).
  const sourceOrder: Record<string, number> = {
    mgeko: 0,
    mangadex: 1,
    webtoons: 2,
    asurascans: 3,
    mangahere: 4,
    fanfox: 5,
    mangapill: 6,
    comick: 7,
    weebcentral: 8,
    toonily: 9,
    mal: 10,
    anilist: 11,
  };
  const STOP = new Set(["and", "the", "of", "a", "an", "to", "in", "vs", "or"]);
  const qNorm = normalizeTitle(query);
  const qWords = titleWords(query).filter((w) => !STOP.has(w));
  function relevance(title: string): number {
    const t = normalizeTitle(title);
    if (!qNorm) return 4;
    if (t === qNorm) return 0; // exact match (whitespace-insensitive)
    if (t.startsWith(qNorm) || qNorm.startsWith(t)) return 1; // one is a prefix of the other
    if (t.includes(qNorm) || qNorm.includes(t)) return 2; // one contains the other
    // every meaningful query word appears in the title (any order), ignoring
    // stopwords — "nan hao AND shang feng" still matches "Nán Hào Shàng Fēng".
    const tw = new Set(titleWords(title).filter((w) => !STOP.has(w)));
    if (qWords.length >= 2 && qWords.every((w) => tw.has(w))) {
      // reward tighter matches: title has no extra words -> rank just below "contains"
      return tw.size <= qWords.length + 2 ? 3 : 3.5;
    }
    // majority of query words present -> still a real candidate, keep it visible
    const hit = qWords.filter((w) => tw.has(w)).length;
    if (qWords.length >= 3 && hit / qWords.length >= 0.6) return 3.8;
    return 4; // keyword / alt-title hit only
  }
  deduped.sort((a, b) => {
    const ra = relevance(a.title);
    const rb = relevance(b.title);
    if (ra !== rb) return ra - rb;
    const sa = sourceOrder[a.source ?? "mangahere"] ?? 99;
    const sb = sourceOrder[b.source ?? "mangahere"] ?? 99;
    return sa - sb;
  });
  // No hiding: every result from every source is returned. The relevance sort
  // above already floats the real matches to the top; weaker keyword hits sit
  // at the bottom where the source-filter chips and the eye can skip them.
  const ranked = deduped;

  return {
    manga: ranked,
    sources: {
      mangahere: mangahere.length,
      fanfox: fanfox.length,
      webtoons: webtoons.length,
      asurascans: asurascans.length,
      mangadex: mangadex.length,
      mangapill: mangapill.length,
      toonily: toonily.length,
      comick: comick.length,
      weebcentral: weebcentral.length,
      mgeko: mgeko.length,
      mal: mal.length,
      anilist: anilist.length,
    },
  };
}

/**
 * Search a single source by name. Used by the frontend to re-resolve a
 * non-scrapeable result to a scrapeable manga before opening the config page.
 */
export async function searchSingleSource(
  query: string,
  source: "mangahere" | "fanfox" | "webtoons" | "mal" | "anilist" | "asurascans" | "mangadex" | "mangapill" | "toonily" | "comick" | "weebcentral" | "mgeko",
  limit = 12
): Promise<MangadexManga[]> {
  switch (source) {
    case "mangahere":
      return searchMangaHere(query, limit);
    case "fanfox":
      return searchFanFox(query, limit);
    case "webtoons":
      return searchWebtoons(query, limit);
    case "asurascans":
      return searchAsuraScans(query, limit);
    case "mangadex":
      return searchMangaDex(query, limit);
    case "mangapill":
      return searchMangaPill(query, limit);
    case "toonily":
      return searchToonily(query, limit);
    case "comick":
      return searchComick(query, limit);
    case "weebcentral":
      return searchWeebCentral(query, limit);
    case "mgeko":
      return searchMgeko(query, limit);
    case "mal":
      return searchJikan(query, limit);
    case "anilist":
      return searchAniList(query, limit);
  }
}
