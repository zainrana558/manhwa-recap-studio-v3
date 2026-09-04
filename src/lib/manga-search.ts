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

  const [mhRes, ffRes, wtRes, asRes, mdRes, mpRes, tlRes, cmRes, wcRes, malRes, alRes] = await Promise.allSettled([
    searchMangaHere(query, safeLimit),
    searchFanFox(query, safeLimit),
    searchWebtoons(query, safeLimit),
    searchAsuraScans(query, safeLimit),
    searchMangaDex(query, safeLimit),
    searchMangaPill(query, safeLimit),
    searchToonily(query, safeLimit),
    searchComick(query, safeLimit),
    searchWeebCentral(query, safeLimit),
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
  const mal = malRes.status === "fulfilled" ? malRes.value : [];
  const anilist = alRes.status === "fulfilled" ? alRes.value : [];

  // Dedupe: keep first occurrence per normalized title.
  // Scraping sources are queried first so they win ties.
  // Upgrade: if a later entry with the same title has a cover image but the
  // existing one does not, replace it (so AsuraScans/MAL/AniList covers win
  // over MangaHere/FanFox coverless results for the same title).
  const seen = new Map<string, MangadexManga>();
  for (const m of [
    ...mangahere,
    ...fanfox,
    ...webtoons,
    ...asurascans,
    ...mangadex,
    ...mangapill,
    ...toonily,
    ...comick,
    ...weebcentral,
    ...mal,
    ...anilist,
  ]) {
    const key = normalizeTitle(m.title);
    if (!key) continue;
    const existing = seen.get(key);
    if (!existing) {
      seen.set(key, m);
      continue;
    }
    // Prefer an entry with a cover over one without.
    if (!existing.coverUrl && m.coverUrl) {
      seen.set(key, m);
    }
  }
  // MAL/AniList are metadata-only catalogs — neither hosts scrapeable manga
  // pages, so a result surviving the dedup above (i.e. no scrapeable source
  // matched the same title) is a dead end: selecting it and starting a job
  // always fails at the scrape phase (getSourceFromId has no mal-/anilist-
  // case, by design — there's nothing to scrape). Scrapeable sources are
  // deduped first (see the loop above), so any MAL/AniList entry that's
  // *not* filtered out here already lost its slot to a real scrapeable
  // result for the same title; this only removes the entries with no
  // scrapeable match at all, rather than hiding MAL/AniList results wholesale.
  const deduped: MangadexManga[] = [...seen.values()].filter(
    (m) => m.source !== "mal" && m.source !== "anilist"
  );

  // Sort: by relevance to the query first (exact title match > starts-with >
  // contains > no match), then by source priority (scrapeable first).
  const sourceOrder: Record<string, number> = {
    mangahere: 0,
    fanfox: 1,
    webtoons: 2,
    asurascans: 3,
    mangadex: 4,
    mangapill: 5,
    toonily: 6,
    comick: 7,
    weebcentral: 8,
    mal: 9,
    anilist: 10,
  };
  const qNorm = normalizeTitle(query);
  const qWords = titleWords(query);
  function relevance(title: string): number {
    const t = normalizeTitle(title);
    if (!qNorm) return 4;
    if (t === qNorm) return 0; // exact match (whitespace-insensitive)
    if (t.startsWith(qNorm) || qNorm.startsWith(t)) return 1; // one is a prefix of the other
    if (t.includes(qNorm) || qNorm.includes(t)) return 2; // one contains the other
    // every query word appears somewhere in the title (any order) — e.g.
    // "nan hao and shang feng" vs an alt title "Nan Hao & Shang Feng: ..."
    const tw = new Set(titleWords(title));
    if (qWords.length >= 2 && qWords.every((w) => tw.has(w))) return 3;
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
  // If there's at least one solid title match (relevance <= 3), drop the pure
  // keyword-noise tail so the user isn't shown 13 unrelated webtoons above the
  // series they searched for.
  const hasSolid = deduped.some((m) => relevance(m.title) <= 3);
  const ranked = hasSolid ? deduped.filter((m) => relevance(m.title) <= 3) : deduped;

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
  source: "mangahere" | "fanfox" | "webtoons" | "mal" | "anilist" | "asurascans" | "mangadex" | "mangapill" | "toonily" | "comick" | "weebcentral",
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
    case "mal":
      return searchJikan(query, limit);
    case "anilist":
      return searchAniList(query, limit);
  }
}
