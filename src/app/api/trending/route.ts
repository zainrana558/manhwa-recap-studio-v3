import { NextResponse } from "next/server";
import { searchSingleSource } from "@/lib/manga-search";
import type { MangadexManga } from "@/types/pipeline";

export const dynamic = "force-dynamic";

interface TrendingItem extends MangadexManga {
  averageScore?: number | null;
  trendingRank?: number;
}

const CACHE_TTL_MS = 15 * 60 * 1000;
let cachedTrending: { ts: number; items: TrendingItem[] } | null = null;

const ADULT_GENRES = new Set(["hentai", "erotica", "ecchi", "smut", "adult"]);

function stripHtml(text: string | null | undefined): string {
  if (!text) return "";
  return text
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

function normTitle(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

async function resolveChapterCount(title: string): Promise<string | null> {
  const target = normTitle(title);
  for (const src of ["comick", "asurascans"] as const) {
    try {
      const hits = await searchSingleSource(title, src, 4);
      const valid = hits.filter((h) => h.lastChapter && parseFloat(h.lastChapter) > 0);
      if (!valid.length) continue;
      const exact = valid.find((h) => normTitle(h.title) === target);
      if (exact?.lastChapter) return exact.lastChapter;
      const fuzzy = valid.find((h) => {
        const k = normTitle(h.title);
        return k.includes(target) || target.includes(k);
      });
      if (fuzzy?.lastChapter) return fuzzy.lastChapter;
      if (valid[0]?.lastChapter) return valid[0].lastChapter;
    } catch {
      // try next source
    }
  }
  return null;
}

export async function GET() {
  if (cachedTrending && Date.now() - cachedTrending.ts < CACHE_TTL_MS) {
    return NextResponse.json(
      { items: cachedTrending.items, updatedAt: new Date(cachedTrending.ts).toISOString() },
      { headers: { "Cache-Control": "public, s-maxage=900, stale-while-revalidate=1800" } }
    );
  }

  try {
    const query = `
      query {
        Page(page: 1, perPage: 14) {
          media(type: MANGA, countryOfOrigin: "KR", isAdult: false, sort: TRENDING_DESC) {
            id
            title { english romaji }
            coverImage { large extraLarge }
            chapters
            status
            startDate { year }
            averageScore
            genres
            description(asHtml: false)
            siteUrl
          }
        }
      }
    `;

    const res = await fetch("https://graphql.anilist.co", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
      },
      body: JSON.stringify({ query }),
      signal: AbortSignal.timeout(9000),
    });

    if (!res.ok) {
      throw new Error(`AniList ${res.status}`);
    }

    const json = await res.json();
    const rawMedia = (json?.data?.Page?.media ?? []) as Array<{
      id: number;
      title?: { english?: string | null; romaji?: string | null };
      coverImage?: { large?: string | null; extraLarge?: string | null };
      chapters?: number | null;
      status?: string | null;
      startDate?: { year?: number | null };
      averageScore?: number | null;
      genres?: string[];
      description?: string | null;
      siteUrl?: string | null;
    }>;

    const filtered = rawMedia.filter((m) => {
      const genres = (m.genres ?? []).map((g) => g.toLowerCase());
      return !genres.some((g) => ADULT_GENRES.has(g));
    });

    const baseItems: TrendingItem[] = filtered.slice(0, 11).map((m, idx) => {
      const title = (m.title?.english || m.title?.romaji || "Untitled").trim();
      const statusMap: Record<string, string> = {
        RELEASING: "ongoing",
        FINISHED: "completed",
        HIATUS: "hiatus",
        CANCELLED: "cancelled",
      };
      return {
        id: `al-${m.id}`,
        title,
        source: "anilist",
        coverUrl: m.coverImage?.extraLarge || m.coverImage?.large || null,
        description: stripHtml(m.description),
        status: statusMap[m.status ?? ""] ?? "ongoing",
        year: m.startDate?.year ?? null,
        originalLanguage: "ko",
        availableTranslatedLanguages: ["en"],
        tags: (m.genres ?? []).slice(0, 4),
        contentRating: "safe",
        lastChapter: m.chapters && m.chapters > 0 ? String(m.chapters) : null,
        externalUrl: m.siteUrl ?? null,
        averageScore: m.averageScore ?? null,
        trendingRank: idx + 1,
      };
    });

    // Enrich missing chapter counts in parallel (cached 15 min)
    const items = await Promise.all(
      baseItems.map(async (it) => {
        if (it.lastChapter) return it;
        const ch = await resolveChapterCount(it.title);
        return ch ? { ...it, lastChapter: ch } : it;
      })
    );

    cachedTrending = { ts: Date.now(), items };
    return NextResponse.json(
      { items, updatedAt: new Date(cachedTrending.ts).toISOString() },
      { headers: { "Cache-Control": "public, s-maxage=900, stale-while-revalidate=1800" } }
    );
  } catch {
    return NextResponse.json(
      { items: cachedTrending?.items ?? [], updatedAt: new Date().toISOString() },
      { status: 200 }
    );
  }
}
