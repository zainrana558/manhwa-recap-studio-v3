import { NextRequest, NextResponse } from "next/server";
import { searchAllManga, searchSingleSource } from "@/lib/manga-search";

export const dynamic = "force-dynamic";

// Increase the max duration for this API route since it calls multiple external sources.
export const maxDuration = 30;

const VALID_SOURCES = ["mangahere", "fanfox", "webtoons", "asurascans", "mangadex", "mangapill", "toonily", "comick", "weebcentral", "mal", "anilist"] as const;
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

    if (!q) {
      return NextResponse.json(
        { error: "Query parameter 'q' is required." },
        { status: 400 }
      );
    }

    // Single-source mode.
    if (isSource(sourceParam)) {
      try {
        const manga = await searchSingleSource(q, sourceParam, limit);
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
      const { manga, sources } = await searchAllManga(q, limit);
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
