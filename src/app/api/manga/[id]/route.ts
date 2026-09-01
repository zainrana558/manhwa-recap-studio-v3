import { NextRequest, NextResponse } from "next/server";
import {
  getSourceFromId,
  getSlugFromId,
  getChaptersForSource,
} from "@/lib/scrapers";

export const dynamic = "force-dynamic";

interface ChapterListItem {
  id: string;
  chapter: string | null;
  title: string | null;
  language: string;
  pages: number;
  volume: string | null;
}

/**
 * GET /api/manga/{id}?lang={language}
 *
 * The id is prefixed with the source: mh- (MangaHere), ff- (FanFox),
 * wt- (Webtoons), as- (AsuraScans), md- (MangaDex), mp- (MangaPill), or
 * tl- (Toonily). Returns manga metadata + chapter list from the
 * appropriate scraping source.
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    if (!id) {
      return NextResponse.json(
        { error: "Manga id is required." },
        { status: 400 }
      );
    }

    const source = getSourceFromId(id);
    if (!source) {
      return NextResponse.json(
        { error: `Invalid manga id. Expected prefix: mh-, ff-, wt-, as-, md-, mp-, or tl-. Got: ${id}` },
        { status: 400 }
      );
    }
    const slug = getSlugFromId(id);

    const chaptersRaw = await getChaptersForSource(source, slug);

    const chapters: ChapterListItem[] = chaptersRaw.map((c) => ({
      id: c.id,
      chapter: c.chapterNum,
      title: c.title,
      language: "en",
      pages: 0,
      volume: null,
    }));

    // Build a minimal manga object.
    const title = source === "webtoons"
      ? id // Webtoons doesn't have a slug, use the titleNo
      : slug.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase());

    const baseUrls: Record<string, string> = {
      mangahere: `https://www.mangahere.cc/manga/${slug}/`,
      fanfox: `https://fanfox.net/manga/${slug}/`,
      webtoons: `https://www.webtoons.com/en/fantasy/_/list?title_no=${slug}`,
      asurascans: `https://asurascans.com/series/${slug}/`,
      mangadex: `https://mangadex.org/title/${slug}`,
      mangapill: `https://mangapill.com/manga/${slug}`,
      toonily: `https://toonily.com/serie/${slug}/`,
      comick: `https://comick.io/comic/${slug}`,
      weebcentral: `https://weebcentral.com/series/${slug}`,
    };

    const manga = {
      id,
      title,
      description: "",
      coverUrl: null,
      status: null,
      year: null,
      originalLanguage: source === "webtoons" ? "ko" : null,
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter: chaptersRaw[0]?.chapterNum ?? null,
      source,
      externalUrl: baseUrls[source],
    };

    return NextResponse.json({ manga, chapters });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to load manga: ${message}` },
      { status: 502 }
    );
  }
}
