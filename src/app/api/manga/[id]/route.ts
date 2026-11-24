import { getPrivateExt } from "@/lib/private-loader";
import { NextRequest, NextResponse } from "next/server";
import {
  getSourceFromId,
  getSlugFromId,
  getChaptersForSource,
  getWeebCentralSeriesTitle,
  getMangaDexTitle,
  searchMgeko,
  searchWeebCentral,
  searchMangaPill,
  searchAsuraScans,
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

    const ext = getPrivateExt();
    const isExtId = Boolean(ext?.isPrivateId(id));
    let source = isExtId ? ext!.getSourceFromId(id) : getSourceFromId(id);
    if (!source) {
      return NextResponse.json(
        { error: `Invalid manga id. Got: ${id}` },
        { status: 400 }
      );
    }
    let slug = isExtId ? ext!.getSlugFromId(id) : getSlugFromId(id);
    let resolvedId = id;
    let resolvedTitle: string | null = null;
    const isAdultReq = _req.nextUrl.searchParams.get("adult") === "1";

    let chaptersRaw: Awaited<ReturnType<typeof getChaptersForSource>> = [];
    try {
      chaptersRaw = isExtId
        ? await ext!.getChapters(source, slug)
        : await getChaptersForSource(source, slug);
    } catch (err) {
      if (source !== "mangadex") throw err;
    }

    // Automatic mirror fallback when a MangaDex title only has external/licensed chapters
    if (source === "mangadex" && chaptersRaw.length === 0) {
      resolvedTitle = _req.nextUrl.searchParams.get("title")?.trim() || (await getMangaDexTitle(slug)) || null;
      if (resolvedTitle) {
        const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
        const target = norm(resolvedTitle);
        const extMirrors = isAdultReq && ext ? await ext.findFallbackMirrors(resolvedTitle) : [];
        const searchers = [
          ...extMirrors,
          () => searchMgeko(resolvedTitle!, 5),
          () => searchWeebCentral(resolvedTitle!, 5),
          () => searchMangaPill(resolvedTitle!, 5),
          () => searchAsuraScans(resolvedTitle!, 5),
        ];
        for (const fn of searchers) {
          try {
            const hits = await fn();
            if (!hits || hits.length === 0) continue;
            const exact = hits.find((h) => norm(h.title) === target) ?? hits[0];
            const fbSource = getSourceFromId(exact.id);
            const fbSlug = getSlugFromId(exact.id);
            if (!fbSource || !fbSlug) continue;
            const fbChaps = await getChaptersForSource(fbSource, fbSlug);
            if (fbChaps.length > 0) {
              source = fbSource;
              slug = fbSlug;
              resolvedId = exact.id;
              chaptersRaw = fbChaps;
              break;
            }
          } catch {
            // Try next mirror
          }
        }
      }
    }

    const chapters: ChapterListItem[] = chaptersRaw.map((c) => ({
      id: c.id,
      chapter: c.chapterNum,
      title: c.title,
      language: "en",
      pages: 0,
      volume: null,
    }));

    // Build a minimal manga object.
    const title = resolvedTitle
      ?? (source === "webtoons"
      ? resolvedId
      : source === "weebcentral"
      ? (await getWeebCentralSeriesTitle(slug)) ?? slug
      : source === "mangadex"
      ? (await getMangaDexTitle(slug)) ?? slug
      : slug.replace(/[-_]/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()));

    const baseUrls: Record<string, string> = {
      mangahere: `https://www.mangahere.cc/manga/${slug}/`,
      fanfox: `https://fanfox.net/manga/${slug}/`,
      webtoons: `https://www.webtoons.com/en/fantasy/_/list?title_no=${slug}`,
      asurascans: `https://asurascans.com/series/${slug}/`,
      thunderscans: `https://en-thunderscans.com/comics/${slug}/`,
      vortexscans: `https://vortexscans.org/series/${slug.includes("--") ? slug.slice(slug.indexOf("--") + 2) : slug}`,
      rizzfables: `https://rizzfables.com/series/${slug}`,
      mangadex: `https://mangadex.org/title/${slug}`,
      mangapill: `https://mangapill.com/manga/${slug}`,
      comick: `https://comick.io/comic/${slug}`,
      weebcentral: `https://weebcentral.com/series/${slug}`,
    };

    const manga = {
      id: resolvedId,
      title,
      description: "",
      coverUrl: null,
      status: null,
      year: null,
      originalLanguage: source === "webtoons" ? "ko" : null,
      availableTranslatedLanguages: ["en"],
      tags: [],
      contentRating: "safe",
      lastChapter: chaptersRaw[chaptersRaw.length - 1]?.chapterNum ?? null,
      source,
      externalUrl: baseUrls[source] || ext?.getBaseUrl(source, slug) || null,
    };

    const extraVoices = (ext && (isAdultReq || ext.isPrivateId(resolvedId))) ? ext.getExtraVoices() : [];
    return NextResponse.json({ manga, chapters, extraVoices, isPrivate: Boolean(ext?.isPrivateId(resolvedId)) });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to load manga: ${message}` },
      { status: 502 }
    );
  }
}
