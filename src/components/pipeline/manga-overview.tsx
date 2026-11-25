"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Calendar,
  Globe,
  Sparkles,
  Bookmark,
  BookmarkCheck,
  Loader2,
  Layers,
  ExternalLink,
} from "lucide-react";
import type { MangadexManga } from "@/types/pipeline";

interface MangaOverviewProps {
  manga: MangadexManga;
  onBack: () => void;
  onStartRecap: (resolvedManga: MangadexManga) => void;
  isBookmarked?: boolean;
  onBookmarkToggle?: () => void;
}

export function MangaOverview({
  manga,
  onBack,
  onStartRecap,
  isBookmarked = false,
  onBookmarkToggle,
}: MangaOverviewProps) {
  const [imgFailed, setImgFailed] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [chapterCount, setChapterCount] = useState<number | null>(
    manga.lastChapter ? parseInt(manga.lastChapter, 10) || null : null
  );
  const [resolvedManga, setResolvedManga] = useState<MangadexManga>(manga);

  const isMetadataOnly =
    (resolvedManga.source ?? "mangahere") === "anilist" ||
    (resolvedManga.source ?? "mangahere") === "mal";

  // Fetch exact chapter count in background if already scrapeable
  useEffect(() => {
    let cancelled = false;
    setResolvedManga(manga);
    setResolveError(null);
    setImgFailed(false);
    setChapterCount(manga.lastChapter ? parseInt(manga.lastChapter, 10) || null : null);

    const src = manga.source ?? "mangahere";
    if (src !== "anilist" && src !== "mal") {
      fetch(`/api/manga/${encodeURIComponent(manga.id)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (cancelled || !d) return;
          const chs = Array.isArray(d.chapters) ? d.chapters : [];
          if (chs.length > 0) setChapterCount(chs.length);
        })
        .catch(() => {});
    }
    return () => {
      cancelled = true;
    };
  }, [manga]);

  const handleDoRecap = async () => {
    setResolveError(null);
    if (!isMetadataOnly) {
      onStartRecap(resolvedManga);
      return;
    }

    setResolving(true);
    try {
      const sources = [
        "asurascans",
        "thunderscans",
        "vortexscans",
        "rizzfables",
        "webtoons",
        "mgeko",
        "mangahere",
        "fanfox",
      ] as const;

      let matched: MangadexManga | undefined;
      for (const s of sources) {
        const r = await fetch(
          `/api/search?q=${encodeURIComponent(manga.title)}&limit=1&source=${s}`
        );
        const d = await r.json().catch(() => ({}));
        const hit = (d.manga ?? [])[0] as MangadexManga | undefined;
        if (r.ok && hit) {
          matched = hit;
          break;
        }
      }

      if (!matched) {
        setResolveError(
          `Could not locate a live scrapeable archive for "${manga.title}". Try searching an alternate title.`
        );
        return;
      }

      const finalManga: MangadexManga = {
        ...matched,
        coverUrl: matched.coverUrl || manga.coverUrl,
        description: matched.description || manga.description,
        tags: matched.tags?.length ? matched.tags : manga.tags,
      };
      setResolvedManga(finalManga);
      onStartRecap(finalManga);
    } catch {
      setResolveError("Archive lookup timed out. Please try again.");
    } finally {
      setResolving(false);
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="max-w-5xl mx-auto space-y-6"
    >
      {/* Top Bar */}
      <div className="flex items-center justify-between gap-4">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-card border border-border hover:border-[#F5D27A]/60 text-xs font-mono uppercase tracking-[0.16em] text-foreground/85 hover:text-[#F5D27A] transition-colors cursor-pointer"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Search
        </button>

        <div className="flex items-center gap-2">
          {onBookmarkToggle && (
            <button
              type="button"
              onClick={onBookmarkToggle}
              className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-card border text-xs font-mono uppercase tracking-[0.14em] transition-colors cursor-pointer ${
                isBookmarked
                  ? "border-[#F43F5E] text-[#F43F5E]"
                  : "border-border text-muted-foreground hover:text-foreground hover:border-[#F5D27A]/50"
              }`}
            >
              {isBookmarked ? (
                <BookmarkCheck className="h-3.5 w-3.5" />
              ) : (
                <Bookmark className="h-3.5 w-3.5" />
              )}
              {isBookmarked ? "Saved" : "Save"}
            </button>
          )}
          {manga.externalUrl && (
            <a
              href={manga.externalUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-card border border-border hover:border-[#F5D27A]/50 text-xs font-mono uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors"
            >
              Source
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      </div>

      {/* Main Editorial Overview Card */}
      <div className="relative rounded-3xl overflow-hidden bg-[#160810] dark:bg-[#15070E] border border-[#F5D27A]/30 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.85)] p-6 sm:p-10">
        {/* Subtle Cherry-Gold top sheen */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-44 opacity-50"
          style={{
            background:
              "radial-gradient(ellipse 70% 100% at 50% 0%, rgba(225,29,72,0.28), transparent 72%)",
          }}
        />

        <div className="relative z-10 grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
          {/* Left: High-Res Poster Showcase */}
          <div className="md:col-span-4 lg:col-span-4">
            <div className="aspect-[3/4] rounded-2xl overflow-hidden relative border border-[#F5D27A]/45 shadow-[0_24px_60px_-14px_rgba(225,29,72,0.5)] bg-[#0E050A] group">
              {manga.coverUrl && !imgFailed ? (
                <img
                  src={`/api/cover?url=${encodeURIComponent(manga.coverUrl)}&w=640`}
                  alt={manga.title}
                  referrerPolicy="no-referrer"
                  onError={() => setImgFailed(true)}
                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center bg-gradient-to-br from-[#320A1B] to-[#0E050A]">
                  <Sparkles className="h-8 w-8 text-[#F5D27A] mb-3" />
                  <p className="font-serif italic text-xl text-[#FFF8F5]">{manga.title}</p>
                </div>
              )}

              {/* Diagonal hover foil sheen */}
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-y-0 -left-full w-2/3 -skew-x-12 bg-gradient-to-r from-transparent via-[#FFF4C6]/25 to-transparent group-hover:left-[160%] transition-all duration-700"
              />

              {/* Chapter Count Jewel */}
              {(chapterCount || manga.lastChapter) && (
                <span className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-gradient-to-r from-[#E11D48] to-[#BE123C] text-[#FFF8F5] border border-[#F5D27A]/75 shadow-lg">
                  <BookOpen className="h-3.5 w-3.5 text-[#FFF0B8]" />
                  {chapterCount ?? manga.lastChapter} Chapters
                </span>
              )}
            </div>
          </div>

          {/* Right: Manhwa Info + Easy-Access "Do the Recap" CTA */}
          <div className="md:col-span-8 lg:col-span-8 space-y-6">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-[0.18em] bg-[#240916] text-[#F5D27A] border border-[#F5D27A]/35">
                  {manga.source ?? "Archive"}
                </span>
                {manga.status && (
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-[0.16em] bg-emerald-500/15 text-emerald-300 border border-emerald-500/35">
                    {manga.status}
                  </span>
                )}
                {manga.year && (
                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-mono bg-card text-muted-foreground border border-border">
                    <Calendar className="h-3 w-3 text-[#F5D27A]" />
                    {manga.year}
                  </span>
                )}
                {manga.originalLanguage && (
                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-mono uppercase bg-card text-muted-foreground border border-border">
                    <Globe className="h-3 w-3 text-[#F43F5E]" />
                    {manga.originalLanguage}
                  </span>
                )}
              </div>

              <h1 className="font-serif text-3xl sm:text-5xl font-semibold tracking-tight leading-[1.08] text-[#FFF8F5]">
                {manga.title}
              </h1>
            </div>

            {/* Easy-to-Access Primary CTA Right Above the Fold */}
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <motion.button
                whileHover={{ scale: 1.03, y: -2 }}
                whileTap={{ scale: 0.97 }}
                type="button"
                disabled={resolving}
                onClick={handleDoRecap}
                className="h-13 px-8 rounded-full btn-sovereign text-xs sm:text-sm font-mono uppercase tracking-[0.18em] font-bold inline-flex items-center gap-2.5 cursor-pointer disabled:opacity-70"
              >
                {resolving ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Matching Live Archive…
                  </>
                ) : (
                  <>
                    <Layers className="h-4 w-4 text-[#FFF0B8]" />
                    Do the Recap
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </motion.button>

              <span className="text-[11px] font-mono text-muted-foreground">
                1920×1080 Cinema · YOLO11 Slicer · Kokoro Voice
              </span>
            </div>

            {resolveError && (
              <p className="text-xs text-rose-300 bg-rose-950/50 border border-rose-500/40 rounded-xl px-4 py-2.5">
                {resolveError}
              </p>
            )}

            {/* Description / Synopsis */}
            <div className="space-y-2 pt-2 border-t border-white/10">
              <p className="text-[10px] font-mono uppercase tracking-[0.24em] text-[#F5D27A]">
                Synopsis
              </p>
              <p className="text-sm sm:text-[15px] text-[#FFE8EC]/85 leading-relaxed whitespace-pre-line">
                {manga.description?.trim()
                  ? manga.description
                  : `Explore ${manga.title} and transform its chapters into a cinema-grade narrated recap with automated panel sculpting, speech-bubble isolation, and studio voice mastering.`}
              </p>
            </div>

            {/* Genre Tags */}
            {manga.tags && manga.tags.length > 0 && (
              <div className="space-y-2">
                <p className="text-[10px] font-mono uppercase tracking-[0.24em] text-[#F5D27A]">
                  Genres &amp; Themes
                </p>
                <div className="flex flex-wrap gap-2">
                  {manga.tags.slice(0, 10).map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 rounded-full text-[11px] font-mono bg-[#220B16] text-[#FFE8EC]/90 border border-primary/30"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.section>
  );
}
