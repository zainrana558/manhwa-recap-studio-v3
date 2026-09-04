"use client";

import { useMemo } from "react";
import { X, ExternalLink, BookOpen, Globe, Calendar, Shield, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { MangadexManga, MangaSource } from "@/types/pipeline";

interface MangaDetailCardProps {
  manga: MangadexManga | null;
  open: boolean;
  onClose: () => void;
  onSelect: (manga: MangadexManga) => void;
  isBookmarked?: boolean;
  onBookmarkToggle?: () => void;
}

const SOURCE_LABEL: Record<MangaSource, string> = {
  mangahere: "MangaHere",
  fanfox: "FanFox",
  webtoons: "Webtoons",
  asurascans: "AsuraScans",
  mal: "MAL",
  anilist: "AniList",
  mangadex: "MangaDex",
  mangapill: "MangaPill",
  toonily: "Toonily",
  comick: "Comick",
  weebcentral: "WeebCentral",
};

const STATUS_CLASSES: Record<string, string> = {
  ongoing: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  completed: "bg-sky-500/15 text-sky-400 border-sky-500/30",
  hiatus: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  cancelled: "bg-rose-500/15 text-rose-400 border-rose-500/30",
};

const GENRE_ICONS: Record<string, string> = {
  Action: "⚔️",
  Adventure: "🗺️",
  Fantasy: "🔮",
  Romance: "💕",
  Comedy: "😂",
  Drama: "🎭",
  Horror: "👻",
  "Sci-Fi": "🚀",
  "Slice of Life": "☕",
  Mystery: "🔍",
  Supernatural: "✨",
  Thriller: "😱",
  "Martial Arts": "🥋",
  Isekai: "🌀",
  "School Life": "📚",
};

export function MangaDetailCard({ manga, open, onClose, onSelect, isBookmarked, onBookmarkToggle }: MangaDetailCardProps) {
  const source = manga?.source ?? "mangahere";
  const isExternal = manga ? (source === "mal" || source === "anilist") : false;
  const mangaStatus = manga?.status?.toLowerCase() ?? null;
  const contentRating = manga?.contentRating ?? "safe";

  const langDisplay = useMemo(() => {
    if (!manga?.availableTranslatedLanguages?.length) return null;
    const labelMap: Record<string, string> = {
      en: "English", ja: "Japanese", ko: "Korean", zh: "Chinese",
      es: "Spanish", fr: "French", de: "German", pt: "Portuguese",
      it: "Italian", ru: "Russian", ar: "Arabic", th: "Thai",
      vi: "Vietnamese", id: "Indonesian", tl: "Filipino", pl: "Polish",
      tr: "Turkish", hu: "Hungarian", nl: "Dutch", cs: "Czech",
    };
    return manga.availableTranslatedLanguages
      .map((l) => labelMap[l] ?? l.toUpperCase())
      .slice(0, 12);
  }, [manga]);

  if (!open || !manga) return null;

  return (
    <div className="max-w-3xl mx-auto mt-4 animate-fade-in-up">
      <div className="rounded-xl border border-border bg-popover overflow-hidden shadow-lg shadow-black/10">
        <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
          <div className="flex items-center gap-2">
            <BookOpen className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-semibold uppercase tracking-widest text-primary">Manga Details</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
            aria-label="Close details"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-4 sm:p-6">
          <div className="flex flex-col sm:flex-row gap-4 sm:gap-5">
            <div className="shrink-0 self-start">
              <div className="w-28 h-40 sm:w-32 sm:h-44 rounded-xl overflow-hidden border border-border bg-muted shadow-md">
                {manga.coverUrl ? (
                  <img
                    src={manga.coverUrl}
                    alt={manga.title}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-muted-foreground/30">
                    <BookOpen className="h-8 w-8" />
                  </div>
                )}
              </div>
            </div>

            <div className="flex-1 min-w-0 space-y-3">
              <div>
                <h3 className="text-lg font-semibold text-foreground leading-tight mb-1">
                  {manga.title}
                </h3>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline" className="text-[10px] font-medium border">
                    {SOURCE_LABEL[source] ?? source}
                  </Badge>
                  {mangaStatus && STATUS_CLASSES[mangaStatus] && (
                    <Badge variant="outline" className={cn("text-[10px] font-medium border", STATUS_CLASSES[mangaStatus])}>
                      {mangaStatus.charAt(0).toUpperCase() + mangaStatus.slice(1)}
                    </Badge>
                  )}
                  {manga.year && (
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Calendar className="h-3 w-3" />
                      {manga.year}
                    </span>
                  )}
                  {manga.lastChapter && (
                    <span className="text-xs text-muted-foreground">
                      Ch. {manga.lastChapter}
                    </span>
                  )}
                </div>
              </div>

              {manga.description && (
                <div>
                  <p className="text-xs text-muted-foreground leading-relaxed line-clamp-4 sm:line-clamp-5 whitespace-pre-line">
                    {manga.description.replace(/<[^>]*>/g, "")}
                  </p>
                </div>
              )}

              <div className="flex flex-wrap gap-1.5">
                {manga.tags.map((tag) => (
                  <Badge
                    key={tag}
                    variant="secondary"
                    className="text-[10px] font-medium px-2 py-0.5 rounded-md"
                  >
                    {GENRE_ICONS[tag] ?? ""} {tag}
                  </Badge>
                ))}
              </div>

              <div className="space-y-2">
                {manga.originalLanguage && (
                  <div className="flex items-center gap-2 text-xs">
                    <Globe className="h-3 w-3 text-muted-foreground/60" />
                    <span className="text-muted-foreground">Original:</span>
                    <span className="text-foreground font-medium uppercase">{manga.originalLanguage}</span>
                  </div>
                )}

                {langDisplay && langDisplay.length > 0 && (
                  <div className="flex items-start gap-2 text-xs">
                    <Shield className="h-3 w-3 text-muted-foreground/60 mt-0.5" />
                    <div className="flex-1">
                      <span className="text-muted-foreground">Translated: </span>
                      <span className="text-foreground">{langDisplay.join(", ")}</span>
                      {(manga.availableTranslatedLanguages?.length ?? 0) > 12 && (
                        <span className="text-muted-foreground">
                          +{manga.availableTranslatedLanguages!.length - 12} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2 pt-1">
                <Button
                  size="sm"
                  className="rounded-lg"
                  onClick={() => onSelect(manga)}
                >
                  {isExternal ? "Select & Match" : "Select"}
                </Button>
                {isExternal && manga.externalUrl && (
                  <a
                    href={manga.externalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
                  >
                    <ExternalLink className="h-3 w-3" />
                    View on {SOURCE_LABEL[source]}
                  </a>
                )}
                {onBookmarkToggle && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="rounded-lg ml-auto"
                    onClick={onBookmarkToggle}
                  >
                    {isBookmarked ? "Saved" : "Save"}
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
