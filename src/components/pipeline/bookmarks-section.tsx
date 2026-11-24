"use client";

import { Bookmark, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useBookmarks, type Bookmark as BookmarkType } from "@/hooks/use-bookmarks";
import { useSectionObserver } from "@/hooks/use-section-observer";
import type { MangadexManga } from "@/types/pipeline";

interface BookmarksSectionProps {
  onSelectManga: (manga: MangadexManga) => void;
}

const SOURCE_LABEL: Record<string, string> = {
  mangahere: "MangaHere",
  fanfox: "FanFox",
  webtoons: "Webtoons",
  asurascans: "Asura",
  thunderscans: "Thunder",
  vortexscans: "Vortex",
  rizzfables: "Rizz",
  mal: "MAL",
  anilist: "AniList",
};

const SOURCE_COLORS: Record<string, string> = {
  mangahere: "text-emerald-400",
  fanfox: "text-orange-400",
  webtoons: "text-green-400",
  asurascans: "text-rose-400",
  thunderscans: "text-amber-400",
  vortexscans: "text-purple-400",
  rizzfables: "text-pink-400",
  mal: "text-sky-400",
  anilist: "text-fuchsia-400",
};

function timeAgo(ts: number): string {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function BookmarksSection({ onSelectManga }: BookmarksSectionProps) {
  const { bookmarks, removeBookmark, clearAll } = useBookmarks();
  const { ref, isVisible } = useSectionObserver(0.1);

  return (
    <section ref={ref} className="max-w-5xl mx-auto">
      <div
        className={`transition-all duration-700 ${isVisible ? "animate-section-in" : "opacity-0"}`}
      >
        <div className="relative">
          <div className="absolute inset-x-0 top-0 h-[2px] rounded-full bg-gradient-to-r from-primary/30 via-primary/10 to-transparent" />

          <div className="flex items-center justify-between mb-3 pt-1">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-primary/10">
                <Bookmark className="h-4 w-4 text-glow" />
              </div>
              <h2 className="text-sm font-medium text-muted-foreground">
                Saved Manga
              </h2>
              <span className="text-xs text-muted-foreground/60">
                ({bookmarks.length})
              </span>
            </div>
            {bookmarks.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAll}
                className="h-7 px-2 text-xs text-muted-foreground hover:text-rose-400 hover:bg-rose-500/10"
              >
                <X className="h-3 w-3 mr-1" />
                Clear all
              </Button>
            )}
          </div>
        </div>

        {bookmarks.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 gap-3 animate-breathe">
            <div className="relative">
              <div className="w-16 h-16 rounded-full bg-primary/5 border border-primary/10 flex items-center justify-center animate-morph" />
              <Bookmark className="h-6 w-6 text-primary/20 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
            </div>
            <p className="text-xs text-muted-foreground/50 text-center">
              No saved manga yet. Search and bookmark your favorites.
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-thin">
            {bookmarks.map((b, i) => (
              <div
                key={b.id}
                role="button"
                tabIndex={0}
                onClick={() => {
                  onSelectManga({
                    id: b.id,
                    title: b.title,
                    coverUrl: b.coverUrl,
                    source: b.source as MangadexManga["source"],
                    description: "",
                    status: b.status,
                    year: b.year,
                    originalLanguage: null,
                    availableTranslatedLanguages: [],
                    tags: [],
                    contentRating: null,
                    lastChapter: b.lastChapter,
                  });
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelectManga({
                      id: b.id,
                      title: b.title,
                      coverUrl: b.coverUrl,
                      source: b.source as MangadexManga["source"],
                      description: "",
                      status: b.status,
                      year: b.year,
                      originalLanguage: null,
                      availableTranslatedLanguages: [],
                      tags: [],
                      contentRating: null,
                      lastChapter: b.lastChapter,
                    });
                  }
                }}
                className={
                  `flex items-center gap-3 p-2.5 rounded-xl border border-border bg-card/50 hover:border-primary/30 hover:bg-card/80 transition-all duration-200 text-left group cursor-pointer hover-lift glass-card-hover ${isVisible ? "animate-item-in" : "opacity-0"}`
                }
                style={{ animationDelay: isVisible ? `${i * 60}ms` : "0ms" }}
              >
                <div className="w-8 h-11 rounded-lg overflow-hidden bg-muted flex-shrink-0 border border-border">
                  {b.coverUrl ? (
                    <img src={b.coverUrl} alt="" referrerPolicy="no-referrer" className="w-full h-full object-cover" loading="lazy" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Bookmark className="h-3 w-3 text-muted-foreground/30" />
                    </div>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate group-hover:text-primary transition-colors">
                    {b.title}
                  </p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                    <span className={SOURCE_COLORS[b.source] ?? "text-muted-foreground"}>
                      {SOURCE_LABEL[b.source] ?? b.source}
                    </span>
                    {b.year && <span>{b.year}</span>}
                    {b.lastChapter && <span>Ch.{b.lastChapter}</span>}
                    <span className="text-muted-foreground/40">·</span>
                    <span>{timeAgo(b.addedAt)}</span>
                  </div>
                </div>

                <button
                  type="button"
                  aria-label={`Remove ${b.title} from bookmarks`}
                  onClick={(e) => {
                    e.stopPropagation();
                    removeBookmark(b.id);
                  }}
                  className="p-1.5 rounded-md text-muted-foreground hover:text-rose-400 hover:bg-rose-500/10 transition flex-shrink-0 opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
