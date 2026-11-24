"use client";

import { Clock, X } from "lucide-react";
import type { RecentManga } from "@/hooks/use-recently-viewed";
import type { MangadexManga } from "@/types/pipeline";

interface RecentlyViewedProps {
  items: RecentManga[];
  onSelectManga: (manga: MangadexManga) => void;
  onRemoveItem: (id: string) => void;
  onClearAll: () => void;
}

function relativeTime(timestamp: number): string {
  const now = Date.now();
  const diffMs = now - timestamp;
  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

export function RecentlyViewed({
  items,
  onSelectManga,
  onRemoveItem,
  onClearAll,
}: RecentlyViewedProps) {
  if (items.length === 0) {
    return (
      <section className="animate-fade-in-up">
        <div className="max-w-6xl mx-auto">
          <div className="relative">
            <div className="absolute inset-x-0 top-0 h-[2px] rounded-full bg-gradient-to-r from-primary/20 via-primary/5 to-transparent" />
            <div className="flex items-center gap-2 mb-3 pt-1">
              <Clock className="h-4 w-4 text-muted-foreground/40" />
              <h2 className="text-sm font-semibold text-muted-foreground/40">Recently Viewed</h2>
            </div>
            <div className="flex items-center justify-center py-6 animate-breathe">
              <p className="text-xs text-muted-foreground/30">Viewed manga will appear here</p>
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="animate-fade-in-up">
      <div className="max-w-6xl mx-auto">
        <div className="relative">
          <div className="absolute inset-x-0 top-0 h-[2px] rounded-full bg-gradient-to-r from-primary/30 via-primary/10 to-transparent" />
          <div className="flex items-center gap-2 mb-3 pt-1">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-muted-foreground">Recently Viewed</h2>
          </div>
        </div>

        <div className="flex items-center gap-3 overflow-x-auto snap-x snap-mandatory scrollbar-thin pb-2 hover-glow-sm rounded-xl">
          {items.map((item) => (
            <div
              key={item.id}
              className="group relative flex flex-col items-center gap-1.5 snap-start shrink-0 cursor-pointer transition-transform duration-200 hover:scale-105"
              onClick={() => {
                onSelectManga({
                  id: item.id,
                  title: item.title,
                  coverUrl: item.coverUrl || null,
                  description: "",
                  status: null,
                  year: null,
                  originalLanguage: null,
                  availableTranslatedLanguages: [],
                  tags: [],
                  contentRating: null,
                  lastChapter: null,
                });
              }}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveItem(item.id);
                }}
                className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-destructive text-destructive-foreground flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-10"
                aria-label={`Remove ${item.title}`}
              >
                <X className="h-2.5 w-2.5" />
              </button>

              <div className="w-12 h-[68px] rounded-lg overflow-hidden bg-muted border border-border group-hover:border-primary/30 transition-colors">
                {item.coverUrl ? (
                  <img
                    src={item.coverUrl}
                    alt={item.title}
                    referrerPolicy="no-referrer"
                    onError={(e) => {
                      const img = e.currentTarget;
                      img.style.display = "none";
                      const sib = img.nextElementSibling as HTMLElement | null;
                      if (sib) sib.style.display = "flex";
                    }}
                    className="w-full h-full object-cover"
                  />
                ) : null}
                <div
                  className="w-full h-full items-center justify-center text-muted-foreground/30"
                  style={{ display: item.coverUrl ? "none" : "flex" }}
                >
                  <Clock className="h-4 w-4" />
                </div>
              </div>

              <p className="text-[10px] text-muted-foreground w-[72px] text-center leading-tight line-clamp-2 group-hover:text-primary transition-colors">
                {item.title}
              </p>

              <span className="text-[9px] text-muted-foreground/50">
                {relativeTime(item.viewedAt)}
              </span>
            </div>
          ))}

          <button
            onClick={(e) => {
              e.stopPropagation();
              onClearAll();
            }}
            className="shrink-0 flex flex-col items-center gap-1.5 snap-start px-3 py-2 rounded-lg border border-dashed border-border hover:border-primary/40 hover:bg-primary/5 transition-colors"
          >
            <X className="h-4 w-4 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground">Clear all</span>
          </button>
        </div>
      </div>
    </section>
  );
}
