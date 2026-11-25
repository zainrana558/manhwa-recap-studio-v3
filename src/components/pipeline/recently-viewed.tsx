"use client";

import { useState } from "react";
import { Clock, X, Sparkles, ArrowRight } from "lucide-react";
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

function RecentPosterCard({
  item,
  onOpen,
  onRemove,
}: {
  item: RecentManga;
  onOpen: () => void;
  onRemove: (e: React.MouseEvent) => void;
}) {
  const [failed, setFailed] = useState(false);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
      className="group relative w-[148px] sm:w-[162px] snap-start shrink-0 rounded-2xl overflow-hidden bg-[#160810] border border-primary/25 hover:border-[#F5D27A]/75 shadow-[0_14px_34px_-14px_rgba(0,0,0,0.8)] hover:shadow-[0_20px_44px_-12px_rgba(225,29,72,0.45)] transition-all duration-300 hover:-translate-y-1.5 cursor-pointer"
    >
      <button
        type="button"
        onClick={onRemove}
        className="absolute top-2 right-2 z-20 h-6 w-6 rounded-full bg-black/75 border border-white/15 text-white/80 hover:bg-[#E11D48] hover:text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
        aria-label={`Remove ${item.title}`}
      >
        <X className="h-3 w-3" />
      </button>

      <div className="aspect-[3/4] relative overflow-hidden">
        {item.coverUrl && !failed ? (
          <img
            src={`/api/cover?url=${encodeURIComponent(item.coverUrl)}&w=320`}
            alt={item.title}
            referrerPolicy="no-referrer"
            loading="lazy"
            decoding="async"
            onError={() => setFailed(true)}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center p-3 text-center bg-gradient-to-br from-[#2B0916] to-[#0E050A]">
            <Sparkles className="h-5 w-5 text-[#F5D27A]/70 mb-1.5" />
            <span className="font-serif italic text-xs text-[#FFF8F5]/80 line-clamp-2">
              {item.title}
            </span>
          </div>
        )}

        {/* Diagonal shine sweep */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 -left-full w-2/3 -skew-x-12 bg-gradient-to-r from-transparent via-[#FFF4C6]/25 to-transparent group-hover:left-[160%] transition-all duration-700"
        />

        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#0D0509] via-[#0D0509]/35 to-transparent" />

        <span className="absolute top-2 left-2 px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#14070D]/90 text-[#F5D27A] border border-[#F5D27A]/30">
          {relativeTime(item.viewedAt)}
        </span>

        <div className="absolute bottom-0 inset-x-0 p-3 space-y-1">
          <p className="font-serif font-medium text-sm leading-snug text-[#FFF8F5] line-clamp-2">
            {item.title}
          </p>
          <span className="inline-flex items-center gap-1 text-[9.5px] font-mono uppercase tracking-[0.14em] text-[#F5D27A] opacity-0 group-hover:opacity-100 transition-opacity">
            Open Overview
            <ArrowRight className="h-2.5 w-2.5" />
          </span>
        </div>
      </div>
    </div>
  );
}

export function RecentlyViewed({
  items,
  onSelectManga,
  onRemoveItem,
  onClearAll,
}: RecentlyViewedProps) {
  if (items.length === 0) return null;

  return (
    <section className="max-w-6xl mx-auto space-y-5">
      <div className="flex items-end justify-between gap-4 border-b border-primary/20 pb-3">
        <div className="space-y-1">
          <p className="text-[10px] font-mono uppercase tracking-[0.26em] text-[#F5D27A] flex items-center gap-2">
            <Clock className="h-3 w-3 text-[#F43F5E]" />
            Continue Screening
          </p>
          <h2 className="font-serif text-2xl sm:text-3xl font-semibold text-foreground">
            Recently Viewed
          </h2>
        </div>

        <button
          type="button"
          onClick={onClearAll}
          className="px-3 py-1.5 rounded-full text-[10px] font-mono uppercase tracking-[0.16em] text-muted-foreground hover:text-rose-300 border border-border hover:border-rose-500/40 bg-card transition-colors cursor-pointer"
        >
          Clear History
        </button>
      </div>

      <div className="flex items-stretch gap-4 overflow-x-auto snap-x snap-mandatory scrollbar-thin pb-3 pt-1">
        {items.map((item) => (
          <RecentPosterCard
            key={item.id}
            item={item}
            onOpen={() =>
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
              })
            }
            onRemove={(e) => {
              e.stopPropagation();
              onRemoveItem(item.id);
            }}
          />
        ))}
      </div>
    </section>
  );
}
