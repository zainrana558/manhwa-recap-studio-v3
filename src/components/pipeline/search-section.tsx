"use client";

import { useState, useCallback, useMemo, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import { Search, Loader2, Sparkles, ExternalLink, X, Clock, Bookmark, BookmarkCheck, ArrowUpDown, BookmarkCheckIcon, History, Heart, Info } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useSectionObserver } from "@/hooks/use-section-observer";
import { useSearchHistory } from "@/hooks/use-search-history";
import { cn } from "@/lib/utils";
import { MangaDetailCard } from "@/components/pipeline/manga-detail-card";
import type { MangadexManga, MangaSource } from "@/types/pipeline";

interface SearchSectionProps {
  onResults: (manga: MangadexManga[], query: string) => void;
  onSelectManga: (manga: MangadexManga) => void;
  externalQuery?: string | null;
  onClearResults?: () => void;
  isBookmarked?: (mangaId: string) => boolean;
  onBookmarkToggle?: (manga: MangadexManga) => void;
}

export interface SearchSectionHandle {
  clearResults: () => void;
}

type SourceFilter = "all" | MangaSource;

type StatusFilter = "all" | "ongoing" | "completed" | "hiatus";

type QuickSort = "relevance" | "updated" | "followed" | "title-az";

const SOURCE_FILTERS: { value: SourceFilter; label: string; color: string }[] = [
  { value: "all", label: "All Sources", color: "" },
  { value: "mangahere", label: "MangaHere", color: "text-emerald-400" },
  { value: "fanfox", label: "FanFox", color: "text-orange-400" },
  { value: "webtoons", label: "Webtoons", color: "text-green-400" },
  { value: "asurascans", label: "AsuraScans", color: "text-rose-400" },
  { value: "mal", label: "MAL", color: "text-sky-400" },
  { value: "anilist", label: "AniList", color: "text-fuchsia-400" },
];

const STATUS_FILTER_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "ongoing", label: "Ongoing" },
  { value: "completed", label: "Completed" },
  { value: "hiatus", label: "Hiatus" },
];

const QUICK_SORT_OPTIONS: { value: QuickSort; label: string }[] = [
  { value: "relevance", label: "Relevance" },
  { value: "updated", label: "Recently Updated" },
  { value: "followed", label: "Most Followed" },
  { value: "title-az", label: "Title A-Z" },
];

const SOURCE_BADGE_CLASSES: Record<MangaSource, string> = {
  mangahere: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  fanfox: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  webtoons: "bg-green-500/15 text-green-300 border-green-500/30",
  asurascans: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  mal: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  anilist: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30",
  mangadex: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  mangapill: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
  toonily: "bg-pink-500/15 text-pink-300 border-pink-500/30",
  comick: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  weebcentral: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
};

const SOURCE_LABEL: Record<MangaSource, string> = {
  mangahere: "MangaHere",
  fanfox: "FanFox",
  webtoons: "Webtoons",
  asurascans: "Asura",
  mal: "MAL",
  anilist: "AniList",
  mangadex: "MangaDex",
  mangapill: "MangaPill",
  toonily: "Toonily",
  comick: "Comick",
  weebcentral: "WeebCentral",
};

const CONTENT_RATING_CLASSES: Record<string, string> = {
  safe: "bg-emerald-500",
  suggestive: "bg-amber-500",
};

const CONTENT_RATING_LABEL: Record<string, string> = {
  safe: "Safe",
  suggestive: "Suggestive",
};

const STATUS_BADGE_CLASSES: Record<string, string> = {
  ongoing: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  completed: "bg-sky-500/15 text-sky-400 border-sky-500/30",
  hiatus: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  cancelled: "bg-rose-500/15 text-rose-400 border-rose-500/30",
};

const GENRE_ICONS: Record<string, string> = {
  "Action": "⚔️",
  "Adventure": "🗺️",
  "Fantasy": "🔮",
  "Romance": "💕",
  "Comedy": "😂",
  "Drama": "🎭",
  "Horror": "👻",
  "Sci-Fi": "🚀",
  "Slice of Life": "☕",
  "Mystery": "🔍",
  "Supernatural": "✨",
  "Thriller": "😱",
  "Martial Arts": "🥋",
  "Isekai": "🌀",
  "School Life": "📚",
};

interface SourceCounts {
  mangahere: number;
  fanfox: number;
  webtoons: number;
  asurascans: number;
  mal: number;
  anilist: number;
}

type SortOption = "relevance" | "title-az" | "title-za" | "year-desc" | "year-asc" | "chapters-desc";

const SORT_OPTIONS: { value: SortOption; label: string }[] = [
  { value: "relevance", label: "Relevance" },
  { value: "title-az", label: "Title A→Z" },
  { value: "title-za", label: "Title Z→A" },
  { value: "year-desc", label: "Newest" },
  { value: "year-asc", label: "Oldest" },
  { value: "chapters-desc", label: "Most Chapters" },
];

function formatTimeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

const SearchSection = forwardRef<SearchSectionHandle, SearchSectionProps>(
  function SearchSection({ onResults, onSelectManga, externalQuery, onClearResults, isBookmarked, onBookmarkToggle }, ref) {
    const { toast } = useToast();
    const { ref: sectionRef, isVisible } = useSectionObserver(0.05);
    const { history: searchHistory, addHistory, removeHistory, clearHistory } = useSearchHistory();
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(false);
    const [resolvingId, setResolvingId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [results, setResults] = useState<MangadexManga[]>([]);
    const [sourceCounts, setSourceCounts] = useState<SourceCounts | null>(null);
    const [filter, setFilter] = useState<SourceFilter>("all");
    const [sortBy, setSortBy] = useState<SortOption>("relevance");
    const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
    const [quickSort, setQuickSort] = useState<QuickSort>("relevance");
    const [hasSearched, setHasSearched] = useState(false);
    const [searchDuration, setSearchDuration] = useState<number | null>(null);
    const [inputFocused, setInputFocused] = useState(false);
    const [poppedBookmark, setPoppedBookmark] = useState<string | null>(null);
    const [focusedResultIdx, setFocusedResultIdx] = useState(-1);
    const [expandedManga, setExpandedManga] = useState<MangadexManga | null>(null);
    const searchStartTime = useRef<number>(0);
    const historyPanelRef = useRef<HTMLDivElement>(null);
    const resultsGridRef = useRef<HTMLDivElement>(null);

    // Check if any results have a source field explicitly set
    const hasExplicitSource = useMemo(() => {
      return results.some((m) => m.source !== undefined);
    }, [results]);

    // Count results by status for filter pills
    const statusCounts = useMemo(() => {
      const counts: Record<string, number> = { all: results.length, ongoing: 0, completed: 0, hiatus: 0 };
      for (const m of results) {
        const s = m.status?.toLowerCase() ?? "";
        if (s === "ongoing") counts.ongoing++;
        else if (s === "completed") counts.completed++;
        else if (s === "hiatus") counts.hiatus++;
      }
      return counts;
    }, [results]);

    // Click outside detection for history panel
    useEffect(() => {
      if (!inputFocused) return;
      function handleClickOutside(e: MouseEvent) {
        const panel = historyPanelRef.current;
        const input = document.getElementById("search-input");
        if (panel && !panel.contains(e.target as Node) && input && !input.contains(e.target as Node)) {
          setInputFocused(false);
        }
      }
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [inputFocused]);

    useImperativeHandle(ref, () => ({
      clearResults: () => {
        setResults([]);
        setSourceCounts(null);
        setHasSearched(false);
        setQuery("");
        setError(null);
        setFilter("all");
        setSortBy("relevance");
        setStatusFilter("all");
        setQuickSort("relevance");
        setSearchDuration(null);
        setExpandedManga(null);
        onClearResults?.();
      },
    }), [onClearResults]);

    useEffect(() => {
      if (externalQuery) {
        setQuery(externalQuery);
        setTimeout(() => {
          const form = document.querySelector<HTMLFormElement>('form');
          if (form) form.requestSubmit();
        }, 100);
      }
    }, [externalQuery]);

    const handleClearResults = useCallback(() => {
      setResults([]);
      setSourceCounts(null);
      setHasSearched(false);
      setQuery("");
      setError(null);
      setFilter("all");
      setSortBy("relevance");
      setStatusFilter("all");
      setQuickSort("relevance");
      setSearchDuration(null);
      setExpandedManga(null);
      onClearResults?.();
    }, [onClearResults]);

    const handleSearch = useCallback(async () => {
      const q = query.trim();
      if (!q) return;
      setLoading(true);
      setError(null);
      setHasSearched(true);
      setExpandedManga(null);
      searchStartTime.current = performance.now();

      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 20000);
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=24`, {
          signal: controller.signal,
        });
        clearTimeout(timeout);
        const elapsed = Math.round(((performance.now() - searchStartTime.current) / 1000) * 10) / 10;
        setSearchDuration(elapsed);
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Search failed (${res.status})`);
        }
        const data = await res.json();
        const manga: MangadexManga[] = data.manga ?? [];
        setResults(manga);
        setSourceCounts(data.sources ?? null);
        onResults(manga, q);
        if (manga.length > 0) addHistory(q, manga.length);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Search failed";
        if (e instanceof DOMException && e.name === "AbortError") {
          setError("Search timed out. Some sources may be slow — try again.");
        } else {
          setError(msg);
        }
        setResults([]);
        setSourceCounts(null);
        setSearchDuration(null);
      } finally {
        setLoading(false);
      }
    }, [query, onResults, addHistory]);

    const visibleResults = useMemo(() => {
      let filtered = filter === "all" ? [...results] : results.filter((m) => (m.source ?? "mangahere") === filter);

      if (statusFilter !== "all") {
        filtered = filtered.filter((m) => m.status?.toLowerCase() === statusFilter);
      }

      const activeSort = quickSort !== "relevance" ? quickSort : sortBy;

      if (activeSort === "relevance") return filtered;
      const sorted = [...filtered];
      switch (activeSort) {
        case "title-az":
          sorted.sort((a, b) => a.title.localeCompare(b.title));
          break;
        case "title-za":
          sorted.sort((a, b) => b.title.localeCompare(a.title));
          break;
        case "year-desc":
          sorted.sort((a, b) => (b.year ?? 0) - (a.year ?? 0));
          break;
        case "year-asc":
          sorted.sort((a, b) => (a.year ?? 9999) - (b.year ?? 9999));
          break;
        case "chapters-desc":
          sorted.sort((a, b) => (parseFloat(b.lastChapter ?? "0") || 0) - (parseFloat(a.lastChapter ?? "0") || 0));
          break;
        case "updated":
          sorted.sort((a, b) => {
            const tA = a.updatedAt ? new Date(a.updatedAt).getTime() : 0;
            const tB = b.updatedAt ? new Date(b.updatedAt).getTime() : 0;
            return tB - tA;
          });
          break;
        case "followed":
          sorted.sort((a, b) => (b.followedCount ?? 0) - (a.followedCount ?? 0));
          break;
      }
      return sorted;
    }, [results, filter, sortBy, statusFilter, quickSort]);

    const activeSourceCount = useMemo(() => {
      if (!sourceCounts) return 0;
      return Object.values(sourceCounts).filter((c) => c > 0).length;
    }, [sourceCounts]);

    const handleSelect = useCallback(
      async (m: MangadexManga) => {
        const source = m.source ?? "mangahere";
        if (source === "mangahere" || source === "fanfox" || source === "webtoons" || source === "asurascans") {
          onSelectManga(m);
          return;
        }

        setResolvingId(m.id);
        const findingToast = toast({
          title: "Resolving on MangaHere",
          description: `Finding on MangaHere…`,
        });

        try {
          const res = await fetch(`/api/search?q=${encodeURIComponent(m.title)}&limit=1&source=mangahere`);
          const data = await res.json().catch(() => ({}));
          const mdMatch: MangadexManga | undefined = (data.manga ?? [])[0];

          if (!res.ok || !mdMatch) {
            findingToast.update({
              id: findingToast.id,
              title: "Not found on MangaHere",
              description: `Could not find on MangaHere for scraping.`,
              variant: "destructive",
            });
            return;
          }

          findingToast.update({
            id: findingToast.id,
            title: "Matched on MangaHere",
            description: `Using "${mdMatch.title}" for scraping.`,
          });
          onSelectManga({
            ...mdMatch,
            externalUrl: mdMatch.externalUrl ?? m.externalUrl ?? null,
          });
        } catch {
          findingToast.update({
            id: findingToast.id,
            title: "Resolution failed",
            description: `Could not find on MangaHere for scraping.`,
            variant: "destructive",
          });
        } finally {
          setResolvingId(null);
        }
      },
      [onSelectManga, toast]
    );

    const handleBookmark = useCallback((e: React.MouseEvent, m: MangadexManga) => {
      e.stopPropagation();
      const wasBookmarked = isBookmarked?.(m.id) ?? false;
      onBookmarkToggle?.(m);
      setPoppedBookmark(m.id);
      setTimeout(() => setPoppedBookmark(null), 350);
      toast({
        title: wasBookmarked ? "Bookmark removed" : "Bookmarked!",
        description: wasBookmarked ? `Removed "${m.title}" from saved manga` : `Saved "${m.title}" to bookmarks`,
      });
    }, [onBookmarkToggle, isBookmarked, toast]);

    const showHistory = inputFocused && query === "" && !hasSearched;

    // Reset focused index when results change
    useEffect(() => {
      setFocusedResultIdx(-1);
    }, [visibleResults.length]);

    // Keyboard navigation for search results
    const handleInputKeyDown = useCallback((e: React.KeyboardEvent) => {
      if (visibleResults.length === 0) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setFocusedResultIdx((prev) => {
          const next = prev < visibleResults.length - 1 ? prev + 1 : 0;
          const grid = resultsGridRef.current;
          if (grid) {
            const items = grid.querySelectorAll<HTMLElement>("[data-result-index]");
            items[next]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
          }
          return next;
        });
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setFocusedResultIdx((prev) => {
          const next = prev > 0 ? prev - 1 : visibleResults.length - 1;
          const grid = resultsGridRef.current;
          if (grid) {
            const items = grid.querySelectorAll<HTMLElement>("[data-result-index]");
            items[next]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
          }
          return next;
        });
      } else if (e.key === "Enter" && focusedResultIdx >= 0 && focusedResultIdx < visibleResults.length) {
        e.preventDefault();
        handleSelect(visibleResults[focusedResultIdx]);
      } else if (e.key === "Escape") {
        setFocusedResultIdx(-1);
        (e.target as HTMLInputElement).blur();
      }
    }, [visibleResults, focusedResultIdx, handleSelect]);

    return (
      <section ref={sectionRef} className="space-y-6">
        {/* Hero with glow orbs */}
        <div className="relative">
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div
              className="hidden lg:block absolute -top-20 left-1/4 w-[300px] h-[300px] rounded-full blur-[80px] opacity-15"
              style={{
                background: "radial-gradient(circle, oklch(0.78 0.17 65), oklch(0.72 0.18 45 / 0.3), transparent)",
                animation: "float-orb-1 8s ease-in-out infinite",
              }}
            />
            <div
              className="hidden lg:block absolute -top-10 right-1/4 w-[250px] h-[250px] rounded-full blur-[80px] opacity-15"
              style={{
                background: "radial-gradient(circle, oklch(0.72 0.18 45), oklch(0.65 0.22 50 / 0.3), transparent)",
                animation: "float-orb-2 12s ease-in-out infinite",
              }}
            />
            <div
              className="hidden lg:block absolute top-10 left-1/2 -translate-x-1/2 w-[200px] h-[200px] rounded-full blur-[80px] opacity-15"
              style={{
                background: "radial-gradient(circle, oklch(0.65 0.2 25), oklch(0.72 0.18 45 / 0.3), transparent)",
                animation: "float-orb-3 15s ease-in-out infinite",
              }}
            />
          </div>

          <div className="absolute inset-0 overflow-hidden pointer-events-none lg:block hidden">
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] rounded-full opacity-[0.07]"
              style={{
                background: "radial-gradient(ellipse, oklch(0.78 0.17 65), oklch(0.65 0.2 25 / 0.3), transparent 70%)",
              }}
            />
          </div>

          <div className={`relative text-center space-y-5 transition-all duration-700 ${isVisible ? "animate-section-in" : "opacity-0"}`}>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-primary/20 bg-primary/5 mb-2">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">AI-Powered Video Pipeline</span>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight leading-tight">
              <span className="text-gradient">Manhwa Recap</span>
              <br />
              <span className="text-gradient">Studio</span>
            </h1>
            <p className="text-muted-foreground text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
              Enter any manhwa, manga, or webtoon name. We search{" "}
              <span className="text-foreground font-medium">6 sources at once</span>,
              scrape every chapter, transcribe dialogue with AI, and render a narrated recap video.
            </p>
            <div className="flex items-center justify-center gap-3 text-xs text-muted-foreground/50">
              <div className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted text-foreground/70 font-mono text-[10px] hover:bg-muted/80 transition-colors cursor-default">/</kbd>
                <span>focus</span>
              </div>
              <span className="text-muted-foreground/20">·</span>
              <div className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted text-foreground/70 font-mono text-[10px] hover:bg-muted/80 transition-colors cursor-default">Esc</kbd>
                <span>clear</span>
              </div>
              <span className="text-muted-foreground/20">·</span>
              <div className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted text-foreground/70 font-mono text-[10px] hover:bg-muted/80 transition-colors cursor-default">B</kbd>
                <span>bookmarks</span>
              </div>
            </div>
          </div>
        </div>

        {/* Glassmorphism search bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          className="flex flex-col sm:flex-row gap-3 max-w-2xl mx-auto"
        >
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input
              id="search-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setTimeout(() => setInputFocused(false), 200)}
              onKeyDown={handleInputKeyDown}
              placeholder="e.g. Solo Leveling, Tower of God, One Piece…"
              className="pl-11 h-13 text-base bg-card/80 border-border/80 backdrop-blur-sm focus:bg-card transition-all shadow-sm focus:shadow-md focus:shadow-primary/5 rounded-xl"
              autoFocus
            />
          </div>
          <Button
            type="submit"
            size="lg"
            className="h-13 px-8 font-semibold rounded-xl shadow-sm hover:shadow-md hover:shadow-primary/10 transition-all"
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Searching…
              </>
            ) : (
              <>
                <Sparkles className="h-5 w-5 mr-2" />
                Search
              </>
            )}
          </Button>
        </form>

        {/* Search history dropdown panel */}
        {showHistory && (
          <div
            ref={historyPanelRef}
            className="max-w-2xl mx-auto mt-1.5 animate-fade-in-up"
          >
            <div className="rounded-xl border border-border bg-card/80 backdrop-blur-sm shadow-lg shadow-black/20 overflow-hidden">
              {searchHistory.length > 0 ? (
                <>
                  <div className="max-h-64 overflow-y-auto">
                    {searchHistory.map((item) => (
                      <button
                        key={`${item.query}-${item.timestamp}`}
                        type="button"
                        onClick={() => {
                          setQuery(item.query);
                          setInputFocused(false);
                          setTimeout(() => {
                            const input = document.getElementById("search-input") as HTMLInputElement | null;
                            if (input) input.focus();
                            const form = input?.closest("form");
                            if (form) form.requestSubmit();
                          }, 0);
                        }}
                        className="group w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-muted/50 transition-colors border-b border-border/50 last:border-b-0"
                      >
                        <Search className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-foreground truncate group-hover:text-primary transition-colors">
                            {item.query}
                          </p>
                          <div className="flex items-center gap-2 mt-0.5">
                            <Clock className="h-2.5 w-2.5 text-muted-foreground/40" />
                            <span className="text-[11px] text-muted-foreground/50">{formatTimeAgo(item.timestamp)}</span>
                            <span className="text-[11px] text-muted-foreground/40">·</span>
                            <span className="text-[11px] text-muted-foreground/50">{item.resultCount} result{item.resultCount !== 1 ? "s" : ""}</span>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeHistory(item.query);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-muted text-muted-foreground/50 hover:text-rose-400 transition-all shrink-0"
                          aria-label={`Remove "${item.query}" from history`}
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center justify-between px-4 py-2 border-t border-border/50 bg-muted/30">
                    <div className="flex items-center gap-1.5">
                      <History className="h-3 w-3 text-muted-foreground/40" />
                      <span className="text-[11px] text-muted-foreground/50">{searchHistory.length} item{searchHistory.length !== 1 ? "s" : ""}</span>
                    </div>
                    <button
                      type="button"
                      onClick={clearHistory}
                      className="text-[11px] text-muted-foreground/50 hover:text-rose-400 transition-colors"
                    >
                      Clear all
                    </button>
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center py-6 gap-2">
                  <History className="h-5 w-5 text-muted-foreground/30" />
                  <p className="text-xs text-muted-foreground/50">No recent searches</p>
                </div>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="max-w-2xl mx-auto">
            <p className="text-center text-destructive text-sm bg-destructive/10 border border-destructive/20 rounded-lg py-2 px-4">{error}</p>
          </div>
        )}

        {/* Status filter + Quick sort pills */}
        {hasSearched && !loading && results.length > 0 && (
          <div className="flex flex-col items-center gap-2 animate-fade-in-up">
            <div className="flex flex-wrap items-center justify-center gap-1.5">
              {STATUS_FILTER_OPTIONS.map((opt) => {
                const isActive = statusFilter === opt.value;
                const count = statusCounts[opt.value] ?? 0;
                return (
                  <Button
                    key={opt.value}
                    type="button"
                    size="sm"
                    variant={isActive ? "default" : "outline"}
                    onClick={() => setStatusFilter(opt.value)}
                    className="h-7 px-2.5 text-[11px] rounded-lg"
                  >
                    {opt.label}
                    <span
                      className={`ml-1.5 rounded px-1 py-0.5 text-[9px] font-mono ${
                        isActive
                          ? "bg-primary-foreground/20 text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {count}
                    </span>
                  </Button>
                );
              })}

              <span className="text-border mx-1">|</span>

              {QUICK_SORT_OPTIONS.map((opt) => {
                const isActive = quickSort === opt.value;
                return (
                  <Button
                    key={opt.value}
                    type="button"
                    size="sm"
                    variant={isActive ? "default" : "outline"}
                    onClick={() => setQuickSort(opt.value)}
                    className="h-7 px-2.5 text-[11px] rounded-lg"
                  >
                    {opt.label}
                  </Button>
                );
              })}

              {hasExplicitSource && (
                <>
                  <span className="text-border mx-1">|</span>
                  <div className="flex items-center gap-1">
                    <select
                      value={filter}
                      onChange={(e) => setFilter(e.target.value as SourceFilter)}
                      className="h-7 px-2 text-[11px] bg-muted/50 border border-border rounded-lg text-foreground focus:outline-none focus:ring-1 focus:ring-ring appearance-none cursor-pointer pr-5 bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22currentColor%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22m6%209%206%206%206-6%22%2F%3E%3C%2Fsvg%3E')] bg-[position:right_4px_center] bg-no-repeat"
                      aria-label="Filter by source"
                    >
                      {SOURCE_FILTERS.map((f) => (
                        <option key={f.value} value={f.value}>{f.label}</option>
                      ))}
                    </select>
                  </div>
                </>
              )}

              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={handleClearResults}
                className="h-7 px-2 text-[11px] text-muted-foreground hover:text-rose-400 hover:bg-rose-500/10 rounded-lg"
              >
                <X className="h-3 w-3 mr-1" />
                Clear
              </Button>
            </div>

            {/* Source filter pills row */}
            <div className="flex flex-wrap items-center justify-center gap-1.5">
              {SOURCE_FILTERS.map((f) => {
                const isActive = filter === f.value;
                const count =
                  f.value === "all"
                    ? results.length
                    : sourceCounts
                      ? sourceCounts[f.value]
                      : results.filter((m) => (m.source ?? "mangahere") === f.value).length;
                if (f.value !== "all" && count === 0) return null;
                return (
                  <Button
                    key={f.value}
                    type="button"
                    size="sm"
                    variant={isActive ? "default" : "outline"}
                    onClick={() => setFilter(f.value)}
                    className={`h-7 px-2.5 text-[11px] rounded-lg ${!isActive && f.color ? `hover:${f.color} hover:border-current/30` : ""}`}
                  >
                    {f.label}
                    <span
                      className={`ml-1.5 rounded px-1 py-0.5 text-[9px] font-mono ${
                        isActive
                          ? "bg-primary-foreground/20 text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {count}
                    </span>
                  </Button>
                );
              })}
              <div className="flex items-center gap-1">
                <ArrowUpDown className="h-3 w-3 text-muted-foreground/60" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortOption)}
                  className="h-7 px-2 text-[11px] bg-muted/50 border border-border rounded-lg text-foreground focus:outline-none focus:ring-1 focus:ring-ring appearance-none cursor-pointer pr-5 bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22currentColor%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22m6%209%206%206%206-6%22%2F%3E%3C%2Fsvg%3E')] bg-[position:right_4px_center] bg-no-repeat"
                  aria-label="Sort results"
                >
                  {SORT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Result count summary */}
        {hasSearched && !loading && results.length > 0 && (
          <p className="text-center text-xs text-muted-foreground animate-fade-in-up">
            Showing{" "}
            <span className="text-foreground font-semibold">{visibleResults.length}</span>{" "}
            of{" "}
            <span className="text-foreground font-semibold">{results.length}</span>{" "}
            results
            {visibleResults.length !== results.length && (
              <span className="text-muted-foreground/60"> (filtered)</span>
            )}
            {activeSourceCount > 0 && (
              <> from{" "}
                <span className="text-foreground font-semibold">{activeSourceCount}</span>{" "}
                source{activeSourceCount !== 1 ? "s" : ""}
              </>
            )}
            {searchDuration !== null && (
              <> in{" "}
                <span className="text-foreground font-semibold">{searchDuration}s</span>
              </>
            )}
          </p>
        )}

        {hasSearched && !loading && visibleResults.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center py-12 space-y-3 animate-fade-in-up">
            <div className="p-4 rounded-full bg-muted/50 border border-border">
              <Search className="h-8 w-8 text-muted-foreground/40" />
            </div>
            <div className="text-center space-y-1">
              <p className="text-sm font-medium text-muted-foreground">
                {results.length === 0
                  ? "No results found"
                  : statusFilter !== "all"
                    ? `No ${statusFilter} results`
                    : `No results from ${SOURCE_FILTERS.find((f) => f.value === filter)?.label}`}
              </p>
              <p className="text-xs text-muted-foreground/60">
                {results.length === 0
                  ? "Try a different title, spelling, or browse trending picks above."
                  : statusFilter !== "all"
                    ? "Try selecting a different status filter."
                    : "Try selecting a different source filter."}
              </p>
            </div>
          </div>
        )}

        {visibleResults.length > 0 && (
          <div ref={resultsGridRef} className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
            {visibleResults.map((m, idx) => {
              const source = m.source ?? "mangahere";
              const isResolving = resolvingId === m.id;
              const isExternal = source === "mal" || source === "anilist";
              const contentRating = m.contentRating ?? "safe";
              const mangaStatus = m.status?.toLowerCase() ?? null;
              const bookmarked = isBookmarked?.(m.id) ?? false;
              const isPopped = poppedBookmark === m.id;

              const genreTags = (m.tags ?? []).slice(0, 2);

              return (
                <div
                  key={m.id}
                  data-result-index={idx}
                  onClick={() => !isResolving && handleSelect(m)}
                  role="button"
                  tabIndex={-1}
                  onMouseEnter={() => setFocusedResultIdx(idx)}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && !isResolving) {
                      e.preventDefault();
                      handleSelect(m);
                    }
                  }}
                  className={cn(
                    "group text-left space-y-2 transition-all duration-300 hover:scale-[1.03] hover:-translate-y-1 focus:outline-none focus:ring-2 focus:ring-ring rounded-lg disabled:opacity-60 disabled:hover:scale-100 cursor-pointer animate-item-in hover-glow-sm",
                    focusedResultIdx === idx && "ring-2 ring-primary/60 bg-accent/30 scale-[1.03] -translate-y-1"
                  )}
                  style={{ animationDelay: `${idx * 40}ms` }}
                >
                  <div className="aspect-[3/4] rounded-xl overflow-hidden bg-muted border border-border relative group-hover:border-primary/40 group-hover:shadow-lg group-hover:shadow-primary/5 transition-all duration-300">
                    {m.coverUrl ? (
                      <>
                        <img
                          src={m.coverUrl}
                          alt={m.title}
                          className="w-full h-full object-cover group-hover:scale-105 group-hover:brightness-110 transition-all duration-500"
                          loading="lazy"
                        />
                        <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/50 to-transparent pointer-events-none" />
                      </>
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground p-2 text-center gap-1">
                        <Search className="h-6 w-6 opacity-30" />
                        <span className="text-xs">No cover</span>
                      </div>
                    )}
                    <span
                      className={`absolute top-1.5 left-1.5 text-[10px] font-semibold px-1.5 py-0.5 rounded-md border backdrop-blur-sm ${SOURCE_BADGE_CLASSES[source]}`}
                    >
                      {isResolving ? "…" : SOURCE_LABEL[source]}
                    </span>
                    <span
                      className={`absolute top-1.5 right-1.5 h-2 w-2 rounded-full ${CONTENT_RATING_CLASSES[contentRating] ?? "bg-emerald-500"}`}
                      title={CONTENT_RATING_LABEL[contentRating] ?? contentRating}
                    />
                    {bookmarked && (
                      <span className="absolute top-1.5 right-7 flex items-center justify-center">
                        <Heart className="h-3 w-3 text-rose-400 fill-rose-400 drop-shadow-sm" />
                      </span>
                    )}
                    {m.lastChapter && (
                      <span className="absolute bottom-1.5 left-1.5 text-[9px] font-semibold px-1.5 py-0.5 rounded-md bg-black/60 backdrop-blur-sm text-white/90 border border-white/10">
                        Ch.{m.lastChapter}
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={(e) => handleBookmark(e, m)}
                      className={`absolute bottom-1.5 right-1.5 p-1.5 rounded-lg bg-black/50 backdrop-blur-sm text-white/70 hover:text-primary hover:bg-black/70 transition-all z-10 ${bookmarked ? "text-primary" : "opacity-0 group-hover:opacity-100"} ${isPopped ? "animate-bookmark-pop" : ""}`}
                      aria-label={bookmarked ? `Remove ${m.title} from bookmarks` : `Bookmark ${m.title}`}
                    >
                      {bookmarked ? <BookmarkCheck className="h-3.5 w-3.5" /> : <Bookmark className="h-3.5 w-3.5" />}
                    </button>
                    {isExternal && m.externalUrl && (
                      <a
                        href={m.externalUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="absolute top-1.5 right-1.5 p-1 rounded bg-black/50 text-white/80 hover:text-white hover:bg-black/70 transition z-10"
                        aria-label={`View ${SOURCE_LABEL[source]} page`}
                      >
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/10 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 flex flex-col items-end justify-between p-2">
                      <div className="flex flex-wrap gap-1 justify-end">
                        {genreTags.map((tag) => (
                          <span key={tag} className="text-[9px] px-1.5 py-0.5 rounded-md bg-black/40 backdrop-blur-sm text-white/80 border border-white/10">
                            {GENRE_ICONS[tag] ?? ""} {tag}
                          </span>
                        ))}
                      </div>
                      <div className="flex items-center gap-2 w-full justify-between">
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); setExpandedManga(m); }}
                          className="flex items-center gap-1 text-[10px] text-white/70 hover:text-white px-1.5 py-0.5 rounded-md bg-white/10 backdrop-blur-sm hover:bg-white/20 transition z-20"
                          aria-label={`Show details for ${m.title}`}
                        >
                          <Info className="h-3 w-3" />
                          Info
                        </button>
                        <span className="text-white text-xs font-medium">
                          {isResolving
                            ? "Finding on MangaHere…"
                            : isExternal
                              ? "Match on MangaDex →"
                              : "Select →"}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium line-clamp-2 leading-tight group-hover:text-primary transition-colors">
                      {m.title}
                    </p>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <p className="text-xs text-muted-foreground">
                        {m.year ? `${m.year}` : ""}
                        {m.year && m.originalLanguage ? " · " : ""}
                        {m.originalLanguage?.toUpperCase() ?? (isExternal ? SOURCE_LABEL[source] : "?")}
                        {m.lastChapter ? ` · Ch.${m.lastChapter}` : ""}
                      </p>
                    </div>
                    {mangaStatus && STATUS_BADGE_CLASSES[mangaStatus] && (
                      <span
                        className={`inline-block text-[9px] font-medium px-1.5 py-0.5 rounded-md border ${STATUS_BADGE_CLASSES[mangaStatus]}`}
                      >
                        {mangaStatus.charAt(0).toUpperCase() + mangaStatus.slice(1)}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {hasSearched && !loading && results.length > 0 && (
          <p className="text-center text-xs text-muted-foreground/60">
            {sourceCounts
              ? `MangaHere: ${sourceCounts.mangahere} · FanFox: ${sourceCounts.fanfox} · Webtoons: ${sourceCounts.webtoons} · AsuraScans: ${sourceCounts.asurascans} · MAL: ${sourceCounts.mal} · AniList: ${sourceCounts.anilist}`
              : ""}{" "}
            — metadata-only results (MAL/AniList) are auto-matched to a scrapeable source on selection.
          </p>
        )}

        <MangaDetailCard
          manga={expandedManga}
          open={expandedManga !== null}
          onClose={() => setExpandedManga(null)}
          onSelect={(m) => { setExpandedManga(null); handleSelect(m); }}
          isBookmarked={expandedManga ? (isBookmarked?.(expandedManga.id) ?? false) : false}
          onBookmarkToggle={expandedManga ? () => {
            const m = expandedManga;
            const wasBookmarked = isBookmarked?.(m.id) ?? false;
            onBookmarkToggle?.(m);
            setPoppedBookmark(m.id);
            setTimeout(() => setPoppedBookmark(null), 350);
            toast({
              title: wasBookmarked ? "Bookmark removed" : "Bookmarked!",
              description: wasBookmarked ? `Removed "${m.title}" from saved manga` : `Saved "${m.title}" to bookmarks`,
            });
          } : undefined}
        />
      </section>
    );
  }
);

export { SearchSection };
