"use client";

import { useState, useCallback, useMemo, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import { Search, Loader2, Sparkles, ExternalLink, X, Clock, Bookmark, BookmarkCheck, ArrowUpDown, BookmarkCheckIcon, History, Heart, Info, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";
import { AtelierCinemaDeck, SearchResults3DCoverflow, Tilt3DResultCard, TypedCinemaTitle, EditorialGenreAtelier, type GenrePreset } from "@/components/pipeline/atelier-stage";
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
  { value: "asurascans", label: "AsuraScans", color: "text-rose-400" },
  { value: "thunderscans", label: "ThunderScans", color: "text-amber-400" },
  { value: "vortexscans", label: "VortexScans", color: "text-purple-400" },
  { value: "rizzfables", label: "RizzFables", color: "text-pink-400" },
  { value: "webtoons", label: "Webtoons", color: "text-green-400" },
  { value: "mgeko", label: "MangaGeko", color: "text-teal-400" },
  { value: "mangadex", label: "MangaDex", color: "text-orange-400" },
  { value: "mangahere", label: "MangaHere", color: "text-emerald-400" },
  { value: "fanfox", label: "FanFox", color: "text-orange-400" },
  { value: "comick", label: "Comick", color: "text-violet-400" },
  { value: "weebcentral", label: "WeebCentral", color: "text-indigo-400" },
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
  thunderscans: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  vortexscans: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  rizzfables: "bg-pink-500/15 text-pink-300 border-pink-500/30",
  mal: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  anilist: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30",
  mangadex: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  mangapill: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
  comick: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  weebcentral: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
  mgeko: "bg-teal-500/15 text-teal-300 border-teal-500/30",
};

const SOURCE_LABEL: Record<MangaSource, string> = {
  mangahere: "MangaHere",
  fanfox: "FanFox",
  webtoons: "Webtoons",
  asurascans: "Asura",
  thunderscans: "Thunder",
  vortexscans: "Vortex",
  rizzfables: "Rizz",
  mal: "MAL",
  anilist: "AniList",
  mangadex: "MangaDex",
  mangapill: "MangaPill",
  comick: "Comick",
  weebcentral: "WeebCentral",
  mgeko: "MangaGeko",
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
  thunderscans?: number;
  vortexscans?: number;
  rizzfables?: number;
  mangadex: number;
  mangapill: number;
  comick: number;
  weebcentral: number;
  mgeko: number;
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
    const [privateMode, setPrivateMode] = useState<{ genres: string[]; prefixes: string[]; sourceLabels: Record<string, { label: string; badge: string; color: string }> } | null>(null);
    const [extQuery, setExtQuery] = useState("");
    const [extGenre, setExtGenre] = useState("");
    const [lastSearchPrivate, setLastSearchPrivate] = useState(false);
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
    const [resultsViewMode, setResultsViewMode] = useState<"3d-both" | "3d-stage" | "3d-grid">("3d-both");
    const [activeGenre, setActiveGenre] = useState<string>("all");
    const [displayLimit, setDisplayLimit] = useState<number>(24);
    const [placeholderIdx, setPlaceholderIdx] = useState(0);
    const PLACEHOLDERS = [
      "Search Nano Machine…",
      "Search Solo Leveling…",
      "Search Omniscient Reader…",
      "Search Tower of God…",
      "Search The Beginning After the End…",
    ];
    useEffect(() => {
      const id = setInterval(() => {
        setPlaceholderIdx((i) => (i + 1) % 5);
      }, 3200);
      return () => clearInterval(id);
    }, []);
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

    const runSearch = useCallback(async (rawQ: string, isExt = false, rawGenre = "") => {
      const q = rawQ.trim();
      const g = rawGenre.trim();
      if (!q && !(isExt && g)) return;
      setLoading(true);
      setError(null);
      if (isExt) setLastSearchPrivate(true);
      else setLastSearchPrivate(false);
      setHasSearched(true);
      setExpandedManga(null);
      searchStartTime.current = performance.now();

      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 25000);
        const params = new URLSearchParams({ limit: "24" });
        if (q) params.set("q", q);
        if (isExt) params.set("adult", "1");
        if (isExt && g) params.set("genre", g);
        const res = await fetch(`/api/search?${params.toString()}`, {
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
        if (data.unlockPrivate) {
          setPrivateMode({ genres: data.genres ?? [], prefixes: data.prefixes ?? [], sourceLabels: data.sourceLabels ?? {} });
          setQuery("");
          setHasSearched(false);
          setError(null);
          return;
        }
        const manga: MangadexManga[] = data.manga ?? [];
        setResults(manga);
        setSourceCounts(data.sources ?? null);
        const label = [q, g ? `[${g}]` : ""].filter(Boolean).join(" ");
        onResults(manga, label);
        if (!isExt && manga.length > 0) addHistory(label, manga.length);
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
    }, [onResults, addHistory]);

    const handleSearch = useCallback(async () => {
      await runSearch(query, false);
    }, [query, runSearch]);

    const handleExtSearch = useCallback(async (overrideGenre?: string) => {
      const g = overrideGenre !== undefined ? overrideGenre : extGenre;
      await runSearch(extQuery, true, g);
    }, [extQuery, extGenre, runSearch]);

    const handleHidePrivate = useCallback(() => {
      const prefixes = privateMode?.prefixes ?? [];
      setPrivateMode(null);
      setExtQuery("");
      setExtGenre("");
      if (lastSearchPrivate) {
        setResults([]);
        setSourceCounts(null);
        setHasSearched(false);
        setExpandedManga(null);
        setLastSearchPrivate(false);
        onClearResults?.();
      }
      try {
        const raw = localStorage.getItem("manhwa-recently-viewed");
        if (raw && prefixes.length > 0) {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed)) {
            const cleaned = parsed.filter((item: any) => !prefixes.some((p) => String(item?.id || "").startsWith(p)));
            localStorage.setItem("manhwa-recently-viewed", JSON.stringify(cleaned));
          }
        }
      } catch {}
      fetch("/api/jobs?scope=private", { method: "DELETE" }).catch(() => {});
    }, [privateMode, lastSearchPrivate, onClearResults]);

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
        // Open the Quick Manhwa Overview page immediately for all sources;
        // MangaOverview resolves MAL/AniList to a live archive when "Do the Recap" is clicked.
        onSelectManga(m);
        return;

        setResolvingId(m.id);
        const findingToast = toast({
          title: "Resolving a scrapeable source",
          description: `Matching "${m.title}" to a source with chapter images…`,
        });

        try {
          // Try the sources that actually work from this host, in order.
          let mdMatch: MangadexManga | undefined;
          for (const s of ["asurascans", "thunderscans", "vortexscans", "rizzfables", "mgeko", "mangadex", "mangahere", "fanfox"] as const) {
            const r = await fetch(`/api/search?q=${encodeURIComponent(m.title)}&limit=1&source=${s}`);
            const d = await r.json().catch(() => ({}));
            const hit: MangadexManga | undefined = (d.manga ?? [])[0];
            if (r.ok && hit) { mdMatch = hit; break; }
          }

          if (!mdMatch) {
            findingToast.update({
              id: findingToast.id,
              title: "No scrapeable source found",
              description: `Could not match "${m.title}" to a source with chapter images.`,
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
      setDisplayLimit(24);
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
        {/* Centered Editorial Hero — Typed-Out Title + Sculpted Search + 5-Card Real-Cover 3D Deck */}
        <div className="relative pt-2 sm:pt-6 pb-4 space-y-10">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            {/* Centered Editorial Eyebrow with flanking gold rules */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              className="inline-flex items-center justify-center gap-3.5"
            >
              <span className="h-[1px] w-10 bg-gradient-to-r from-transparent to-[#F5D27A]" />
              <span className="text-[10.5px] font-mono uppercase tracking-[0.28em] text-[#F5D27A]">
                Edition III · Private Screening House
              </span>
              <span className="h-[1px] w-10 bg-gradient-to-l from-transparent to-[#F5D27A]" />
            </motion.div>

            {/* Centered Sculptural Headline with Live Typed-Out Animation */}
            <div className="space-y-1">
              <motion.h1
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
                className="font-serif text-5xl sm:text-6xl lg:text-[4.6rem] font-semibold tracking-tight leading-[1.02] text-foreground"
              >
                Turn Panels
              </motion.h1>
              <div className="text-4xl sm:text-6xl lg:text-[4.5rem] leading-[1.08] tracking-tight">
                <TypedCinemaTitle />
              </div>
            </div>

            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
              className="text-muted-foreground text-sm sm:text-base max-w-xl mx-auto leading-relaxed"
            >
              Search fourteen live archives, sculpt every panel with neural segmentation, and master narrated widescreen editions with studio-grade cadence.
            </motion.p>

            {/* Centered Sculpted Floating Command Search Bar */}
            <motion.form
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.24, ease: [0.22, 1, 0.36, 1] }}
              onSubmit={(e) => {
                e.preventDefault();
                handleSearch();
              }}
              className="max-w-2xl mx-auto"
            >
              <div className="group flex items-center gap-2 p-1.5 pl-5 rounded-full bg-card border border-primary/45 shadow-[0_22px_54px_-16px_rgba(225,29,72,0.38),0_0_0_1px_rgba(245,210,122,0.18)] focus-within:border-[#F5D27A] focus-within:shadow-[0_26px_60px_-14px_rgba(244,63,94,0.52)] transition-all duration-300">
                <Search className="h-4 w-4 text-primary shrink-0 transition-transform duration-300 group-focus-within:scale-110" />
                <Input
                  id="search-input"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onFocus={() => setInputFocused(true)}
                  onBlur={() => setTimeout(() => setInputFocused(false), 200)}
                  onKeyDown={handleInputKeyDown}
                  placeholder={PLACEHOLDERS[placeholderIdx]}
                  className="h-11 border-0 !bg-transparent dark:!bg-transparent shadow-none focus-visible:ring-0 focus-visible:border-0 px-2 text-sm sm:text-base placeholder:text-muted-foreground/65"
                  autoFocus
                />
                <Button
                  type="submit"
                  size="lg"
                  className="h-11 px-6 rounded-full btn-sovereign border-0 shrink-0 group/btn"
                  disabled={loading || !query.trim()}
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      <span className="tracking-[0.14em] uppercase text-[11px] font-semibold">Searching</span>
                    </>
                  ) : (
                    <span className="inline-flex items-center gap-2 tracking-[0.16em] uppercase text-[11px] font-bold">
                      Explore
                      <ArrowRight className="h-3.5 w-3.5 transition-transform duration-300 group-hover/btn:translate-x-1" />
                    </span>
                  )}
                </Button>
              </div>
            </motion.form>

            {/* Unique Editorial Genre Atelier */}
            <EditorialGenreAtelier
              activeGenre={activeGenre}
              onSelectGenre={(g: GenrePreset) => {
                setActiveGenre(g.id);
                if (g.id === "all") {
                  if (hasSearched) handleClearResults();
                  return;
                }
                setQuery(g.query);
                runSearch(g.query, false);
              }}
            />
          </div>

          {/* Centered 9-Card Tiltable Real-Cover 3D Recommendation Deck (Direct Recap Open!) */}
          <div className="pt-2">
            <AtelierCinemaDeck onSelectManga={handleSelect} />
          </div>
        </div>

        {privateMode && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleExtSearch();
            }}
            className="max-w-2xl mx-auto mt-3 p-3 rounded-xl border border-rose-500/40 bg-rose-950/20  space-y-2.5 animate-fade-in-up"
          >
            <div className="flex flex-col sm:flex-row gap-2.5">
              <div className="relative flex-1 flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40 shrink-0">
                  Private
                </span>
                <Input
                  id="ext-search-input"
                  value={extQuery}
                  onChange={(e) => setExtQuery(e.target.value)}
                  placeholder="Title (optional if category set)..."
                  className="h-11 text-sm bg-card/80 border-rose-500/30 rounded-lg"
                  autoFocus
                />
              </div>
              <Input
                id="ext-genre-input"
                value={extGenre}
                onChange={(e) => setExtGenre(e.target.value)}
                placeholder="Category / Genre..."
                className="sm:w-56 h-11 text-sm bg-card/80 border-rose-500/30 rounded-lg"
              />
              <div className="flex gap-2">
                <Button
                  type="submit"
                  size="default"
                  className="h-11 px-5 font-semibold rounded-lg bg-rose-600 hover:bg-rose-500 text-white"
                  disabled={loading || (!extQuery.trim() && !extGenre.trim())}
                >
                  Search
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="default"
                  className="h-11 px-3 text-xs text-muted-foreground hover:text-foreground"
                  onClick={handleHidePrivate}
                >
                  Hide
                </Button>
              </div>
            </div>
            {privateMode.genres.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-rose-300/70 mr-1">Categories:</span>
                {privateMode.genres.map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => {
                      setExtGenre(tag);
                      handleExtSearch(tag);
                    }}
                    className={`px-2 py-0.5 rounded-md text-[11px] border transition-colors ${
                      extGenre.toLowerCase() === tag
                        ? "bg-rose-500/30 text-rose-200 border-rose-400"
                        : "bg-card/60 text-muted-foreground border-border/60 hover:text-rose-200 hover:border-rose-500/40"
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            )}
          </form>
        )}

        {/* Search history dropdown panel */}
        {showHistory && (
          <div
            ref={historyPanelRef}
            className="max-w-2xl mx-auto mt-1.5 animate-fade-in-up"
          >
            <div className="rounded-xl border border-border bg-card/80  shadow-lg shadow-black/20 overflow-hidden">
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
              {[
                ...SOURCE_FILTERS,
                ...Object.entries(privateMode?.sourceLabels ?? {}).map(([value, meta]) => ({
                  value: value as SourceFilter,
                  label: meta.label,
                  color: meta.color,
                })),
              ].map((f) => {
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
          <div className="space-y-8">
            {/* 3D Display Mode Switcher */}
            <div className="flex items-center justify-center">
              <div className="inline-flex items-center gap-1 p-1 rounded-full bg-card border border-primary/30 shadow-md">
                {(
                  [
                    { id: "3d-both" as const, label: "3D Stage + Grid" },
                    { id: "3d-stage" as const, label: "3D Deck Only" },
                    { id: "3d-grid" as const, label: "3D Tilt Grid" },
                  ]
                ).map((mode) => (
                  <button
                    key={mode.id}
                    type="button"
                    onClick={() => setResultsViewMode(mode.id)}
                    className={`relative px-3.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-[0.15em] transition-colors cursor-pointer ${
                      resultsViewMode === mode.id
                        ? "text-[#FFF8F5]"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {resultsViewMode === mode.id && (
                      <motion.span
                        layoutId="resultsModePill"
                        transition={{ type: "spring", stiffness: 360, damping: 28 }}
                        className="absolute inset-0 rounded-full bg-gradient-to-r from-[#E11D48] to-[#BE123C] shadow-[0_4px_14px_rgba(225,29,72,0.55)]"
                      />
                    )}
                    <span className="relative z-10">{mode.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Interactive 3D Fanned Coverflow Stage with Real Cover Images & Chapter Counts */}
            {(resultsViewMode === "3d-both" || resultsViewMode === "3d-stage") && (
              <SearchResults3DCoverflow
                items={visibleResults}
                onSelect={handleSelect}
                onExpandInfo={setExpandedManga}
                isBookmarked={isBookmarked}
                onBookmark={handleBookmark}
                resolvingId={resolvingId}
                sourceLabelMap={{
                  ...SOURCE_LABEL,
                  ...Object.fromEntries(
                    Object.entries(privateMode?.sourceLabels ?? {}).map(([k, v]) => [k, v.label])
                  ),
                }}
              />
            )}

            {/* Full Interactive 3D Pointer-Tilt Poster Grid */}
            {(resultsViewMode === "3d-both" || resultsViewMode === "3d-grid") && (
              <div className="space-y-6">
              <div ref={resultsGridRef} className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 sm:gap-5">
                {visibleResults.slice(0, displayLimit).map((m, idx) => {
                  const source = m.source ?? "mangahere";
                  const isResolving = resolvingId === m.id;
                  const bookmarked = isBookmarked?.(m.id) ?? false;
                  const isPopped = poppedBookmark === m.id;
                  const genreTags = (m.tags ?? []).slice(0, 2);

                  return (
                    <Tilt3DResultCard key={m.id}>
                      <div
                        data-result-index={idx}
                        onClick={() => !isResolving && handleSelect(m)}
                        role="button"
                        tabIndex={0}
                        onMouseEnter={() => setFocusedResultIdx(idx)}
                        onKeyDown={(e) => {
                          if ((e.key === "Enter" || e.key === " ") && !isResolving) {
                            e.preventDefault();
                            handleSelect(m);
                          }
                        }}
                        className={cn(
                          "group text-left focus:outline-none rounded-2xl overflow-hidden cursor-pointer luxury-card aspect-[3/4] relative",
                          focusedResultIdx === idx && "ring-2 ring-primary/60"
                        )}
                      >
                        {m.coverUrl ? (
                          <img
                            src={`/api/cover?url=${encodeURIComponent(m.coverUrl)}&w=360`}
                            alt={m.title}
                            referrerPolicy="no-referrer"
                            loading="lazy"
                            decoding="async"
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col items-center justify-center p-4 text-center bg-gradient-to-br from-[#2B0916] to-[#0E050A]">
                            <Sparkles className="h-5 w-5 text-[#F5D27A]/70 mb-1.5" />
                            <span className="font-serif italic text-xs text-[#FFF8F5]/80 line-clamp-2">{m.title}</span>
                          </div>
                        )}

                        {/* Scrim darkens smoothly on hover to make popped-up description crisp */}
                        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#0B0408] via-[#0B0408]/35 to-transparent group-hover:via-[#0B0408]/80 group-hover:to-[#0B0408]/30 transition-all duration-300" />

                        {/* Idle: ONLY Chapter Count badge top-right; Hover: Source pill + Bookmark */}
                        <span
                          className={`absolute top-2 left-2 text-[9px] font-mono uppercase px-2 py-0.5 rounded-full border opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20 ${SOURCE_BADGE_CLASSES[source] ?? privateMode?.sourceLabels?.[source]?.badge ?? "bg-zinc-500/20 text-zinc-200 border-zinc-500/40"}`}
                        >
                          {isResolving ? "…" : (SOURCE_LABEL[source] ?? privateMode?.sourceLabels?.[source]?.label ?? source)}
                        </span>

                        {m.lastChapter && (
                          <span className="absolute top-2 right-2 text-[9.5px] font-mono font-bold px-2 py-0.5 rounded-full bg-gradient-to-r from-[#E11D48] to-[#9F1239] text-[#FFF8F5] border border-[#F5D27A]/70 shadow-md z-20">
                            Ch.{m.lastChapter}
                          </span>
                        )}

                        <button
                          type="button"
                          onClick={(e) => handleBookmark(e, m)}
                          className={`absolute bottom-2.5 right-2.5 p-1.5 rounded-full bg-black/65 border border-white/15 text-white/80 hover:text-primary transition-all z-20 ${bookmarked ? "text-primary opacity-100" : "opacity-0 group-hover:opacity-100"} ${isPopped ? "animate-bookmark-pop" : ""}`}
                          aria-label={bookmarked ? `Remove ${m.title} from bookmarks` : `Bookmark ${m.title}`}
                        >
                          {bookmarked ? <BookmarkCheck className="h-3.5 w-3.5" /> : <Bookmark className="h-3.5 w-3.5" />}
                        </button>

                        {/* Bottom Plate: Idle = Title ONLY; Hover = Smooth Description + Genres Pop-Up */}
                        <div className="absolute bottom-0 inset-x-0 p-3 z-10 space-y-1.5">
                          <p className="font-serif text-[15px] font-semibold line-clamp-2 leading-snug text-[#FFF8F5] drop-shadow pr-6">
                            {m.title}
                          </p>
                          <div className="grid grid-rows-[0fr] opacity-0 group-hover:grid-rows-[1fr] group-hover:opacity-100 transition-all duration-300">
                            <div className="overflow-hidden space-y-1.5">
                              {m.description && (
                                <p className="text-[10.5px] text-[#FFE8EC]/85 leading-snug line-clamp-3">
                                  {m.description}
                                </p>
                              )}
                              {genreTags.length > 0 && (
                                <div className="flex flex-wrap gap-1 pr-6">
                                  {genreTags.map((tag) => (
                                    <span key={tag} className="text-[8.5px] font-mono px-1.5 py-0.5 rounded-full bg-[#240A16] text-[#F5D27A] border border-[#F5D27A]/30">
                                      {tag}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </Tilt3DResultCard>
                  );
                })}
              </div>

              {visibleResults.length > displayLimit && (
                <div className="flex justify-center pt-2">
                  <button
                    type="button"
                    onClick={() => setDisplayLimit((n) => n + 24)}
                    className="px-6 py-2.5 rounded-full bg-card border border-[#F5D27A]/45 hover:border-primary text-xs font-mono uppercase tracking-[0.18em] text-[#F5D27A] hover:text-foreground transition-colors cursor-pointer"
                  >
                    Load More ({visibleResults.length - displayLimit} Remaining)
                  </button>
                </div>
              )}
              </div>
            )}
          </div>
        )}

        {hasSearched && !loading && results.length > 0 && (
          <p className="text-center text-xs text-muted-foreground/60">
            {sourceCounts
              ? `AsuraScans: ${sourceCounts.asurascans} · ThunderScans: ${sourceCounts.thunderscans ?? 0} · VortexScans: ${sourceCounts.vortexscans ?? 0} · RizzFables: ${sourceCounts.rizzfables ?? 0} · Webtoons: ${sourceCounts.webtoons} · MangaGeko: ${sourceCounts.mgeko} · MangaDex: ${sourceCounts.mangadex} · MangaHere: ${sourceCounts.mangahere} · FanFox: ${sourceCounts.fanfox} · MAL: ${sourceCounts.mal} · AniList: ${sourceCounts.anilist}`
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
