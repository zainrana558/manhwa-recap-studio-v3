"use client";

import { useState, useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { Github, Zap, Keyboard, ChevronUp, Sun, Moon, Heart, ArrowRight, Sparkles, Palette, Check } from "lucide-react";
import { NotificationCenter } from "@/components/pipeline/notification-center";
import { CommandPalette } from "@/components/pipeline/command-palette";
import { RecentlyViewed } from "@/components/pipeline/recently-viewed";
import { useRecentlyViewed } from "@/hooks/use-recently-viewed";
import { useTheme } from "next-themes";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { SearchSection } from "@/components/pipeline/search-section";
import { MangaOverview } from "@/components/pipeline/manga-overview";
import { MangaConfig } from "@/components/pipeline/manga-config";
import { JobProgress } from "@/components/pipeline/job-progress";
import { JobHistory } from "@/components/pipeline/job-history";
import { HowItWorks } from "@/components/pipeline/how-it-works";
import { StatsBar } from "@/components/pipeline/stats-bar";
import { TrendingSearches } from "@/components/pipeline/trending-searches";
import { FAQ } from "@/components/pipeline/faq";
import { ConnectionIndicator } from "@/components/pipeline/connection-indicator";
import { SettingsDialog } from "@/components/pipeline/settings-dialog";
import { FeaturesGrid } from "@/components/pipeline/features-grid";
import { BookmarksSection } from "@/components/pipeline/bookmarks-section";
import { OnboardingTour } from "@/components/pipeline/onboarding-tour";
import { PipelineStats } from "@/components/pipeline/pipeline-stats";
import { ActivityFeed } from "@/components/pipeline/activity-feed";
import { useJobProgress } from "@/hooks/use-job-progress";
import { motion } from "framer-motion";
import { CursorVelvetGlow, AtelierAmbientParticles, EditorialMarqueeStrip, Trending3DShowcase } from "@/components/pipeline/atelier-stage";
import { useBookmarks } from "@/hooks/use-bookmarks";
import type { MangadexManga } from "@/types/pipeline";

type View = "search" | "overview" | "config" | "job";

const SHORTCUTS = [
  { key: "/", desc: "Focus search" },
  { key: "Esc", desc: "Clear search results" },
  { key: "B", desc: "Toggle bookmarks" },
  { key: "⌘K", desc: "Command palette" },
];


type BgVariant = "curtain" | "marble" | "deco" | "runway" | "ink" | "lattice";

const BG_VARIANTS: Array<{
  id: BgVariant;
  name: string;
  vibe: string;
  swatch: string;
  className: string;
}> = [
  {
    id: "curtain",
    name: "Crimson Silk Curtain",
    vibe: "Opera house velvet folds",
    swatch: "linear-gradient(90deg, #1E0610, #6B0F2E, #16050C, #520C23)",
    className: "bg-variant-curtain",
  },
  {
    id: "marble",
    name: "Imperial Marble & Gold",
    vibe: "Nero obsidian & gold veins",
    swatch: "linear-gradient(135deg, #1A0812 0%, #0B0509 45%, #F5D27A 50%, #150710 56%)",
    className: "bg-variant-marble",
  },
  {
    id: "deco",
    name: "Sunburst Art-Deco Fan",
    vibe: "1920s Gatsby cinema palais",
    swatch: "radial-gradient(circle at 50% 100%, #F5D27A 0%, #881337 45%, #12060B 100%)",
    className: "bg-variant-deco",
  },
  {
    id: "runway",
    name: "Cherry Spotlight Grid",
    vibe: "High-fashion lookbook",
    swatch: "radial-gradient(circle at 50% 40%, #E11D48 0%, #14060D 75%)",
    className: "bg-variant-runway",
  },
  {
    id: "ink",
    name: "Nocturne Ink & Gold Dust",
    vibe: "Sumi wash & maki-e flakes",
    swatch: "linear-gradient(135deg, #2D091B 0%, #0D050A 55%, #C49334 100%)",
    className: "bg-variant-ink",
  },
  {
    id: "lattice",
    name: "Velvet Diamond Lattice",
    vibe: "Original salon diamond weave",
    swatch: "linear-gradient(135deg, #240915 0%, #0E060B 60%, #9F1239 100%)",
    className: "studio-canvas",
  },
];

const TECH_BADGES = ["Next.js 16", "Tailwind CSS 4", "Prisma", "Framer Motion", "Socket.IO", "TypeScript"];

export default function Home() {
  const [view, setView] = useState<View>("search");
  const [selectedManga, setSelectedManga] = useState<MangadexManga | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [trendingQuery, setTrendingQuery] = useState<string | null>(null);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [bgPickerOpen, setBgPickerOpen] = useState(false);
  const [bgVariant, setBgVariant] = useState<BgVariant>("curtain");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("manhwa-bg-variant") as BgVariant | null;
      if (saved && BG_VARIANTS.some((v) => v.id === saved)) {
        setBgVariant(saved);
      }
    } catch {}
  }, []);

  const handleSelectBg = useCallback((id: BgVariant) => {
    setBgVariant(id);
    try {
      localStorage.setItem("manhwa-bg-variant", id);
    } catch {}
  }, []);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [activeNav, setActiveNav] = useState<"explore" | "collection" | "archive">("explore");
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const { recentlyViewed, addViewed, removeItem: removeRecentlyViewed, clearAll: clearRecentlyViewed } = useRecentlyViewed();
  const { theme, setTheme } = useTheme();
  const { isBookmarked, toggleBookmark } = useBookmarks();

  // Hydration-safe client-only check
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );

  const mainRef = useRef<HTMLDivElement>(null);
  const searchSectionRef = useRef<{ clearResults: () => void } | null>(null);

  const { job, logs, connected } = useJobProgress(currentJobId);

  const handleSelectManga = useCallback((manga: MangadexManga) => {
    if (manga.contentRating !== "erotica") {
      addViewed({ id: manga.id, title: manga.title, coverUrl: manga.coverUrl || "" });
    }
    setSelectedManga(manga);
    setView("overview");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [addViewed]);

  const handleJobCreated = useCallback((jobId: string) => {
    setCurrentJobId(jobId);
    setView("job");
    setHistoryRefresh((n) => n + 1);
    try {
      localStorage.setItem("activeJobId", jobId);
    } catch {
      // ignore — private browsing / storage disabled
    }
  }, []);

  const handleNewJob = useCallback(() => {
    setCurrentJobId(null);
    setSelectedManga(null);
    setView("search");
    setHistoryRefresh((n) => n + 1);
    try {
      localStorage.removeItem("activeJobId");
    } catch {
      // ignore
    }
  }, []);

  const handleSelectHistoryJob = useCallback((jobId: string) => {
    setCurrentJobId(jobId);
    setView("job");
    try {
      localStorage.setItem("activeJobId", jobId);
    } catch {
      // ignore
    }
  }, []);

  // Jobs keep running server-side whether or not the browser is open —
  // closing the laptop/tab and coming back should land you back on the job
  // you were watching (or its finished/errored result), not a blank search
  // page with the job buried in history. Restore it once on mount.
  useEffect(() => {
    let cancelled = false;
    let savedId: string | null = null;
    try {
      savedId = localStorage.getItem("activeJobId");
    } catch {
      savedId = null;
    }

    const ACTIVE = new Set([
      "pending",
      "awaiting_review",
      "scraping",
      "transcribing",
      "translating",
      "rendering",
      "merging",
    ]);

    fetch("/api/jobs")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled) return;
        const jobs = (data?.jobs ?? []) as Array<{ id: string; status: string; createdAt: string }>;
        if (!jobs.length) {
          if (savedId) {
            try {
              localStorage.removeItem("activeJobId");
            } catch {
              // ignore
            }
          }
          return;
        }

        const activeJob = jobs.find((j) => ACTIVE.has(j.status));
        if (activeJob) {
          try {
            localStorage.setItem("activeJobId", activeJob.id);
          } catch {
            // ignore
          }
          setCurrentJobId(activeJob.id);
          setView("job");
          return;
        }

        if (!savedId) return;

        const savedJob = jobs.find((j) => j.id === savedId);
        const latest = jobs[0];
        if (!savedJob) {
          if (latest) {
            try {
              localStorage.setItem("activeJobId", latest.id);
            } catch {
              // ignore
            }
            setCurrentJobId(latest.id);
            setView("job");
          } else {
            try {
              localStorage.removeItem("activeJobId");
            } catch {
              // ignore
            }
          }
          return;
        }

        // If the saved job is already finished and a newer job exists on the server,
        // promote the latest job so refreshes reflect recent backend runs.
        const target =
          !ACTIVE.has(savedJob.status) &&
          latest &&
          latest.id !== savedJob.id &&
          new Date(latest.createdAt).getTime() > new Date(savedJob.createdAt).getTime()
            ? latest
            : savedJob;

        try {
          localStorage.setItem("activeJobId", target.id);
        } catch {
          // ignore
        }
        setCurrentJobId(target.id);
        setView("job");
      })
      .catch(() => {
        // Pipeline/web service unreachable at load time — leave the search
        // view up rather than getting stuck; the user can navigate to
        // Job History once the service is back.
      });

    return () => {
      cancelled = true;
    };
    // Intentionally run once on mount only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleTrendingPick = useCallback((query: string) => {
    setTrendingQuery(query);
  }, []);

  const handleClearResults = useCallback(() => {
    setTrendingQuery(null);
  }, []);

  const handleBookmarkToggle = useCallback((manga: MangadexManga) => {
    toggleBookmark(manga);
  }, [toggleBookmark]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      const isTyping = tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement).isContentEditable;

      if (e.key === "/" && view === "search" && !isTyping) {
        const input = document.getElementById("search-input") as HTMLInputElement | null;
        if (input) {
          e.preventDefault();
          input.focus();
        }
      }

      if (e.key === "Escape" && view === "search" && !isTyping) {
        searchSectionRef.current?.clearResults();
      }

      if (e.key === "b" && view === "search" && !isTyping) {
        e.preventDefault();
        setShowBookmarks((v) => !v);
      }

      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setCommandPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [view]);

  // Scroll detection for back-to-top
  useEffect(() => {
    const container = mainRef.current;
    if (!container) return;
    const handleScroll = () => {
      setShowScrollTop(container.scrollTop > 400);
    };
    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = useCallback(() => {
    mainRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  const handleCommandAction = useCallback(
    (actionId: string) => {
      switch (actionId) {
        case "search": {
          if (view !== "search") handleNewJob();
          const input = document.getElementById("search-input") as HTMLInputElement | null;
          if (input) {
            input.focus();
            input.scrollIntoView({ behavior: "smooth", block: "center" });
          }
          break;
        }
        case "bookmarks":
          if (view !== "search") handleNewJob();
          setShowBookmarks(true);
          break;
        case "settings":
          window.dispatchEvent(new CustomEvent("open-settings-dialog"));
          break;
        case "theme":
          toggleTheme();
          break;
        case "history": {
          if (view !== "search") handleNewJob();
          const el = document.getElementById("section-job-history");
          if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          break;
        }
        case "trending": {
          if (view !== "search") handleNewJob();
          const el = document.getElementById("section-trending");
          if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          break;
        }
        case "how-it-works": {
          if (view !== "search") handleNewJob();
          const el = document.getElementById("section-how-it-works");
          if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          break;
        }
        case "new-job":
          handleNewJob();
          break;
      }
    },
    [view, handleNewJob, toggleTheme]
  );

  return (
    <div className={`min-h-screen flex flex-col ${BG_VARIANTS.find((v) => v.id === bgVariant)?.className ?? "studio-canvas"} relative overflow-x-clip transition-[background] duration-500`}>
      {/* Living interactive velvet spotlight + slow drifting cherry/champagne silk orbs */}
      <CursorVelvetGlow />
      <AtelierAmbientParticles />
      <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        {/* Architectural print-editorial side margin rails */}
        <div className="hidden xl:block fixed top-0 bottom-0 left-8 w-px bg-gradient-to-b from-transparent via-[#F5D27A]/20 to-transparent" />
        <div className="hidden xl:block fixed top-0 bottom-0 right-8 w-px bg-gradient-to-b from-transparent via-[#F5D27A]/20 to-transparent" />
      </div>
      {/* Onboarding tour */}
      <OnboardingTour />

      {/* Header — Center-Based Couture Maison Bar (Zero Progress Loader, Zero Glass) */}
      <header className="sticky top-0 z-40 bg-[#FFFDF9] dark:bg-[#12070C] border-b border-primary/25 shadow-[0_14px_40px_-18px_rgba(0,0,0,0.9)]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3.5 grid grid-cols-[1fr_auto_1fr] items-center gap-4">
          {/* Left: Kinetic Sliding-Indicator Navigation */}
          <nav className="hidden md:flex items-center gap-1 p-1 rounded-full bg-secondary/70 border border-border justify-self-start">
            {(
              [
                {
                  id: "explore" as const,
                  label: "Explore",
                  action: () => {
                    setActiveNav("explore");
                    handleNewJob();
                  },
                },
                {
                  id: "collection" as const,
                  label: "Collection",
                  action: () => {
                    setActiveNav("collection");
                    if (view !== "search") handleNewJob();
                    setShowBookmarks((v) => !v);
                  },
                },
                {
                  id: "archive" as const,
                  label: "Archive",
                  action: () => {
                    setActiveNav("archive");
                    if (view !== "search") handleNewJob();
                    setTimeout(() => {
                      document.getElementById("section-job-history")?.scrollIntoView({ behavior: "smooth" });
                    }, 50);
                  },
                },
              ]
            ).map((item) => {
              const isActive = activeNav === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={item.action}
                  className={`relative px-3.5 py-1.5 rounded-full text-[10.5px] font-mono uppercase tracking-[0.16em] transition-colors cursor-pointer ${
                    isActive ? "text-[#FFF8F5]" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="maisonNavPill"
                      transition={{ type: "spring", stiffness: 360, damping: 30 }}
                      className="absolute inset-0 rounded-full bg-gradient-to-r from-[#E11D48] to-[#BE123C] shadow-[0_6px_18px_-4px_rgba(225,29,72,0.7)]"
                    />
                  )}
                  <span className="relative z-10">{item.label}</span>
                </button>
              );
            })}
          </nav>
          <div className="flex md:hidden items-center justify-self-start">
            {view !== "search" && <ConnectionIndicator connected={connected} />}
          </div>

          {/* Center: Bespoke Crest + Wordmark (Centered Anchor) */}
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              setActiveNav("explore");
              handleNewJob();
            }}
            className="flex items-center gap-3 group justify-self-center cursor-pointer"
          >
            <div className="relative h-10 w-10 flex items-center justify-center shrink-0">
              <svg viewBox="0 0 44 44" className="h-10 w-10 drop-shadow-[0_6px_18px_rgba(225,29,72,0.55)] transition-transform duration-500 group-hover:rotate-[12deg]" aria-hidden="true">
                <defs>
                  <linearGradient id="cherryCrest" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#FB7185" />
                    <stop offset="45%" stopColor="#E11D48" />
                    <stop offset="100%" stopColor="#881337" />
                  </linearGradient>
                  <linearGradient id="goldTrim" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#FFF0B8" />
                    <stop offset="50%" stopColor="#F5D27A" />
                    <stop offset="100%" stopColor="#C49334" />
                  </linearGradient>
                </defs>
                <polygon points="13,3 31,3 41,13 41,31 31,41 13,41 3,31 3,13" fill="url(#cherryCrest)" stroke="url(#goldTrim)" strokeWidth="1.75" />
                <polygon points="14.5,6 29.5,6 38,14.5 38,29.5 29.5,38 14.5,38 6,29.5 6,14.5" fill="none" stroke="#FFF0B8" strokeOpacity="0.35" strokeWidth="0.75" />
                <path d="M12.5 14.5 L20.5 12.5 L20.5 31.5 L12.5 29.5 Z" fill="#FFF8F5" fillOpacity="0.96" />
                <path d="M23.5 12.5 L31.5 14.5 L31.5 23.5 L23.5 25.5 Z" fill="url(#goldTrim)" />
                <polygon points="24,27.2 31,29.3 24,31.4" fill="#FFF8F5" />
              </svg>
            </div>
            <div className="text-left">
              <span className="block font-serif font-semibold text-lg sm:text-2xl tracking-tight leading-none text-foreground">
                Manhwa <span className="italic gold-foil-live">Recap</span>
              </span>
              <span className="hidden sm:block text-[8.5px] font-mono uppercase tracking-[0.3em] text-muted-foreground mt-0.5">
                Atelier · Edition III
              </span>
            </div>
          </motion.button>

          {/* Right: Quick Command Pill + Sculpted Utility Controls */}
          <div className="flex items-center justify-end gap-1.5 sm:gap-2 justify-self-end">
            <div className="hidden md:flex items-center">
              {view !== "search" && <ConnectionIndicator connected={connected} />}
            </div>
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              type="button"
              onClick={() => setCommandPaletteOpen(true)}
              className="hidden sm:inline-flex items-center gap-1.5 h-8 px-3 rounded-full border border-[#F5D27A]/35 bg-card text-[10px] font-mono uppercase tracking-[0.14em] text-foreground/85 hover:border-primary hover:text-primary transition-colors cursor-pointer"
            >
              <span>Search</span>
              <kbd className="text-[9px] text-[#F5D27A] font-semibold">⌘K</kbd>
            </motion.button>

            {view === "search" && <NotificationCenter />}
            <SettingsDialog />

            {/* Live Background Switcher */}
            <Popover open={bgPickerOpen} onOpenChange={setBgPickerOpen}>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className="h-8 px-2.5 rounded-full border border-[#F5D27A]/40 bg-card flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-[0.14em] text-[#F5D27A] hover:border-primary hover:text-primary transition-colors cursor-pointer"
                  aria-label="Switch background theme"
                  title="Preview Backgrounds"
                >
                  <Palette className="h-3.5 w-3.5" />
                  <span className="hidden lg:inline">Backdrop</span>
                </button>
              </PopoverTrigger>
              <PopoverContent side="bottom" align="end" className="w-72 p-3 space-y-2.5 bg-[#160810] dark:bg-[#14070E] border border-[#F5D27A]/35 shadow-2xl">
                <div className="flex items-center justify-between border-b border-white/10 pb-2">
                  <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#F5D27A]">
                    Live Backdrop Preview
                  </p>
                  <span className="text-[9px] font-mono text-muted-foreground">6 Styles</span>
                </div>
                <div className="space-y-1.5">
                  {BG_VARIANTS.map((v) => {
                    const active = bgVariant === v.id;
                    return (
                      <button
                        key={v.id}
                        type="button"
                        onClick={() => handleSelectBg(v.id)}
                        className={`w-full flex items-center gap-3 p-2 rounded-xl border text-left transition-all cursor-pointer ${
                          active
                            ? "bg-[#260B18] border-[#F5D27A]/75 shadow-[0_6px_18px_-6px_rgba(225,29,72,0.55)]"
                            : "bg-card/50 border-white/5 hover:border-primary/40 hover:bg-[#1E0A14]"
                        }`}
                      >
                        <span
                          className="h-8 w-8 rounded-lg border border-[#F5D27A]/40 shrink-0 shadow-inner"
                          style={{ background: v.swatch }}
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-[#FFF8F5] truncate">{v.name}</p>
                          <p className="text-[10px] text-muted-foreground truncate">{v.vibe}</p>
                        </div>
                        {active && <Check className="h-3.5 w-3.5 text-[#F5D27A] shrink-0" />}
                      </button>
                    );
                  })}
                </div>
              </PopoverContent>
            </Popover>

            <Popover open={shortcutsOpen} onOpenChange={setShortcutsOpen}>
              <PopoverTrigger asChild>
                <button
                  className="h-8 w-8 rounded-full border border-border bg-card flex items-center justify-center text-muted-foreground hover:text-primary hover:border-primary/60 transition-colors cursor-pointer"
                  aria-label="Keyboard shortcuts"
                >
                  <Keyboard className="h-3.5 w-3.5" />
                </button>
              </PopoverTrigger>
              <PopoverContent side="bottom" align="end" className="w-64 p-3 space-y-3">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Shortcuts
                </p>
                <div className="space-y-2">
                  {SHORTCUTS.map((s) => (
                    <div key={s.key} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground text-xs">{s.desc}</span>
                      <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted text-foreground/80 font-mono text-[10px]">
                        {s.key}
                      </kbd>
                    </div>
                  ))}
                </div>
              </PopoverContent>
            </Popover>

            {mounted && (
              <button
                onClick={toggleTheme}
                className="h-8 w-8 rounded-full border border-border bg-card flex items-center justify-center text-muted-foreground hover:text-primary hover:border-primary/60 transition-colors cursor-pointer"
                aria-label="Toggle theme"
              >
                <Sun
                  className={`h-3.5 w-3.5 transition-all duration-300 ${theme === "dark" ? "rotate-0 scale-100" : "rotate-90 scale-0 absolute"}`}
                />
                <Moon
                  className={`h-3.5 w-3.5 transition-all duration-300 ${theme === "dark" ? "-rotate-90 scale-0 absolute" : "rotate-0 scale-100"}`}
                />
              </button>
            )}

            <a
              href="https://github.com/zainrana558/manhwa-recap-studio-v3"
              target="_blank"
              rel="noopener noreferrer"
              className="h-8 w-8 rounded-full border border-border bg-card flex items-center justify-center text-muted-foreground hover:text-primary hover:border-primary/60 transition-colors"
              aria-label="Source"
            >
              <Github className="h-3.5 w-3.5" />
            </a>
          </div>
        </div>
      </header>

      {/* Main */}
      <main ref={mainRef} className="relative z-10 flex-1 overflow-x-hidden px-4 py-8 sm:py-12">
        {view === "search" && (
          <div className="space-y-14 max-w-6xl mx-auto">
            <SearchSection
              ref={searchSectionRef}
              onResults={() => {}}
              onSelectManga={handleSelectManga}
              externalQuery={trendingQuery}
              onClearResults={handleClearResults}
              isBookmarked={isBookmarked}
              onBookmarkToggle={handleBookmarkToggle}
            />

            <div className="rounded-2xl overflow-hidden border border-[#F5D27A]/20">
              <EditorialMarqueeStrip />
            </div>

            <div id="section-trending">
              <Trending3DShowcase
                onSelectManga={handleSelectManga}
                isBookmarked={isBookmarked}
                onBookmarkToggle={handleBookmarkToggle}
              />
            </div>

            <RecentlyViewed
              items={recentlyViewed}
              onSelectManga={handleSelectManga}
              onRemoveItem={removeRecentlyViewed}
              onClearAll={clearRecentlyViewed}
            />

            {showBookmarks && (
              <>
                <div className="editorial-rule max-w-4xl mx-auto" />
                <BookmarksSection onSelectManga={handleSelectManga} />
              </>
            )}

            <div className="editorial-rule max-w-4xl mx-auto" />

            <div id="section-job-history">
              <JobHistory onSelectJob={handleSelectHistoryJob} refreshKey={historyRefresh} />
            </div>
          </div>
        )}

        {view === "overview" && selectedManga && (
          <MangaOverview
            manga={selectedManga}
            onBack={handleNewJob}
            onStartRecap={(resolved) => {
              setSelectedManga(resolved);
              setView("config");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            isBookmarked={isBookmarked(selectedManga.id)}
            onBookmarkToggle={() => handleBookmarkToggle(selectedManga)}
          />
        )}

        {view === "config" && selectedManga && (
          <MangaConfig
            manga={selectedManga}
            onBack={() => setView("overview")}
            onJobCreated={handleJobCreated}
          />
        )}

        {view === "job" && (
          <JobProgress
            job={job}
            logs={logs}
            connected={connected}
            onCancel={handleNewJob}
            onNewJob={handleNewJob}
          />
        )}
      </main>

      {/* Back to top floating button */}
      <div
        className={`fixed bottom-20 right-6 z-30 transition-all duration-300 ${showScrollTop && view === "search" ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4 pointer-events-none"}`}
      >
        <button
          onClick={scrollToTop}
          className="p-2.5 rounded-full bg-primary/90 text-primary-foreground shadow-lg hover:bg-primary hover:shadow-xl hover:shadow-primary/20 transition-all active:scale-95 group"
          aria-label="Back to top"
        >
          <ChevronUp className="h-5 w-5" />
        </button>
      </div>

      {/* Command palette */}
      <CommandPalette
        open={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        onAction={handleCommandAction}
        isDark={theme === "dark"}
      />

      {/* Footer — Monumental Couture Maison Colophon */}
      <footer className="relative z-10 mt-auto bg-[#FFF9F6] dark:bg-[#11060B] border-t border-primary/35 overflow-hidden">
        {/* Giant subtle architectural serif watermark */}
        <div
          aria-hidden="true"
          className="pointer-events-none select-none absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap font-serif font-bold text-[15vw] sm:text-[11vw] leading-none tracking-tight text-primary/[0.04] dark:text-[#F5D27A]/[0.035]"
        >
          MANHWA RECAP
        </div>

        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 py-14 grid grid-cols-1 md:grid-cols-12 gap-10 items-start">
          {/* Col 1 — Brand House */}
          <div className="md:col-span-5 space-y-4">
            <div className="flex items-center gap-3">
              <svg viewBox="0 0 44 44" className="h-9 w-9 shrink-0 drop-shadow-[0_4px_12px_rgba(225,29,72,0.45)]" aria-hidden="true">
                <polygon points="13,3 31,3 41,13 41,31 31,41 13,41 3,31 3,13" fill="#E11D48" stroke="#F5D27A" strokeWidth="1.8" />
                <path d="M12.5 14.5 L20.5 12.5 L20.5 31.5 L12.5 29.5 Z" fill="#FFF8F5" />
                <path d="M23.5 12.5 L31.5 14.5 L31.5 23.5 L23.5 25.5 Z" fill="#F5D27A" />
                <polygon points="24,27.2 31,29.3 24,31.4" fill="#FFF8F5" />
              </svg>
              <div>
                <p className="font-serif font-semibold text-xl leading-none text-foreground">
                  Manhwa <span className="italic gold-foil-live">Recap</span>
                </p>
                <p className="text-[9px] font-mono uppercase tracking-[0.28em] text-[#F5D27A] mt-1">
                  Private Screening Atelier
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed max-w-sm">
              Sculpting vertical webtoons into cinema-grade narrated editions with neural segmentation, zero-clip vertical pans, and studio acoustics.
            </p>
          </div>

          {/* Col 2 — Navigation */}
          <div className="md:col-span-3 space-y-3">
            <p className="text-[10px] font-mono uppercase tracking-[0.24em] text-primary">
              Navigation
            </p>
            <div className="flex flex-col items-start gap-2.5 text-xs text-muted-foreground">
              <button type="button" onClick={handleNewJob} className="hover:text-foreground hover:translate-x-1 transition-all cursor-pointer">
                Explore Archives
              </button>
              <button
                type="button"
                onClick={() => {
                  if (view !== "search") handleNewJob();
                  setShowBookmarks((v) => !v);
                }}
                className="hover:text-foreground hover:translate-x-1 transition-all cursor-pointer"
              >
                Saved Collection
              </button>
              <button
                type="button"
                onClick={() => {
                  if (view !== "search") handleNewJob();
                  setTimeout(() => document.getElementById("section-job-history")?.scrollIntoView({ behavior: "smooth" }), 50);
                }}
                className="hover:text-foreground hover:translate-x-1 transition-all cursor-pointer"
              >
                Studio History
              </button>
              <button type="button" onClick={() => setCommandPaletteOpen(true)} className="hover:text-foreground hover:translate-x-1 transition-all cursor-pointer">
                Command Palette (⌘K)
              </button>
            </div>
          </div>

          {/* Col 3 — Craft & Engine */}
          <div className="md:col-span-4 space-y-3">
            <p className="text-[10px] font-mono uppercase tracking-[0.24em] text-primary">
              Atelier Engine
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              YOLO11n-seg Slicer C · RapidOCR · Kokoro Neural Voice · 1920×1080 Cinema Mastering
            </p>
            <div className="pt-1 flex items-center gap-3">
              <a
                href="https://github.com/zainrana558/manhwa-recap-studio-v3"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/40 bg-card text-[11px] font-mono uppercase tracking-[0.14em] text-foreground hover:border-primary hover:text-primary transition-colors"
              >
                <Github className="h-3.5 w-3.5" />
                Repository
              </a>
            </div>
          </div>
        </div>

        <div className="relative border-t border-border/50">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] font-mono text-muted-foreground/70">
            <span>© Manhwa Recap Studio · Edition III</span>
            <span>For personal archival &amp; narration use</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
