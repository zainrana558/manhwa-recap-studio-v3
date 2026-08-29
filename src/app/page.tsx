"use client";

import { useState, useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { Github, Zap, Keyboard, ChevronUp, Sun, Moon, Heart, ArrowRight, Sparkles } from "lucide-react";
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
import { useScrollProgress } from "@/hooks/use-section-observer";
import { useBookmarks } from "@/hooks/use-bookmarks";
import type { MangadexManga } from "@/types/pipeline";

type View = "search" | "config" | "job";

const SHORTCUTS = [
  { key: "/", desc: "Focus search" },
  { key: "Esc", desc: "Clear search results" },
  { key: "B", desc: "Toggle bookmarks" },
  { key: "⌘K", desc: "Command palette" },
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
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const { recentlyViewed, addViewed, removeItem: removeRecentlyViewed, clearAll: clearRecentlyViewed } = useRecentlyViewed();
  const { theme, setTheme } = useTheme();
  const scrollProgress = useScrollProgress();
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
    addViewed({ id: manga.id, title: manga.title, coverUrl: manga.coverUrl || "" });
    setSelectedManga(manga);
    setView("config");
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
    if (!savedId) return;

    fetch(`/api/jobs/${savedId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data?.job) {
          setCurrentJobId(savedId);
          setView("job");
        } else {
          // Job no longer exists (deleted/DB reset) — clear the stale pointer.
          try {
            localStorage.removeItem("activeJobId");
          } catch {
            // ignore
          }
        }
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
    <div className="min-h-screen flex flex-col bg-background bg-grain">
      {/* Onboarding tour */}
      <OnboardingTour />

      {/* Header */}
      <header className="sticky top-0 z-40 glass backdrop-blur-strong relative shadow-[0_4px_30px_-10px_oklch(0.78_0.17_65/0.08)]">
        {/* Scroll progress bar */}
        <div className="absolute inset-x-0 top-0 h-[2px]">
          <div
            className="h-full transition-all duration-150 ease-out"
            style={{
              width: `${scrollProgress * 100}%`,
              background: "linear-gradient(90deg, oklch(0.78 0.17 65), oklch(0.72 0.18 45), oklch(0.85 0.15 75))",
              backgroundSize: "200% 100%",
              animation: "progress-shimmer 3s linear infinite",
            }}
          />
        </div>

        <div className="max-w-6xl mx-auto flex items-center justify-between px-4 py-3">
          <button onClick={handleNewJob} className="flex items-center gap-2.5 group hover-glow-sm rounded-lg -m-1 p-1">
            <div className="p-1.5 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-all duration-300 group-hover:shadow-md group-hover:shadow-primary/10">
              <Zap className="h-5 w-5 text-primary" />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm sm:text-base">Manhwa Recap Studio</span>
              <span className="hidden sm:inline-block text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary/80 border border-primary/20">
                v3
              </span>
            </div>
          </button>
          <div className="flex items-center gap-1.5 sm:gap-2">
            {/* Connection indicator — only shown when not in search view */}
            {view !== "search" && <ConnectionIndicator connected={connected} />}

            {/* Notification center — only shown on search view */}
            {view === "search" && <NotificationCenter />}

            {/* Settings dialog button */}
            <SettingsDialog />

            {/* Keyboard shortcut helper */}
            <Popover open={shortcutsOpen} onOpenChange={setShortcutsOpen}>
              <PopoverTrigger asChild>
                <button
                  className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
                  aria-label="Keyboard shortcuts"
                >
                  <Keyboard className="h-4 w-4" />
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
                <div className="pt-1 border-t border-border">
                  <p className="text-[10px] text-muted-foreground/60 leading-relaxed">
                    Bookmarks are saved locally. Press <kbd className="px-1 py-0.5 rounded border border-border bg-muted font-mono text-[9px]">B</kbd> to toggle. Press <kbd className="px-1 py-0.5 rounded border border-border bg-muted font-mono text-[9px]">⌘K</kbd> for commands.
                  </p>
                </div>
              </PopoverContent>
            </Popover>

            {/* Theme toggle */}
            {mounted && (
              <button
                onClick={toggleTheme}
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-all duration-300"
                aria-label="Toggle theme"
              >
                <Sun
                  className={`h-4 w-4 transition-all duration-300 ${theme === "dark" ? "rotate-0 scale-100" : "rotate-90 scale-0 absolute"}`}
                />
                <Moon
                  className={`h-4 w-4 transition-all duration-300 ${theme === "dark" ? "-rotate-90 scale-0 absolute" : "rotate-0 scale-100"}`}
                />
              </button>
            )}

            <a
              href="https://github.com/zainrana558/manhwa-recap-studio-v3"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition"
              aria-label="Source"
            >
              <Github className="h-4 w-4" />
            </a>
          </div>
        </div>

        <div
          className="absolute bottom-0 inset-x-0 h-px"
          style={{
            background: "linear-gradient(90deg, transparent, oklch(0.78 0.17 65 / 0.4), oklch(0.72 0.18 45 / 0.3), transparent)",
          }}
        />
      </header>

      {/* Main */}
      <main ref={mainRef} className="flex-1 overflow-y-auto px-4 py-8 sm:py-12">
        {view === "search" && (
          <div className="space-y-12 max-w-6xl mx-auto">
            <StatsBar />
            <SearchSection
              ref={searchSectionRef}
              onResults={() => {}}
              onSelectManga={handleSelectManga}
              externalQuery={trendingQuery}
              onClearResults={handleClearResults}
              isBookmarked={isBookmarked}
              onBookmarkToggle={handleBookmarkToggle}
            />

            <RecentlyViewed
              items={recentlyViewed}
              onSelectManga={handleSelectManga}
              onRemoveItem={removeRecentlyViewed}
              onClearAll={clearRecentlyViewed}
            />

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            {/* Bookmarks toggle + section */}
            {showBookmarks && (
              <BookmarksSection onSelectManga={handleSelectManga} />
            )}

            <div id="section-trending">
              <TrendingSearches onPick={handleTrendingPick} />
            </div>

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            <div id="section-how-it-works">
              <HowItWorks />
            </div>

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            <FeaturesGrid />

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            <PipelineStats />

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            <div id="section-job-history">
              <JobHistory onSelectJob={handleSelectHistoryJob} refreshKey={historyRefresh} />
            </div>

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            <FAQ />

            <div className="gradient-separator max-w-4xl mx-auto my-0" />

            <ActivityFeed />
          </div>
        )}

        {view === "config" && selectedManga && (
          <MangaConfig
            manga={selectedManga}
            onBack={handleNewJob}
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

      {/* CTA Section */}
      {view === "search" && (
        <div className="relative border-t border-border overflow-hidden">
          {/* Animated gradient mesh background */}
          <div
            className="absolute inset-0 opacity-[0.07]"
            style={{
              background: "radial-gradient(ellipse 600px 400px at 20% 50%, oklch(0.78 0.17 65 / 0.6), transparent), radial-gradient(ellipse 500px 350px at 80% 40%, oklch(0.72 0.18 45 / 0.5), transparent), radial-gradient(ellipse 400px 300px at 50% 80%, oklch(0.85 0.15 75 / 0.4), transparent)",
              backgroundSize: "200% 200%",
              animation: "gradient-mesh 12s ease-in-out infinite",
            }}
          />

          {/* Floating decorative shapes */}
          <div className="absolute top-8 left-[10%] w-3 h-3 rounded-full bg-primary/20 animate-float" style={{ animationDelay: "0s" }} />
          <div className="absolute top-12 right-[15%] w-2 h-2 rotate-45 bg-primary/15 animate-float" style={{ animationDelay: "0.8s" }} />
          <div className="absolute bottom-10 left-[25%] w-2.5 h-2.5 rounded-full bg-primary/10 animate-float" style={{ animationDelay: "1.5s" }} />
          <div className="absolute top-1/2 right-[8%] w-2 h-2 rotate-45 bg-primary/12 animate-float" style={{ animationDelay: "2.2s" }} />
          <div className="absolute bottom-8 right-[30%] w-1.5 h-1.5 rounded-full bg-primary/18 animate-float" style={{ animationDelay: "0.5s" }} />

          <div className="relative max-w-6xl mx-auto px-4 py-16 sm:py-20">
            <div className="flex flex-col lg:flex-row items-center justify-between gap-10">
              <div className="space-y-4 text-center lg:text-left">
                <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight leading-tight">
                  Ready to create your{" "}
                  <br className="hidden sm:block" />
                  <span className="text-gradient">first recap video?</span>
                </h2>
                <p className="text-sm text-muted-foreground max-w-md leading-relaxed mx-auto lg:mx-0">
                  Search for any manhwa, configure your preferences, and let the AI pipeline handle the rest. It takes about 6 minutes per chapter.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row items-center gap-3">
                {/* Main CTA button with glow + pulsing ring */}
                <div className="relative">
                  {/* Pulsing ring behind button */}
                  <div className="absolute inset-0 rounded-xl bg-primary/20 animate-pulse-ring" />
                  {/* Subtle glow behind button */}
                  <div className="absolute -inset-3 rounded-2xl bg-primary/10 blur-xl" />
                  <button
                    onClick={() => {
                      const input = document.getElementById("search-input") as HTMLInputElement | null;
                      if (input) {
                        input.focus();
                        input.scrollIntoView({ behavior: "smooth", block: "center" });
                      }
                    }}
                    className="relative inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 shadow-lg hover:shadow-xl hover:shadow-primary/20 transition-all active:scale-95 shine-effect"
                  >
                    <Sparkles className="h-4 w-4" />
                    Get Started
                    <ArrowRight className="h-4 w-4" />
                  </button>
                </div>
                <button
                  onClick={() => {
                    const input = document.getElementById("search-input") as HTMLInputElement | null;
                    if (input) {
                      input.value = "Solo Leveling";
                      input.dispatchEvent(new Event("input", { bubbles: true }));
                      input.focus();
                      const form = input.closest("form");
                      if (form) form.requestSubmit();
                      input.scrollIntoView({ behavior: "smooth", block: "center" });
                    }
                  }}
                  className="inline-flex items-center gap-2 px-5 py-3 rounded-xl border border-primary/30 text-primary hover:bg-primary/5 hover:border-primary/50 transition-all hover-lift"
                >
                  Try Demo
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="relative mt-auto border-t border-border bg-background/50 overflow-hidden">
        {/* Subtle top gradient glow line */}
        <div
          className="absolute top-0 inset-x-0 h-px"
          style={{
            background: "linear-gradient(90deg, transparent, oklch(0.78 0.17 65 / 0.4), oklch(0.72 0.18 45 / 0.3), transparent)",
          }}
        />

        <div className="relative max-w-6xl mx-auto px-4 py-8 space-y-5">
          {/* Brand section */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-primary/10 border border-primary/15">
                <Zap className="h-4 w-4 text-primary" />
              </div>
              <span className="font-bold text-sm">Manhwa Recap Studio</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary/80 border border-primary/20">
                v3
              </span>
            </div>

            {/* Social links */}
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/zainrana558/manhwa-recap-studio-v3"
                target="_blank"
                rel="noopener noreferrer"
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
                aria-label="GitHub"
              >
                <Github className="h-4 w-4" />
              </a>
              <span className="text-muted-foreground/30">|</span>
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                Built with <Heart className="h-3 w-3 text-rose-400 inline" /> for manhwa fans
              </span>
            </div>
          </div>

          <Separator className="opacity-30" />

          {/* Technology badges with marquee scroll */}
          <div className="relative overflow-hidden py-1">
            {/* Fade edges */}
            <div className="absolute inset-y-0 left-0 w-12 bg-gradient-to-r from-background/80 to-transparent z-10 pointer-events-none" />
            <div className="absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-background/80 to-transparent z-10 pointer-events-none" />
            <div className="flex animate-marquee w-max gap-2">
              {[...TECH_BADGES, ...TECH_BADGES].map((badge, i) => (
                <span
                  key={`${badge}-${i}`}
                  className="bg-card border border-border rounded-full px-2.5 py-0.5 text-[10px] text-muted-foreground font-medium hover:border-primary/30 hover:text-foreground transition-colors cursor-default whitespace-nowrap"
                >
                  {badge}
                </span>
              ))}
            </div>
          </div>

          <Separator className="opacity-20" />

          {/* Bottom links */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground/60">Powered by</span>
              <a href="https://asurascans.com" target="_blank" rel="noopener noreferrer" className="text-foreground/80 hover:text-primary transition">
                AsuraScans
              </a>
              <span className="text-muted-foreground/40">·</span>
              <a href="https://groq.com" target="_blank" rel="noopener noreferrer" className="text-foreground/80 hover:text-primary transition">
                Groq
              </a>
              <span className="text-muted-foreground/40">·</span>
              <span className="text-foreground/80">VLM · edge-tts · ffmpeg · YOLO</span>
            </div>
            <div className="flex items-center gap-3">
              <span>For personal use only</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
