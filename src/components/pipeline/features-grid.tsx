"use client";

import { useSectionObserver } from "@/hooks/use-section-observer";
import {
  Search,
  ScanLine,
  FileText,
  Scissors,
  Clapperboard,
  Globe,
  Zap,
  Shield,
  Grid3X3,
  type LucideIcon,
} from "lucide-react";

interface Feature {
  icon: LucideIcon;
  title: string;
  desc: string;
  stat: string;
  gradient: string;
  glowColor: string;
}

const FEATURES: Feature[] = [
  {
    icon: Search,
    title: "Multi-Source Search",
    desc: "Query 6 manga databases simultaneously — MangaHere, FanFox, Webtoons, AsuraScans, MAL & AniList.",
    stat: "6 Sources",
    gradient: "from-amber-500/20 to-orange-500/10",
    glowColor: "shadow-amber-500/10",
  },
  {
    icon: ScanLine,
    title: "Auto Scraping",
    desc: "Source-specific scrapers download every panel image from every chapter, deduplicated across mirrors.",
    stat: "Auto Chapter",
    gradient: "from-emerald-500/20 to-teal-500/10",
    glowColor: "shadow-emerald-500/10",
  },
  {
    icon: FileText,
    title: "PaddleOCR Transcription",
    desc: "PP-OCRv5 reads speech bubbles and captions from each panel — fast, local, no API keys needed.",
    stat: "PP-OCRv5",
    gradient: "from-sky-500/20 to-blue-500/10",
    glowColor: "shadow-sky-500/10",
  },
  {
    icon: Scissors,
    title: "Panel Detection",
    desc: "YOLO + contour analysis finds panel boundaries ensuring each panel stays complete and un-split.",
    stat: "YOLO AI",
    gradient: "from-fuchsia-500/20 to-purple-500/10",
    glowColor: "shadow-fuchsia-500/10",
  },
  {
    icon: Clapperboard,
    title: "Video Rendering",
    desc: "Panels voiced with 55+ TTS voices and merged into a single recap MP4 with ffmpeg.",
    stat: "55+ Voices",
    gradient: "from-rose-500/20 to-pink-500/10",
    glowColor: "shadow-rose-500/10",
  },
  {
    icon: Globe,
    title: "Multi-Language",
    desc: "Translate from Korean, Japanese, Chinese, Spanish, French, German and more to English.",
    stat: "9+ Languages",
    gradient: "from-teal-500/20 to-cyan-500/10",
    glowColor: "shadow-teal-500/10",
  },
  {
    icon: Zap,
    title: "Groq Accelerated",
    desc: "Optional Groq LPU hardware delivers 3-5x faster VLM fallback transcription and narration rewriting.",
    stat: "3-5x Faster",
    gradient: "from-yellow-500/20 to-amber-500/10",
    glowColor: "shadow-yellow-500/10",
  },
  {
    icon: Shield,
    title: "100% Free & Local",
    desc: "Open-source stack with no paid requirements. All AI processing uses free-tier APIs.",
    stat: "$0 Cost",
    gradient: "from-lime-500/20 to-green-500/10",
    glowColor: "shadow-lime-500/10",
  },
];

export function FeaturesGrid() {
  const { ref, isVisible } = useSectionObserver(0.1);

  return (
    <section className="max-w-6xl mx-auto py-4">
      <div
        ref={ref}
        className={`transition-all duration-700 ${isVisible ? "animate-section-in" : "opacity-0"}`}
      >
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 mb-3">
            <Grid3X3 className="h-3.5 w-3.5 text-primary" />
            <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
              Platform Capabilities
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Everything you need</h2>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-lg mx-auto leading-relaxed">
            A complete toolkit for manhwa recap video creation
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                className={`group relative p-5 rounded-2xl border border-border bg-card/40 hover:border-primary/20 hover:bg-card/80 transition-all duration-300 hover:shadow-xl ${f.glowColor} cursor-default hover-glow-sm ${isVisible ? "animate-item-in" : "opacity-0"}`}
                style={{ animationDelay: isVisible ? `${i * 80}ms` : "0ms" }}
              >
                {/* Gradient accent at top */}
                <div className={`absolute inset-x-0 top-0 h-[2px] rounded-t-2xl bg-gradient-to-r ${f.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
                
                {/* Inner glow on hover */}
                <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${f.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                
                <div className="relative flex items-start gap-3">
                  <div className="p-2.5 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-all duration-300 group-hover:shadow-md group-hover:shadow-primary/10 relative">
                    <Icon className="h-4 w-4 text-primary" />
                    {/* Orbiting dot on icon container when hovered */}
                    <span className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <span className="w-1 h-1 rounded-full bg-primary opacity-0 group-hover:opacity-100 transition-opacity duration-300 animate-orbit" />
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-bold mb-1.5 group-hover:text-primary transition-colors">
                      {f.title}
                    </h3>
                    <p className="text-xs text-muted-foreground leading-relaxed group-hover:text-foreground/80 group-hover:text-[13px] transition-all duration-300">
                      {f.desc}
                    </p>
                  </div>
                </div>

                {/* Stat badge */}
                <div className="relative mt-4 flex justify-end">
                  <span className="text-[10px] font-mono font-medium px-2.5 py-1 rounded-full bg-muted/80 text-muted-foreground border border-border group-hover:border-primary/20 group-hover:text-foreground transition-colors">
                    {f.stat}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
