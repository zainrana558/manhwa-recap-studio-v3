"use client";

import { useState } from "react";
import { Search, ScanLine, FileText, Scissors, Clapperboard, ChevronDown, Workflow } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSectionObserver } from "@/hooks/use-section-observer";

const STEPS = [
  {
    icon: Search,
    title: "Search",
    desc: "Enter any manhwa title. We query 6 sources at once — MangaHere, FanFox, Webtoons, AsuraScans, MAL & AniList.",
    details: "Our parallel search engine hits all 6 manga databases simultaneously, returning results in under 5 seconds. Each source has specialized scrapers that understand the site structure.",
    duration: "~5s",
  },
  {
    icon: ScanLine,
    title: "Download",
    desc: "Scrapes all panel images from every chapter automatically using source-specific scrapers.",
    details: "Source-specific scrapers navigate chapter listings, download every panel image in order, and deduplicate across mirror sources. Supports MangaHere, FanFox, Webtoons, and AsuraScans formats.",
    duration: "~30s/ch",
  },
  {
    icon: FileText,
    title: "OCR",
    desc: "PaddleOCR PP-OCRv4 reads speech bubbles and captions from each panel — fast, local, no API keys needed.",
    details: "PP-OCRv4 is the primary transcription engine. It runs entirely on-device with no API keys. If OCR quality is low, VLM providers (SiliconFlow, Gemini, Groq) serve as automatic fallback.",
    duration: "~15s/ch",
  },
  {
    icon: Scissors,
    title: "Detect Panels",
    desc: "YOLO + contour-based detection finds panel boundaries. No chopping — each panel stays complete.",
    details: "A YOLO model identifies individual panels within each manga page, then contour-based analysis refines boundaries to ensure panels are never split. Each panel becomes a video frame.",
    duration: "~30s/ch",
  },
  {
    icon: Clapperboard,
    title: "Render Video",
    desc: "Panels are voiced with edge-tts narration and merged into a single recap MP4 with ffmpeg.",
    details: "edge-tts generates natural-sounding narration from panel text in 55+ voices across 8 English accents. ffmpeg syncs panel images to audio, adds transitions, optional BGM, and renders the final MP4.",
    duration: "~2min/ch",
  },
];

function StepCard({ s, i, isLast, expandedStep, setExpandedStep, hoveredStep, setHoveredStep, isVisible }: {
  s: typeof STEPS[number];
  i: number;
  isLast: boolean;
  expandedStep: number | null;
  setExpandedStep: (v: number | null) => void;
  hoveredStep: number | null;
  setHoveredStep: (v: number | null) => void;
  isVisible: boolean;
}) {
  const Icon = s.icon;
  const isExpanded = expandedStep === i;
  const isHovered = hoveredStep === i;
  const isActive = isHovered || isExpanded;

  const staggerClasses = [
    "stagger-1",
    "stagger-2",
    "stagger-3",
    "stagger-4",
    "stagger-5",
  ];

  return (
    <div className="flex items-stretch flex-1 min-w-0">
      <div
        className={cn(
          "group p-5 rounded-2xl border space-y-4 relative overflow-hidden transition-all duration-300 flex-1 cursor-pointer hover-lift hover-glow-sm animate-item-in",
          isActive
            ? "animate-glow-border bg-card/90 shadow-xl shadow-primary/10 border-primary/30"
            : "border-border bg-card/40 hover:border-primary/20 hover:bg-card/70 hover:shadow-lg hover:shadow-primary/5",
          isVisible && staggerClasses[i]
        )}
        onMouseEnter={() => setHoveredStep(i)}
        onMouseLeave={() => setHoveredStep(null)}
        onClick={() => setExpandedStep(isExpanded ? null : i)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setExpandedStep(isExpanded ? null : i);
          }
        }}
      >
        <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-primary/40 via-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

        <div className="flex items-center justify-between">
          <div
            className={cn(
              "p-2.5 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-all duration-300",
              isActive && "animate-icon-bounce shadow-md shadow-primary/10",
              "hover:shadow-lg hover:shadow-primary/20"
            )}
          >
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-muted-foreground/60 px-2 py-0.5 rounded-md bg-muted/50 border border-border">
              {s.duration}
            </span>
            <span
              className={cn(
                "text-3xl font-bold select-none",
                isActive && "border-glow rounded-lg px-1"
              )}
              style={{
                background: "linear-gradient(135deg, oklch(0.78 0.17 65 / 0.15), oklch(0.78 0.17 65 / 0.02))",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
          </div>
        </div>

        <div className="relative z-[2]">
          <h3 className="text-sm font-bold mb-1.5 group-hover:text-primary transition-colors">{s.title}</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">{s.desc}</p>
        </div>

        {isExpanded && (
          <div className="animate-fade-in-up pt-3 border-t border-border/50 relative z-[2]">
            <p className="text-xs text-muted-foreground/80 leading-relaxed">
              {s.details}
            </p>
          </div>
        )}

        <ChevronDown
          className={cn(
            "h-3 w-3 text-muted-foreground/40 transition-transform duration-200 absolute bottom-4 right-4",
            isExpanded && "rotate-180"
          )}
        />
      </div>

      {!isLast && (
        <div className="flex items-center px-1.5 flex-shrink-0">
          <div
            className="h-[2px] w-10 flex-shrink-0 rounded-full"
            style={{
              background: "linear-gradient(90deg, oklch(0.78 0.17 65 / 0.4), oklch(0.78 0.17 65 / 0.05))",
              backgroundSize: "200% 100%",
              animation: "gradient-slide 2s linear infinite",
            }}
          />
        </div>
      )}
    </div>
  );
}

export function HowItWorks() {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const { ref, isVisible } = useSectionObserver(0.1);

  const staggerClasses = [
    "stagger-1",
    "stagger-2",
    "stagger-3",
    "stagger-4",
    "stagger-5",
  ];

  return (
    <section ref={ref} className="max-w-6xl mx-auto py-8">
      <div className={`transition-all duration-700 ${isVisible ? "animate-section-in" : "opacity-0"}`}>
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 mb-3">
            <Workflow className="h-3.5 w-3.5 text-primary" />
            <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
              How It Works
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">5-Step Automated Pipeline</h2>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-lg mx-auto leading-relaxed">
            From search to finished video in minutes — fully automated
          </p>
        </div>

        <div className="hidden lg:flex items-stretch gap-0">
          {STEPS.map((s, i) => (
            <StepCard
              key={s.title}
              s={s}
              i={i}
              isLast={i === STEPS.length - 1}
              expandedStep={expandedStep}
              setExpandedStep={setExpandedStep}
              hoveredStep={hoveredStep}
              setHoveredStep={setHoveredStep}
              isVisible={isVisible}
            />
          ))}
        </div>

        <div className="flex flex-col lg:hidden gap-0 relative">
          <div className="absolute left-[18px] top-6 bottom-6 w-[2px]">
            <div
              className="w-full h-full rounded-full"
              style={{
                background: "linear-gradient(180deg, oklch(0.78 0.17 65 / 0.4), oklch(0.78 0.17 65 / 0.08))",
                backgroundSize: "100% 200%",
                animation: "gradient-slide 3s linear infinite",
              }}
            />
          </div>

          {STEPS.map((s, i) => {
            const Icon = s.icon;
            const isExpanded = expandedStep === i;
            const isHovered = hoveredStep === i;
            const isLast = i === STEPS.length - 1;
            const isActive = isHovered || isExpanded;

            return (
              <div key={s.title} className={cn("flex gap-4", !isLast && "pb-3")}>
                <div className="relative z-10 flex-shrink-0 mt-5">
                  <div className={cn(
                    "w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-300 border",
                    isActive
                      ? "bg-primary/20 border-primary/40 shadow-md shadow-primary/10 border-glow"
                      : "bg-card border-border hover:shadow-md hover:shadow-primary/10"
                  )}>
                    <Icon className="h-4 w-4 text-primary" />
                  </div>
                </div>

                <div
                  className={cn(
                    "group flex-1 p-4 rounded-xl border space-y-3 relative overflow-hidden transition-all duration-300 cursor-pointer mb-1 hover-lift hover-glow-sm animate-item-in",
                    isActive
                      ? "animate-glow-border bg-card/90 shadow-lg shadow-primary/5 border-primary/30"
                      : "border-border bg-card/40 hover:border-primary/20 hover:bg-card/70",
                    isVisible && staggerClasses[i]
                  )}
                  onMouseEnter={() => setHoveredStep(i)}
                  onMouseLeave={() => setHoveredStep(null)}
                  onClick={() => setExpandedStep(isExpanded ? null : i)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setExpandedStep(isExpanded ? null : i);
                    }
                  }}
                >
                  <div className="flex items-center justify-between relative z-[2]">
                    <h3 className="text-sm font-bold group-hover:text-primary transition-colors">{s.title}</h3>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-muted-foreground/60 px-2 py-0.5 rounded-md bg-muted/50 border border-border">
                        {s.duration}
                      </span>
                      <ChevronDown
                        className={cn(
                          "h-3 w-3 text-muted-foreground/40 transition-transform duration-200",
                          isExpanded && "rotate-180"
                        )}
                      />
                    </div>
                  </div>
                  <div className="relative z-[2]">
                    <p className="text-xs text-muted-foreground leading-relaxed">{s.desc}</p>
                  </div>
                  {isExpanded && (
                    <div className="animate-fade-in-up pt-2 border-t border-border/50 relative z-[2]">
                      <p className="text-xs text-muted-foreground/80 leading-relaxed">{s.details}</p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
