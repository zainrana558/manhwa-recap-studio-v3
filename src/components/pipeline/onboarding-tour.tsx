"use client";

import { useState, useEffect, useCallback } from "react";
import { X, Sparkles, Search, Settings, Keyboard } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useFirstVisit } from "@/hooks/use-section-observer";

interface TourStep {
  title: string;
  desc: string;
  target: string; // CSS selector or "center"
  icon: typeof Sparkles;
}

const STEPS: TourStep[] = [
  {
    title: "Welcome to Manhwa Recap Studio!",
    desc: "Turn any manhwa, manga, or webtoon into a narrated recap video in minutes. Let's take a quick tour.",
    target: "center",
    icon: Sparkles,
  },
  {
    title: "Search for a Manhwa",
    desc: "Type any title and search every source at once. Click a result to configure your video.",
    target: "#search-input",
    icon: Search,
  },
  {
    title: "Configure Settings",
    desc: "Adjust voice, language, chapter limits, and API keys in the settings dialog.",
    target: "[aria-label=\"Open settings\"]",
    icon: Settings,
  },
  {
    title: "Keyboard Shortcuts",
    desc: "Press / to focus the search bar, Esc to clear results. Press this button anytime to see shortcuts.",
    target: "[aria-label=\"Keyboard shortcuts\"]",
    icon: Keyboard,
  },
];

export function OnboardingTour() {
  const { isFirst, dismiss } = useFirstVisit();
  const [step, setStep] = useState(0);
  const [spotlightRect, setSpotlightRect] = useState<DOMRect | null>(null);

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  // Update spotlight position on step change
  useEffect(() => {
    if (!isFirst) return;

    const updatePosition = () => {
      if (current.target === "center") {
        setSpotlightRect(null);
        return;
      }
      const el = document.querySelector(current.target);
      if (el) {
        const rect = el.getBoundingClientRect();
        setSpotlightRect({
          top: rect.top - 8,
          left: rect.left - 8,
          width: rect.width + 16,
          height: rect.height + 16,
          bottom: rect.bottom + 8,
          right: rect.right + 8,
          x: rect.x - 8,
          y: rect.y - 8,
        } as DOMRect);
      }
    };

    // Small delay to let the DOM settle
    const timer = setTimeout(updatePosition, 100);
    window.addEventListener("resize", updatePosition);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", updatePosition);
    };
  }, [step, isFirst, current.target]);

  const handleDismiss = useCallback(() => {
    dismiss();
    setStep(0);
  }, [dismiss]);

  const handleNext = useCallback(() => {
    if (isLast) {
      handleDismiss();
    } else {
      setStep((s) => s + 1);
    }
  }, [isLast, handleDismiss]);

  const handlePrev = useCallback(() => {
    setStep((s) => Math.max(0, s - 1));
  }, []);

  if (!isFirst) return null;

  const Icon = current.icon;

  // Tooltip position: below the spotlight or centered
  const isCenter = current.target === "center";
  const tooltipStyle = isCenter
    ? { position: "fixed" as const, top: "50%", left: "50%", transform: "translate(-50%, -50%)" }
    : spotlightRect
      ? { position: "fixed" as const, top: `${spotlightRect.bottom + 16}px`, left: "50%", transform: "translateX(-50%)" }
      : { position: "fixed" as const, top: "50%", left: "50%", transform: "translate(-50%, -50%)" };

  return (
    <>
      {/* Backdrop overlay */}
      <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm" onClick={handleDismiss} />

      {/* Spotlight ring on target */}
      {spotlightRect && (
        <div
          className="fixed z-[101] rounded-xl border-2 border-primary animate-pulse-ring pointer-events-none"
          style={{
            top: spotlightRect.top,
            left: spotlightRect.left,
            width: spotlightRect.width,
            height: spotlightRect.height,
          }}
        />
      )}

      {/* Tooltip card */}
      <div
        className="fixed z-[102] w-[320px] sm:w-[380px] rounded-xl border border-border bg-card shadow-2xl shadow-black/40 p-5 space-y-4 animate-fade-in-up"
        style={tooltipStyle}
      >
        <button
          onClick={handleDismiss}
          className="absolute top-3 right-3 p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
          aria-label="Dismiss tour"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">{current.title}</h3>
            <p className="text-[10px] text-muted-foreground font-mono mt-0.5">
              Step {step + 1} of {STEPS.length}
            </p>
          </div>
        </div>

        <p className="text-sm text-muted-foreground leading-relaxed">
          {current.desc}
        </p>

        {/* Progress dots */}
        <div className="flex items-center gap-1.5">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                i === step ? "bg-primary" : i < step ? "bg-primary/40" : "bg-muted"
              }`}
            />
          ))}
        </div>

        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={handlePrev}
            disabled={step === 0}
            className="h-8 text-xs"
          >
            Back
          </Button>
          <Button
            size="sm"
            onClick={handleNext}
            className="h-8 text-xs"
          >
            {isLast ? "Get Started" : "Next"}
          </Button>
        </div>
      </div>
    </>
  );
}
