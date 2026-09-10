"use client";

import { useState, useEffect } from "react";
import { CheckCircle2, AlertCircle, Clock, Loader2, Cloud, Zap } from "lucide-react";
import { useSectionObserver } from "@/hooks/use-section-observer";
import type { JobStatus } from "@/types/pipeline";

interface Activity {
  id: string;
  mangaTitle: string;
  status: JobStatus;
  progress: number;
  createdAt: string;
  totalChapters: number;
  archiveProvider?: string | null;
}

const STATUS_CONFIG: Record<JobStatus, { icon: typeof CheckCircle2; color: string; label: string; bg: string }> = {
  pending: { icon: Clock, color: "text-amber-400", label: "Queued", bg: "bg-amber-500/10" },
  awaiting_review: { icon: Clock, color: "text-amber-300", label: "Review", bg: "bg-amber-500/10" },
  scraping: { icon: Loader2, color: "text-amber-400", label: "Scraping", bg: "bg-amber-500/10" },
  transcribing: { icon: Loader2, color: "text-orange-400", label: "Transcribing", bg: "bg-orange-500/10" },
  translating: { icon: Loader2, color: "text-purple-400", label: "Translating", bg: "bg-purple-500/10" },
  rendering: { icon: Loader2, color: "text-emerald-400", label: "Rendering", bg: "bg-emerald-500/10" },
  merging: { icon: Loader2, color: "text-teal-400", label: "Merging", bg: "bg-teal-500/10" },
  done: { icon: CheckCircle2, color: "text-emerald-400", label: "Completed", bg: "bg-emerald-500/10" },
  error: { icon: AlertCircle, color: "text-rose-400", label: "Failed", bg: "bg-rose-500/10" },
  cancelled: { icon: AlertCircle, color: "text-muted-foreground", label: "Cancelled", bg: "bg-muted/50" },
};

function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
  if (diff < 604800) return Math.floor(diff / 86400) + "d ago";
  return date.toLocaleDateString();
}

export function ActivityFeed() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const { ref, isVisible } = useSectionObserver(0.05);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/jobs?limit=5")
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) setActivities((data.jobs ?? []).slice(0, 5));
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  if (!loading && activities.length === 0) return null;

  return (
    <section ref={ref} className={"max-w-5xl mx-auto transition-all duration-700 " + (isVisible ? "animate-section-in" : "opacity-0")}>
      <div className="flex items-center gap-2 mb-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5">
          <Zap className="h-3.5 w-3.5 text-primary" />
          <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            Recent Activity
          </span>
        </div>
      </div>

      {loading ? (
        <div className="space-y-2">
          {[1, 2, 3].map((n) => (
            <div key={n} className="flex items-center gap-3 p-3 rounded-xl border border-border bg-card/40 animate-pulse">
              <div className="w-8 h-8 rounded-lg bg-muted/50" />
              <div className="flex-1 space-y-1.5">
                <div className="h-3 w-32 rounded bg-muted/50" />
                <div className="h-2.5 w-20 rounded bg-muted/30" />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="relative space-y-2 pl-4">
          {/* Vertical timeline line */}
          {activities.length > 1 && (
            <div
              className="absolute left-[7px] top-2 bottom-2 w-px"
              style={{
                background: "linear-gradient(180deg, oklch(0.78 0.17 65 / 0.3), transparent)",
              }}
            />
          )}
          {activities.map((a, i) => {
            const cfg = STATUS_CONFIG[a.status] ?? STATUS_CONFIG.pending;
            const Icon = cfg.icon;
            const isDone = a.status === "done";
            const isActive = ["scraping", "transcribing", "translating", "rendering", "merging"].includes(a.status);
            const statusBorderColor = isDone
              ? "border-l-emerald-500/40"
              : isActive
                ? "border-l-amber-500/40"
                : a.status === "error"
                  ? "border-l-rose-500/40"
                  : "border-l-border";
            return (
              <div
                key={a.id}
                className={
                  "relative flex items-center gap-3 p-3 rounded-xl border bg-card/40 transition-all duration-200 border-l-2 " +
                  statusBorderColor +
                  " " +
                  (isVisible ? "animate-item-in" : "opacity-0") +
                  " " +
                  (isDone
                    ? "border-emerald-500/15 hover:border-emerald-500/30"
                    : isActive
                      ? "border-amber-500/15"
                      : "border-border"
                  ) +
                  (isActive ? " animate-breathe" : "")
                }
                style={{ animationDelay: isVisible ? `${i * 60}ms` : "0ms" }}
              >
                {/* Timeline dot */}
                <div
                  className={"absolute -left-4 top-1/2 -translate-y-1/2 w-[7px] h-[7px] rounded-full border-2 border-background " + (isDone ? "bg-emerald-400" : isActive ? "bg-amber-400" : a.status === "error" ? "bg-rose-400" : "bg-muted-foreground/40")}
                />
                <div className={"p-1.5 rounded-lg " + cfg.bg + " flex-shrink-0 relative"}>
                  <Icon className={"h-3.5 w-3.5 " + cfg.color + (isActive ? " animate-spin" : "")} />
                  {isActive && (
                    <span className="absolute inset-0 rounded-lg animate-ping bg-current opacity-20" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{a.mangaTitle}</p>
                  <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                    <span className={cfg.color}>{cfg.label}</span>
                    {a.status !== "done" && a.status !== "error" ? (
                      <>
                        <span className="text-muted-foreground/30">·</span>
                        <span>{a.progress}%</span>
                      </>
                    ) : null}
                    {isDone && a.archiveProvider ? (
                      <>
                        <span className="text-muted-foreground/30">·</span>
                        <Cloud className="h-3 w-3 text-sky-400" />
                        <span className="text-sky-400">Archived</span>
                      </>
                    ) : null}
                  </div>
                </div>

                <div className="text-right flex-shrink-0">
                  <p className="text-[10px] text-muted-foreground/50">{timeAgo(a.createdAt)}</p>
                  <p className="text-[10px] text-muted-foreground/40">{a.totalChapters} ch</p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}