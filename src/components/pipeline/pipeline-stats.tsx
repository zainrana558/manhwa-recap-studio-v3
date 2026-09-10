"use client";

import { useEffect, useState, useRef } from "react";
import {
  BarChart3,
  ImageIcon,
  Layers,
  Film,
  Clock,
  TrendingUp,
  Mic,
  Activity,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useSectionObserver } from "@/hooks/use-section-observer";
import type { JobDetail, JobStatus } from "@/types/pipeline";

const ACTIVE_STATUSES = new Set<JobStatus>([
  "pending",
  "scraping",
  "transcribing",
  "translating",
  "rendering",
  "merging",
]);

function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const STATUS_BAR_COLORS: Record<string, { bg: string; label: string }> = {
  done: { bg: "bg-emerald-500", label: "Done" },
  error: { bg: "bg-rose-500", label: "Error" },
  cancelled: { bg: "bg-muted-foreground/40", label: "Cancelled" },
  active: { bg: "", label: "Active" },
};

const STATUS_ICON_MAP: Record<JobStatus, { color: string; bg: string }> = {
  pending: { color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" },
  awaiting_review: { color: "text-amber-300", bg: "bg-amber-500/10 border-amber-500/20" },
  scraping: { color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" },
  transcribing: { color: "text-orange-400", bg: "bg-orange-500/10 border-orange-500/20" },
  translating: { color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/20" },
  rendering: { color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20" },
  merging: { color: "text-teal-400", bg: "bg-teal-500/10 border-teal-500/20" },
  done: { color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20" },
  error: { color: "text-rose-400", bg: "bg-rose-500/10 border-rose-500/20" },
  cancelled: { color: "text-muted-foreground", bg: "bg-muted/50 border-border" },
};

export function PipelineStats() {
  const [jobs, setJobs] = useState<JobDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState(false);
  const { ref, isVisible } = useSectionObserver(0.05);
  const isFirstFetchDone = useRef(false);

  useEffect(() => {
    let cancelled = false;

    const fetchJobs = async () => {
      try {
        const res = await fetch("/api/jobs");
        const data = await res.json();
        if (!cancelled) {
          setJobs(data.jobs ?? []);
          if (!isFirstFetchDone.current) {
            isFirstFetchDone.current = true;
            setLoading(false);
            setPolling(true);
          }
        }
      } catch {
        if (!isFirstFetchDone.current && !cancelled) {
          isFirstFetchDone.current = true;
          setLoading(false);
        }
      }
    };

    fetchJobs();
    const interval = setInterval(fetchJobs, 15000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // --- Calculations ---
  const doneJobs = jobs.filter((j) => j.status === "done");
  const errorJobs = jobs.filter((j) => j.status === "error");
  const cancelledJobs = jobs.filter((j) => j.status === "cancelled");
  const activeJobs = jobs.filter((j) => ACTIVE_STATUSES.has(j.status));
  const nonActiveJobs = jobs.filter((j) => !ACTIVE_STATUSES.has(j.status));

  const total = jobs.length;
  const doneCount = doneJobs.length;
  const errorCount = errorJobs.length;
  const cancelledCount = cancelledJobs.length;
  const activeCount = activeJobs.length;

  // Total processing stats (from done jobs)
  const totalImages = doneJobs.reduce((s, j) => s + (j.totalImages || 0), 0);
  const totalChapters = doneJobs.reduce((s, j) => s + (j.totalChapters || 0), 0);
  const videosCreated = doneCount;

  // Average processing time (done jobs)
  const avgTimeMinutes =
    doneJobs.length > 0
      ? doneJobs.reduce((sum, j) => {
          const start = new Date(j.createdAt).getTime();
          const end = new Date(j.updatedAt).getTime();
          return sum + (end - start) / 60000;
        }, 0) / doneJobs.length
      : 0;

  // Success rate
  const successRate =
    nonActiveJobs.length > 0
      ? Math.round((doneCount / nonActiveJobs.length) * 100)
      : 0;

  // Most used voice
  const voiceCounts: Record<string, number> = {};
  doneJobs.forEach((j) => {
    const v = j.voice || "default";
    voiceCounts[v] = (voiceCounts[v] || 0) + 1;
  });
  const topVoice =
    Object.entries(voiceCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

  // Recent activity (last 5)
  const recentJobs = [...jobs]
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    )
    .slice(0, 5);

  // Status bar segments
  const barSegments = [
    { key: "done", count: doneCount, color: "bg-emerald-500" },
    { key: "error", count: errorCount, color: "bg-rose-500" },
    { key: "cancelled", count: cancelledCount, color: "bg-muted-foreground/40" },
    { key: "active", count: activeCount, color: "bg-amber-500" },
  ];

  // Live indicator element
  const liveIndicator = polling && !loading ? (
    <span className="inline-flex items-center gap-1.5 ml-2">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
      </span>
      <span className="text-[10px] font-medium text-emerald-400">Live</span>
    </span>
  ) : null;

  // Empty state
  if (!loading && total === 0) {
    return (
      <section
        ref={ref}
        className={`max-w-5xl mx-auto transition-all duration-700 ${
          isVisible ? "animate-section-in" : "opacity-0"
        }`}
      >
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 mb-3">
            <BarChart3 className="h-3.5 w-3.5 text-primary" />
            <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
              Pipeline Stats
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
            Your Studio Analytics
          </h2>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-lg mx-auto leading-relaxed">
            Overview of your video creation pipeline performance
          </p>
        </div>
        <div className="flex flex-col items-center justify-center py-12 animate-fade-in-up">
          <div className="p-4 rounded-2xl bg-muted/30 border border-border mb-4">
            <BarChart3 className="h-8 w-8 text-muted-foreground/30" />
          </div>
          <p className="text-sm text-muted-foreground">
            No jobs yet — stats will appear here once you create your first
            recap video.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="max-w-5xl mx-auto py-4">
      <div
        ref={ref}
        className={`transition-all duration-700 ${
          isVisible ? "animate-section-in" : "opacity-0"
        }`}
      >
        {/* Section heading */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 mb-3">
            <BarChart3 className="h-3.5 w-3.5 text-primary" />
            <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
              Pipeline Stats
            </span>
            {liveIndicator}
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
            Your Studio Analytics
          </h2>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-lg mx-auto leading-relaxed">
            Overview of your video creation pipeline performance
          </p>
        </div>

        {/* Status Distribution Bar */}
        <div
          className={`rounded-xl bg-gradient-to-br from-card/60 via-primary/[0.03] to-card/60 border border-border p-4 mb-4 hover-glow-sm ${
            isVisible ? "animate-item-in" : "opacity-0"
          }`}
          style={{ animationDelay: isVisible ? "80ms" : "0ms" }}
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">Job Status Distribution</h3>
            <span className="text-xs text-muted-foreground">
              {total} total
            </span>
          </div>
          {/* Stacked bar */}
          <div className="flex h-3 rounded-full overflow-hidden bg-muted/50 shadow-[inset_0_1px_2px_oklch(0_0_0/0.2)]">
            {barSegments.map((seg) => {
              if (seg.count === 0) return null;
              const pct = (seg.count / total) * 100;
              return (
                <div
                  key={seg.key}
                  className={`${seg.color} transition-all duration-700 shadow-[inset_0_1px_0_oklch(1/0.2),inset_0_-1px_0_oklch(0/0.15)]`}
                  style={{ width: `${pct}%` }}
                  title={`${STATUS_BAR_COLORS[seg.key].label}: ${seg.count}`}
                />
              );
            })}
          </div>
          {/* Legend */}
          <div className="flex flex-wrap items-center gap-4 mt-3">
            {barSegments.map((seg) => {
              const info = STATUS_BAR_COLORS[seg.key];
              return (
                <div key={seg.key} className="flex items-center gap-1.5">
                  <div
                    className={`h-2.5 w-2.5 rounded-sm ${seg.color}`}
                  />
                  <span className="text-xs text-muted-foreground">
                    {info.label}
                  </span>
                  <span className="text-xs font-semibold text-foreground">
                    {seg.count}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Stat cards grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          {/* Total Images */}
          <StatCard
            icon={ImageIcon}
            label="Images Processed"
            value={totalImages.toLocaleString()}
            delay={160}
            isVisible={isVisible}
          />
          {/* Total Chapters */}
          <StatCard
            icon={Layers}
            label="Chapters Processed"
            value={totalChapters.toLocaleString()}
            delay={240}
            isVisible={isVisible}
          />
          {/* Videos Created */}
          <StatCard
            icon={Film}
            label="Videos Created"
            value={videosCreated.toLocaleString()}
            delay={320}
            isVisible={isVisible}
          />
          {/* Avg Processing Time */}
          <StatCard
            icon={Clock}
            label="Avg Processing Time"
            value={
              avgTimeMinutes > 0
                ? `${Math.round(avgTimeMinutes)}min`
                : "—"
            }
            delay={400}
            isVisible={isVisible}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
          {/* Success Rate */}
          <div
            className={`rounded-xl bg-card/60 border border-border p-4 hover:border-primary/20 transition-all duration-300 hover-lift hover-glow-sm relative ${
              isVisible ? "animate-item-in" : "opacity-0"
            }`}
            style={{ animationDelay: isVisible ? "480ms" : "0ms" }}
          >
            <div
              className="absolute top-0 inset-x-0 h-px rounded-t-xl"
              style={{
                background: "linear-gradient(90deg, transparent, oklch(0.78 0.17 65 / 0.3), transparent)",
              }}
            />
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/10">
                  <TrendingUp className="h-4 w-4 text-emerald-400" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">
                    Success Rate
                  </p>
                  <p className="text-lg font-bold animate-count-pulse">
                    {nonActiveJobs.length > 0
                      ? `${successRate}%`
                      : "—"}{" "}
                    <span className="text-xs font-normal text-muted-foreground">
                      ({doneCount}/{nonActiveJobs.length} completed)
                    </span>
                  </p>
                </div>
              </div>
              {/* Mini progress ring */}
              {nonActiveJobs.length > 0 && (
                <div className="relative h-10 w-10">
                  <svg className="h-10 w-10 -rotate-90" viewBox="0 0 36 36">
                    <defs>
                      <filter id="ring-glow">
                        <feDropShadow dx="0" dy="0" stdDeviation="1.5" flood-color="oklch(0.7 0.2 160 / 0.5)" />
                      </filter>
                    </defs>
                    <circle
                      cx="18"
                      cy="18"
                      r="15.5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="3"
                      className="text-muted/50"
                    />
                    <circle
                      cx="18"
                      cy="18"
                      r="15.5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="3"
                      strokeDasharray={`${(successRate / 100) * 97.4} 97.4`}
                      strokeLinecap="round"
                      className="text-emerald-500"
                      filter="url(#ring-glow)"
                    />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold">
                    {successRate}%
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Most Used Voice */}
          <div
            className={`rounded-xl bg-card/60 border border-border p-4 hover:border-primary/20 transition-all duration-300 hover-lift hover-glow-sm relative ${
              isVisible ? "animate-item-in" : "opacity-0"
            }`}
            style={{ animationDelay: isVisible ? "560ms" : "0ms" }}
          >
            <div
              className="absolute top-0 inset-x-0 h-px rounded-t-xl"
              style={{
                background: "linear-gradient(90deg, transparent, oklch(0.78 0.17 65 / 0.3), transparent)",
              }}
            />
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Mic className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">
                  Most Used Voice
                </p>
                <p className="text-lg font-bold animate-count-pulse">
                  {topVoice
                    ? topVoice.charAt(0).toUpperCase() + topVoice.slice(1)
                    : "—"}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div
          className={`rounded-xl bg-card/60 border border-border p-4 hover:border-primary/20 transition-all duration-300 hover-glow-sm relative ${
            isVisible ? "animate-item-in" : "opacity-0"
          }`}
          style={{ animationDelay: isVisible ? "640ms" : "0ms" }}
        >
          <div
            className="absolute top-0 inset-x-0 h-px rounded-t-xl"
            style={{
              background: "linear-gradient(90deg, transparent, oklch(0.78 0.17 65 / 0.3), transparent)",
            }}
          />
          <div className="flex items-center gap-2 mb-3">
            <Activity className="h-3.5 w-3.5 text-primary" />
            <h3 className="text-sm font-semibold">Recent Activity</h3>
          </div>
          {recentJobs.length === 0 ? (
            <p className="text-xs text-muted-foreground py-4 text-center">
              No recent activity
            </p>
          ) : (
            <div className="space-y-2">
              {recentJobs.map((job) => {
                const style = STATUS_ICON_MAP[job.status];
                return (
                  <div
                    key={job.id}
                    className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span
                        className={`text-xs font-medium capitalize px-2 py-0.5 rounded-md border ${style?.bg} ${style?.color}`}
                      >
                        {job.status}
                      </span>
                      <span className="text-sm truncate">
                        {job.mangaTitle}
                      </span>
                    </div>
                    <span className="text-xs text-muted-foreground flex-shrink-0 ml-2">
                      {timeAgo(job.createdAt)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  delay,
  isVisible,
}: {
  icon: typeof BarChart3;
  label: string;
  value: string;
  delay: number;
  isVisible: boolean;
}) {
  return (
    <div
      className={`rounded-xl bg-card/60 border border-border p-4 hover:border-primary/20 transition-all duration-300 group hover-glow-sm relative ${
        isVisible ? "animate-item-in" : "opacity-0"
      }`}
      style={{ animationDelay: isVisible ? `${delay}ms` : "0ms" }}
    >
      <div
        className="absolute top-0 inset-x-0 h-px rounded-t-xl"
        style={{
          background: "linear-gradient(90deg, transparent, oklch(0.78 0.17 65 / 0.3), transparent)",
        }}
      />
      <div className="flex items-center gap-2.5 mb-2">
        <div className="p-1.5 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
          <Icon className="h-3.5 w-3.5 text-primary" />
        </div>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className="text-xl font-bold tracking-tight animate-count-pulse">{value}</p>
    </div>
  );
}
