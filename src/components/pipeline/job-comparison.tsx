"use client";

import { useState, useCallback } from "react";
import { GitCompareArrows, X, CheckCircle2, AlertCircle, Clock, Loader2, ArrowUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { JobDetail, JobStatus } from "@/types/pipeline";

const statusIcon: Record<JobStatus, typeof Clock> = {
  pending: Clock,
  scraping: Loader2,
  transcribing: Loader2,
  translating: Loader2,
  rendering: Loader2,
  merging: Loader2,
  done: CheckCircle2,
  error: AlertCircle,
  cancelled: X,
};

function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatDuration(start: string, end: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m ${rs}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

function formatShortVoice(voice: string): string {
  // Extract just the name part: en-US-AndrewNeural → Andrew
  const parts = voice.split("-");
  return parts.slice(2).join("-").replace("Neural", "").replace("Multilingual", "ML");
}

interface ComparisonRow {
  label: string;
  key: string;
  format?: (val: JobDetail) => React.ReactNode;
}

const COMPARISON_ROWS: ComparisonRow[] = [
  { label: "Status", key: "status" },
  { label: "Progress", key: "progress", format: (j) => <span className="font-mono">{j.progress}%</span> },
  { label: "Chapters", key: "chapters", format: (j) => `${j.doneChapters} / ${j.totalChapters}` },
  { label: "Images", key: "images", format: (j) => `${j.doneImages} / ${j.totalImages}` },
  { label: "Voice", key: "voice", format: (j) => <span className="text-xs">{formatShortVoice(j.voice)}</span> },
  { label: "Language", key: "language", format: (j) => <span className="uppercase text-xs">{j.language}</span> },
  { label: "Translate", key: "translate", format: (j) => j.translate ? "Yes" : "No" },
  { label: "Stage", key: "stage", format: (j) => j.stage ? <span className="capitalize text-xs">{j.stage}</span> : "—" },
  { label: "Created", key: "createdAt", format: (j) => <span className="text-xs">{timeAgo(j.createdAt)}</span> },
  { label: "Updated", key: "updatedAt", format: (j) => <span className="text-xs">{timeAgo(j.updatedAt)}</span> },
  {
    label: "Duration",
    key: "duration",
    format: (j) =>
      j.status === "done" ? formatDuration(j.createdAt, j.updatedAt) : <span className="text-muted-foreground">In progress</span>,
  },
  {
    label: "Archive",
    key: "archive",
    format: (j) =>
      j.archiveProvider ? (
        <span className="text-xs text-emerald-400">{j.archiveProvider}</span>
      ) : j.autoArchive ? (
        <span className="text-xs text-amber-400">Pending</span>
      ) : (
        <span className="text-muted-foreground text-xs">None</span>
      ),
  },
];

interface JobComparisonProps {
  jobs: JobDetail[];
}

export function JobComparison({ jobs }: JobComparisonProps) {
  const [open, setOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const completedJobs = jobs.filter((j) => j.status === "done" || j.status === "error");

  const toggleJob = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 2) {
        next.add(id);
      }
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  const jobA = selectedIds.size >= 1 ? jobs.find((j) => j.id === Array.from(selectedIds)[0]) ?? null : null;
  const jobB = selectedIds.size >= 2 ? jobs.find((j) => j.id === Array.from(selectedIds)[1]) ?? null : null;

  const canCompare = selectedIds.size === 2 && jobA && jobB;

  // Reset selection when dialog closes (render-time pattern)
  const [prevOpen, setPrevOpen] = useState(false);
  if (!open && prevOpen) {
    setPrevOpen(false);
    setSelectedIds(new Set());
  }
  if (open && !prevOpen) {
    setPrevOpen(true);
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        className="gap-1.5 text-xs"
        disabled={completedJobs.length < 2}
      >
        <GitCompareArrows className="h-3.5 w-3.5" />
        Compare
      </Button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)} />

          {/* Dialog */}
          <div className="relative w-full max-w-4xl max-h-[85vh] rounded-xl border border-border bg-background shadow-2xl animate-fade-in-up flex flex-col overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-primary/10">
                  <GitCompareArrows className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">Compare Jobs</h3>
                  <p className="text-[11px] text-muted-foreground">
                    Select 2 completed or failed jobs to compare side by side
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {selectedIds.size > 0 && (
                  <Button variant="ghost" size="sm" onClick={clearSelection} className="h-7 text-xs">
                    Clear
                  </Button>
                )}
                <button
                  onClick={() => setOpen(false)}
                  className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
                  aria-label="Close"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto scrollbar-thin">
              {!canCompare ? (
                /* Selection mode */
                <div className="p-4 space-y-3">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Badge variant="secondary" className="text-[10px]">
                      {selectedIds.size} / 2 selected
                    </Badge>
                    <span>Click jobs to select them for comparison</span>
                  </div>

                  <div className="space-y-1.5 max-h-[50vh] overflow-y-auto scrollbar-thin">
                    {completedJobs.length === 0 ? (
                      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
                        No completed or failed jobs to compare
                      </div>
                    ) : (
                      completedJobs.map((job) => {
                        const isSelected = selectedIds.has(job.id);
                        const Icon = statusIcon[job.status] ?? Clock;
                        return (
                          <button
                            key={job.id}
                            type="button"
                            onClick={() => toggleJob(job.id)}
                            className={cn(
                              "w-full flex items-center gap-3 p-3 rounded-lg border text-left transition-all",
                              isSelected
                                ? "border-primary/40 bg-primary/5"
                                : "border-border bg-card hover:border-primary/20 hover:bg-accent/30"
                            )}
                          >
                            {/* Selection indicator */}
                            <div
                              className={cn(
                                "w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 transition-colors",
                                isSelected ? "border-primary bg-primary" : "border-muted-foreground/30"
                              )}
                            >
                              {isSelected && (
                                <CheckCircle2 className="h-3 w-3 text-primary-foreground" />
                              )}
                            </div>

                            {/* Cover */}
                            <div className="w-8 h-11 rounded overflow-hidden bg-muted flex-shrink-0 border border-border">
                              {job.coverUrl ? (
                                <img src={job.coverUrl} alt="" className="w-full h-full object-cover" />
                              ) : null}
                            </div>

                            {/* Info */}
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">{job.mangaTitle}</p>
                              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <Icon className={cn("h-3 w-3", job.status === "done" ? "text-emerald-400" : "text-rose-400", job.status !== "done" && job.status !== "error" && job.status !== "cancelled" && "animate-spin")} />
                                <span className="capitalize">{job.status}</span>
                                <span className="text-muted-foreground/40">·</span>
                                <span>{job.totalChapters} ch</span>
                                <span className="text-muted-foreground/40">·</span>
                                <span>{job.progress}%</span>
                              </div>
                            </div>

                            <span className="text-[10px] text-muted-foreground">{timeAgo(job.createdAt)}</span>
                          </button>
                        );
                      })
                    )}
                  </div>
                </div>
              ) : (
                <div className="p-4 space-y-4">
                  {/* Job headers */}
                  <div className="grid grid-cols-[140px_1fr_1fr] gap-3">
                    <div /> {/* Empty corner */}
                    <JobHeader job={jobA!} />
                    <JobHeader job={jobB!} />
                  </div>

                  <Separator />

                  {/* Comparison rows */}
                  <div className="space-y-0">
                    {COMPARISON_ROWS.map((row, idx) => (
                      <div
                        key={row.key}
                        className={cn(
                          "grid grid-cols-[140px_1fr_1fr] gap-3 py-2.5 px-2 rounded-md",
                          idx % 2 === 0 && "bg-muted/20"
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-muted-foreground">{row.label}</span>
                        </div>
                        <div className="flex items-center text-sm">
                          {row.format ? row.format(jobA!) : String((jobA as unknown as Record<string, unknown>)[row.key] ?? "—")}
                        </div>
                        <div className="flex items-center text-sm">
                          {row.format ? row.format(jobB!) : String((jobB as unknown as Record<string, unknown>)[row.key] ?? "—")}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Progress comparison visual */}
                  <Separator />
                  <div className="space-y-3">
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Progress Comparison</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <ProgressCard job={jobA!} label="Job A" />
                      <ProgressCard job={jobB!} label="Job B" />
                    </div>
                  </div>

                  {/* Back button */}
                  <div className="flex justify-center pt-2">
                    <Button variant="outline" size="sm" onClick={clearSelection} className="gap-1.5 text-xs">
                      <ArrowUpDown className="h-3.5 w-3.5" />
                      Select different jobs
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function JobHeader({ job }: { job: JobDetail }) {
  return (
    <div className="flex items-center gap-2.5 p-2 rounded-lg bg-card border border-border">
      <div className="w-10 h-14 rounded overflow-hidden bg-muted flex-shrink-0 border border-border">
        {job.coverUrl ? (
          <img src={job.coverUrl} alt={job.mangaTitle} className="w-full h-full object-cover" />
        ) : null}
      </div>
      <div className="min-w-0">
        <p className="text-sm font-semibold truncate">{job.mangaTitle}</p>
        <Badge
          variant={job.status === "done" ? "default" : "destructive"}
          className="text-[10px] mt-1"
        >
          {job.status}
        </Badge>
      </div>
    </div>
  );
}

function ProgressCard({ job, label }: { job: JobDetail; label: string }) {
  const isSuccess = job.status === "done";
  const pct = job.totalChapters > 0 ? Math.round((job.doneChapters / job.totalChapters) * 100) : 0;
  const imgPct = job.totalImages > 0 ? Math.round((job.doneImages / job.totalImages) * 100) : 0;

  return (
    <div className="p-3 rounded-lg border border-border bg-card space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{label}</span>
        <span className={cn("text-xs", isSuccess ? "text-emerald-400" : "text-rose-400")}>
          {job.progress}%
        </span>
      </div>

      {/* Chapter progress */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span>Chapters</span>
          <span>{job.doneChapters}/{job.totalChapters}</span>
        </div>
        <div className="h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${pct}%`,
              background: isSuccess
                ? "oklch(0.7 0.15 160)"
                : "oklch(0.62 0.22 25)",
            }}
          />
        </div>
      </div>

      {/* Image progress */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span>Images</span>
          <span>{job.doneImages}/{job.totalImages}</span>
        </div>
        <div className="h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${imgPct}%`,
              background: isSuccess
                ? "oklch(0.75 0.18 195)"
                : "oklch(0.62 0.22 25)",
            }}
          />
        </div>
      </div>
    </div>
  );
}
