"use client";

import { useEffect, useState } from "react";
import { History, ChevronRight, Loader2, CheckCircle2, AlertCircle, Clock, XCircle, Trash2, Film, Cloud, Play, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { useSectionObserver } from "@/hooks/use-section-observer";
import { JobComparison } from "@/components/pipeline/job-comparison";
import { JobDetailModal } from "@/components/pipeline/job-detail-modal";
import type { JobDetail, JobStatus } from "@/types/pipeline";

interface JobHistoryProps {
  onSelectJob: (jobId: string) => void;
  refreshKey: number;
}

const statusIcon: Record<JobStatus, typeof Clock> = {
  pending: Clock,
  scraping: Loader2,
  transcribing: Loader2,
  translating: Loader2,
  rendering: Loader2,
  merging: Loader2,
  done: CheckCircle2,
  error: AlertCircle,
  cancelled: XCircle,
};

const statusColor: Record<JobStatus, string> = {
  pending: "text-amber-400",
  scraping: "text-amber-400",
  transcribing: "text-orange-400",
  translating: "text-purple-400",
  rendering: "text-emerald-400",
  merging: "text-teal-400",
  done: "text-emerald-400",
  error: "text-rose-400",
  cancelled: "text-muted-foreground",
};

const statusBgColor: Record<JobStatus, string> = {
  pending: "bg-amber-500/10 border-amber-500/20",
  scraping: "bg-amber-500/10 border-amber-500/20",
  transcribing: "bg-orange-500/10 border-orange-500/20",
  translating: "bg-purple-500/10 border-purple-500/20",
  rendering: "bg-emerald-500/10 border-emerald-500/20",
  merging: "bg-teal-500/10 border-teal-500/20",
  done: "bg-emerald-500/10 border-emerald-500/20",
  error: "bg-rose-500/10 border-rose-500/20",
  cancelled: "bg-muted/50 border-border",
};

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

function exportJobsCsv(jobs: JobDetail[]) {
  const header = ["ID", "Title", "Status", "Progress", "Chapters", "Images", "Voice", "Language", "Created", "Updated"];
  const rows = jobs.map((j) => [
    j.id,
    `"${(j.mangaTitle || "").replace(/"/g, '""')}"`,
    j.status,
    `${j.progress}%`,
    `${j.doneChapters}/${j.totalChapters}`,
    `${j.doneImages}/${j.totalImages}`,
    j.voice || "",
    j.language || "",
    j.createdAt,
    j.updatedAt,
  ].join(","));
  const csv = [header.join(","), ...rows].join("\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const now = new Date();
  const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  a.href = url;
  a.download = `manhwa-recap-jobs-${dateStr}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function JobHistory({ onSelectJob, refreshKey }: JobHistoryProps) {
  const [jobs, setJobs] = useState<JobDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [detailJob, setDetailJob] = useState<JobDetail | null>(null);
  const { ref, isVisible } = useSectionObserver(0.05);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/jobs")
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) setJobs(data.jobs ?? []);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const handleDelete = async (e: React.MouseEvent, job: JobDetail) => {
    e.stopPropagation();
    const isActive = ACTIVE_STATUSES.has(job.status);
    const confirmMsg = isActive
      ? `"${job.mangaTitle}" is still running. Stop and delete it?`
      : `Delete "${job.mangaTitle}" from recent jobs? This can't be undone.`;
    if (!window.confirm(confirmMsg)) return;

    setDeletingId(job.id);
    try {
      const res = await fetch(`/api/jobs/${job.id}?force=true`, { method: "DELETE" });
      if (res.ok) {
        setJobs((prev) => prev.filter((j) => j.id !== job.id));
      }
    } catch {
      // best-effort
    } finally {
      setDeletingId(null);
    }
  };

  const completedCount = jobs.filter((j) => j.status === "done").length;

  return (
    <section ref={ref} className={`max-w-5xl mx-auto transition-all duration-700 ${isVisible ? "animate-section-in" : "opacity-0"}`}>
      {/* Section heading */}
      <div className="flex items-center gap-3 mb-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5">
          <History className="h-3.5 w-3.5 text-primary" />
          <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            Job History
          </span>
        </div>
        {jobs.length > 0 && (
          <span className="text-xs text-muted-foreground/60">
            {jobs.length} job{jobs.length !== 1 ? "s" : ""}
            {completedCount > 0 && ` · ${completedCount} completed`}
          </span>
        )}
        {jobs.length > 0 && (
          <div className="ml-auto flex items-center gap-1.5">
            <JobComparison jobs={jobs} />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportJobsCsv(jobs)}
                  className="gap-1.5 text-xs"
                >
                  <Download className="h-3.5 w-3.5" />
                  Export CSV
                </Button>
              </TooltipTrigger>
              <TooltipContent>Export as CSV</TooltipContent>
            </Tooltip>
          </div>
        )}
        <button
          onClick={() => setOpen((o) => !o)}
          className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
          aria-label={open ? "Collapse history" : "Expand history"}
        >
          <ChevronRight className={`h-4 w-4 transition-transform duration-200 ${open ? "rotate-90" : ""}`} />
        </button>
      </div>

      {/* Empty state */}
      {!loading && jobs.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 space-y-4 animate-fade-in-up">
          <div className="relative">
            {/* Decorative background shapes */}
            <div className="absolute -inset-6 -z-10">
              <div className="absolute top-0 left-1/4 w-16 h-16 rounded-full border border-border/50 opacity-30" style={{ animation: "morph 8s ease-in-out infinite" }} />
              <div className="absolute bottom-2 right-1/4 w-12 h-12 rounded-lg border border-border/50 opacity-20 rotate-12" style={{ animation: "breathe 4s ease-in-out infinite" }} />
              <div className="absolute top-4 right-1/3 w-8 h-8 rounded-full bg-primary/5 opacity-40" style={{ animation: "breathe 6s ease-in-out infinite 1s" }} />
            </div>
            <div className="p-5 rounded-2xl bg-muted/30 border border-border">
              <Film className="h-10 w-10 text-muted-foreground/30" />
            </div>
            <div className="absolute -bottom-1 -right-1 p-2 rounded-xl bg-primary/10 border border-primary/20">
              <Play className="h-3.5 w-3.5 text-primary" />
            </div>
          </div>
          <div className="text-center space-y-1.5">
            <p className="text-sm font-semibold text-muted-foreground">No jobs yet</p>
            <p className="text-xs text-muted-foreground/60 max-w-xs leading-relaxed">
              Search for a manhwa above to create your first recap video. It takes about 6 minutes per chapter.
            </p>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground/40">
            <span className="px-2 py-0.5 rounded-md border border-border bg-muted/50">Scrape</span>
            <span className="text-primary/30">→</span>
            <span className="px-2 py-0.5 rounded-md border border-border bg-muted/50">Transcribe</span>
            <span className="text-primary/30">→</span>
            <span className="px-2 py-0.5 rounded-md border border-border bg-muted/50">Render</span>
            <span className="text-primary/30">→</span>
            <span className="px-2 py-0.5 rounded-md border border-primary/20 bg-primary/5 text-primary">Video!</span>
          </div>
        </div>
      )}

      {open && jobs.length > 0 && (
        <div className="space-y-2 max-h-96 overflow-y-auto scrollbar-thin">
          {jobs.map((job) => {
            const Icon = statusIcon[job.status] ?? Clock;
            const spinning = ["scraping", "transcribing", "translating", "rendering", "merging"].includes(job.status);
            const isDone = job.status === "done";
            return (
              <div
                key={job.id}
                role="button"
                tabIndex={0}
                onClick={() => setDetailJob(job)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") setDetailJob(job);
                }}
                className={`w-full flex items-center gap-3 p-3 rounded-xl border bg-card/60 hover:border-primary/40 hover:bg-accent/30 transition-all duration-200 text-left group cursor-pointer relative ${statusBgColor[job.status]} animate-fade-in-up`}
              >
                {/* Cover image */}
                <div className="w-10 h-14 rounded-lg overflow-hidden bg-muted flex-shrink-0 border border-border relative">
                  {job.coverUrl ? (
                    <img src={job.coverUrl} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Film className="h-4 w-4 text-muted-foreground/30" />
                    </div>
                  )}
                  {isDone && (
                    <div className="absolute bottom-0 right-0 p-0.5 rounded-tl bg-emerald-500/80">
                      <CheckCircle2 className="h-3 w-3 text-white" />
                    </div>
                  )}
                </div>

                {/* Progress bar */}
                {ACTIVE_STATUSES.has(job.status) && job.progress > 0 && job.progress < 100 && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-b-xl overflow-hidden">
                    <div
                      className="h-full transition-all duration-1000 ease-out"
                      style={{
                        width: `${job.progress}%`,
                        background: "linear-gradient(90deg, oklch(0.78 0.17 65), oklch(0.72 0.18 45))",
                      }}
                    />
                  </div>
                )}

                {/* Job info */}
                <div className="flex-1 min-w-0 space-y-1">
                  <p className="text-sm font-medium truncate group-hover:text-primary transition-colors">{job.mangaTitle}</p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    {/* Animated status indicator dot for active jobs */}
                    {ACTIVE_STATUSES.has(job.status) && (
                      <span
                        className="h-1.5 w-1.5 rounded-full inline-block"
                        style={{
                          background: statusColor[job.status].includes("amber") ? "oklch(0.78 0.17 65)" : statusColor[job.status].includes("orange") ? "oklch(0.72 0.18 45)" : statusColor[job.status].includes("purple") ? "oklch(0.65 0.2 280)" : statusColor[job.status].includes("emerald") ? "oklch(0.7 0.15 160)" : statusColor[job.status].includes("teal") ? "oklch(0.7 0.15 195)" : "oklch(0.78 0.17 65)",
                          animation: "pulse-glow 2s ease-in-out infinite",
                          boxShadow: `0 0 6px 1px ${statusColor[job.status].includes("amber") ? "oklch(0.78 0.17 65 / 0.4)" : "oklch(0.7 0.15 160 / 0.4)"}`,
                        }}
                      />
                    )}
                    <Icon className={`h-3.5 w-3.5 ${statusColor[job.status]} ${spinning ? "animate-spin" : ""}`} />
                    <span className={`capitalize ${statusColor[job.status]}`}>{job.status}</span>
                    <span className="text-muted-foreground/40">·</span>
                    <span>{job.totalChapters} ch</span>
                    <span className="text-muted-foreground/40">·</span>
                    <span>{job.progress}%</span>
                    <span className="text-muted-foreground/40">·</span>
                    <span>{timeAgo(job.createdAt)}</span>
                    {isDone && job.archiveProvider && (
                      <>
                        <span className="text-muted-foreground/40">·</span>
                        <span className="flex items-center gap-0.5 text-sky-400">
                          <Cloud className="h-3 w-3" />
                          <span className="text-[10px]">Mega</span>
                        </span>
                      </>
                    )}
                    {isDone && job.autoArchive && !job.archiveProvider && (
                      <>
                        <span className="text-muted-foreground/40">·</span>
                        <span className="flex items-center gap-0.5 text-amber-400">
                          <Cloud className="h-3 w-3 animate-pulse" />
                          <span className="text-[10px]">Archiving…</span>
                        </span>
                      </>
                    )}
                  </div>
                </div>

                {/* Delete button */}
                <button
                  type="button"
                  aria-label={`Delete ${job.mangaTitle}`}
                  onClick={(e) => handleDelete(e, job)}
                  className="p-1.5 rounded-md text-muted-foreground hover:text-rose-400 hover:bg-rose-500/10 transition flex-shrink-0 opacity-0 group-hover:opacity-100"
                >
                  {deletingId === job.id ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </button>
                <button
                  type="button"
                  aria-label={`View progress for ${job.mangaTitle}`}
                  onClick={(e) => { e.stopPropagation(); onSelectJob(job.id); }}
                  className="p-0.5 rounded-md text-muted-foreground hover:text-foreground transition shrink-0"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            );
          })}
        </div>
      )}

      <JobDetailModal job={detailJob} open={detailJob !== null} onClose={() => setDetailJob(null)} />
    </section>
  );
}
