"use client";

import { useCallback } from "react";
import {
  Loader2,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Wifi,
  WifiOff,
  Search as SearchIcon,
  ScanLine,
  FileText,
  Scissors,
  Clapperboard,
  RotateCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ChapterGrid } from "./chapter-grid";
import { LogStream } from "./log-stream";
import { VideoResult } from "./video-result";
import type { JobDetail, JobLogEntry } from "@/types/pipeline";
import { cn } from "@/lib/utils";

interface JobProgressProps {
  job: JobDetail | null;
  logs: JobLogEntry[];
  connected: boolean;
  onCancel: () => void;
  onNewJob: () => void;
}

interface StageInfo {
  key: string;
  label: string;
  icon: typeof SearchIcon;
}

const PIPELINE_STAGES: StageInfo[] = [
  { key: "search", label: "Search", icon: SearchIcon },
  { key: "scrape", label: "Download", icon: ScanLine },
  { key: "transcribe", label: "OCR", icon: FileText },
  { key: "slice", label: "Slice", icon: Scissors },
  { key: "render", label: "Render", icon: Clapperboard },
  { key: "done", label: "Done", icon: CheckCircle2 },
];

function getActiveStageIndex(job: JobDetail | null): number {
  if (!job) return -1;
  if (job.status === "done") return PIPELINE_STAGES.length - 1;
  if (job.status === "error" || job.status === "cancelled") return -1;
  const statusMap: Record<string, number> = {
    pending: 0,
    scraping: 1,
    transcribing: 2,
    rendering: 4,
  };
  const stageMap: Record<string, number> = {
    search: 0,
    scrape: 1,
    transcribe: 2,
    translate: 2,
    slice: 3,
    narrate: 4,
    tts: 4,
    captions: 4,
    render: 4,
    merge: 4,
    bgm: 4,
  };
  const stageIdx = job.stage ? (stageMap[job.stage] ?? -1) : -1;
  const statusIdx = job.status ? (statusMap[job.status] ?? -1) : -1;
  return Math.max(stageIdx, statusIdx);
}

export function JobProgress({ job, logs, connected, onCancel, onNewJob }: JobProgressProps) {
  const handleCancel = useCallback(() => {
    if (!job) return;
    if (confirm("Cancel this job? The pipeline will stop after the current step.")) {
      fetch(`/api/jobs/${job.id}`, { method: "DELETE" });
    }
  }, [job]);

  const handleRetry = useCallback(() => {
    if (!job) return;
    fetch(`/api/jobs/${job.id}`, { method: "POST" }).then((r) => {
      if (r.ok) {
      }
    });
  }, [job]);

  if (!job) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const activeStageIdx = getActiveStageIndex(job);
  const isDone = job.status === "done";
  const isError = job.status === "error";
  const isCancelled = job.status === "cancelled";
  const isPending = job.status === "pending";
  const isRunning = !isDone && !isError && !isCancelled;
  const isProcessing = isRunning && !isPending;

  return (
    <section className="max-w-5xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center gap-4 p-6 rounded-xl border border-border bg-card">
        <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-lg overflow-hidden bg-muted flex-shrink-0 border border-border">
          {job.coverUrl ? (
            <img src={job.coverUrl} alt={job.mangaTitle} className="w-full h-full object-cover" />
          ) : null}
        </div>
        <div className="flex-1 min-w-0 space-y-1">
          <h2 className="text-xl font-bold truncate">{job.mangaTitle}</h2>
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <Badge
              variant="outline"
              className={cn(
                "capitalize",
                isProcessing && "pulse-glow border-primary/40"
              )}
            >
              {job.status}
            </Badge>
            <span>·</span>
            <span>{job.totalChapters} chapters</span>
            <span>·</span>
            <span>{job.doneImages}/{job.totalImages} images</span>
            <span>·</span>
            <span className="flex items-center gap-1">
              {connected ? (
                <>
                  <Wifi className="h-3.5 w-3.5 text-emerald-400" />
                  live
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5 text-sky-400" />
                  syncing…
                </>
              )}
            </span>
          </div>
          {job.message && (
            <p className="text-sm text-muted-foreground truncate">{job.message}</p>
          )}
        </div>
        <div className="flex gap-2">
          {isRunning && !isPending && (
            <Button variant="destructive" size="sm" onClick={handleCancel}>
              <XCircle className="h-4 w-4 mr-1.5" />
              Cancel
            </Button>
          )}
          {isPending && (
            <Button variant="default" size="sm" onClick={handleRetry}>
              <RotateCw className="h-4 w-4 mr-1.5" />
              Retry
            </Button>
          )}
          {isError && (
            <Button variant="default" size="sm" onClick={handleRetry}>
              <RotateCw className="h-4 w-4 mr-1.5" />
              Retry
            </Button>
          )}
          {(isDone || isError || isCancelled) && (
            <Button variant="outline" size="sm" onClick={onNewJob}>
              <SearchIcon className="h-4 w-4 mr-1.5" />
              New search
            </Button>
          )}
        </div>
      </div>

      <div className="p-6 rounded-xl border border-border bg-card space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">
            {isDone ? "Complete" : isError ? "Failed" : isCancelled ? "Cancelled" : isPending ? "Waiting for pipeline…" : "Processing…"}
          </span>
          <span className="text-2xl font-bold tabular-nums text-primary">{job.progress}%</span>
        </div>
        <div className="relative">
          <Progress
            value={job.progress}
            className={`h-2.5 transition-all ${
              isError ? "[&>div]:bg-rose-500" :
              isDone ? "[&>div]:bg-emerald-500" :
              isPending ? "[&>div]:bg-amber-500" :
              "[&>div]:bg-primary"
            }`}
          />
          {isProcessing && (
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                background: "linear-gradient(90deg, transparent 0%, oklch(0.78 0.17 65 / 0.15) 50%, transparent 100%)",
                backgroundSize: "200% 100%",
                animation: "shimmer 1.5s ease-in-out infinite",
              }}
            />
          )}
        </div>

        <div className="flex flex-wrap items-center gap-1.5 pt-2">
          {PIPELINE_STAGES.map((stage, idx) => {
            const isDone_ = idx < activeStageIdx;
            const isActive = idx === activeStageIdx && isRunning;
            const isFuture = idx > activeStageIdx;
            const Icon = stage.icon;
            return (
              <div
                key={stage.key}
                className={cn(
                  "flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-all hover-glow-sm",
                  isDone_ && "text-emerald-400",
                  isActive && "text-primary bg-primary/10 ring-1 ring-primary/30",
                  isFuture && "text-muted-foreground/50"
                )}
              >
                {isDone_ ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : isActive ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Icon className="h-3.5 w-3.5" />
                )}
                <span className="hidden sm:inline">{stage.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {isError && job.error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-sm font-medium text-rose-300">Pipeline failed</p>
            <p className="text-sm text-rose-400/80 font-mono break-all">{job.error}</p>
            <p className="text-xs text-muted-foreground mt-2">
              Partial progress is saved. You can start a new search or retry with fewer chapters.
            </p>
          </div>
        </div>
      )}

      {isPending && (
        <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 flex items-start gap-3">
          <Loader2 className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5 animate-spin" />
          <div className="space-y-1">
            <p className="text-sm font-medium text-amber-300">Waiting for pipeline to start</p>
            <p className="text-xs text-amber-400/80">
              The pipeline service may be starting up or busy. If this takes more than a few seconds, click <strong>Retry</strong> to re-queue the job.
            </p>
          </div>
        </div>
      )}

      {isDone && <VideoResult job={job} />}

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="p-5 rounded-xl border border-border bg-card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Chapters</h3>
            <span className="text-xs text-muted-foreground">
              {job.chapters.filter((c) => c.rendered || c.transcribed || c.status === "scraped").length}/{job.chapters.length} processed
            </span>
          </div>
          <ChapterGrid chapters={job.chapters} jobId={job.id} />
        </div>

        <div className="glass-card rounded-lg overflow-hidden">
          <LogStream logs={logs} />
        </div>
      </div>
    </section>
  );
}
