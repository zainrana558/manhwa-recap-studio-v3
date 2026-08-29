"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSocket } from "@/lib/socket";
import { toast } from "@/hooks/use-toast";
import type { JobDetail, JobLogEntry, ChapterInfo, ServerEvent } from "@/types/pipeline";

interface UseJobProgressResult {
  job: JobDetail | null;
  logs: JobLogEntry[];
  connected: boolean;
}

/**
 * Subscribe to a job's live progress via socket.io.
 * Also bootstraps the initial state from the REST API so the UI
 * isn't blank while the socket connects.
 */
export function useJobProgress(jobId: string | null): UseJobProgressResult {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [logs, setLogs] = useState<JobLogEntry[]>([]);
  const [connected, setConnected] = useState(false);

  // Track previously notified status to avoid duplicate toasts
  const notifiedStatusRef = useRef<string | null>(null);
  const stuckTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingSinceRef = useRef<number | null>(null);

  const mergeChapter = useCallback((chapter: ChapterInfo) => {
    setJob((prev) =>
      prev
        ? {
            ...prev,
            chapters: prev.chapters.map((c) =>
              c.index === chapter.index ? chapter : c
            ),
          }
        : prev
    );
  }, []);

  // Reset state when jobId changes (React-recommended render-time pattern).
  const [prevJobId, setPrevJobId] = useState<string | null>(jobId);
  if (jobId !== prevJobId) {
    setPrevJobId(jobId);
    setJob(null);
    setLogs([]);
    setConnected(false);
  }

  // Reset refs when jobId changes (separate effect to satisfy react-hooks/refs rule).
  const prevJobIdRef = useRef<string | null>(jobId);
  useEffect(() => {
    if (jobId !== prevJobIdRef.current) {
      prevJobIdRef.current = jobId;
      notifiedStatusRef.current = null;
      pendingSinceRef.current = null;
      if (stuckTimerRef.current) {
        clearTimeout(stuckTimerRef.current);
        stuckTimerRef.current = null;
      }
    }
  }, [jobId]);

  // Toast notification on status transitions
  useEffect(() => {
    if (!job) return;
    const status = job.status;
    if (notifiedStatusRef.current === status) return;
    const prev = notifiedStatusRef.current;
    notifiedStatusRef.current = status;

    // Only show toasts on transitions, not on initial load.
    if (prev === null && status === "pending") return;

    if (status === "done") {
      toast({
        title: "✅ Recap complete!",
        description: `"${job.mangaTitle}" finished processing — ${job.totalChapters} chapters, ${job.totalImages} images.`,
      });
    } else if (status === "error") {
      toast({
        title: "Pipeline failed",
        description: `"${job.mangaTitle}" encountered an error: ${job.error || "Unknown error"}`,
        variant: "destructive",
      });
    } else if (status === "cancelled") {
      toast({
        title: "Job cancelled",
        description: `"${job.mangaTitle}" was cancelled.`,
      });
    }
  }, [job]);

  // Stuck detection: if job stays in "pending" for > 30s, notify user
  useEffect(() => {
    if (stuckTimerRef.current) {
      clearTimeout(stuckTimerRef.current);
      stuckTimerRef.current = null;
    }

    if (!job || job.status !== "pending") {
      pendingSinceRef.current = null;
      return;
    }

    if (!pendingSinceRef.current) {
      pendingSinceRef.current = Date.now();
    }

    const elapsed = Date.now() - pendingSinceRef.current;
    const remaining = Math.max(0, 30_000 - elapsed);

    if (remaining === 0) {
      toast({
        title: "⏳ Pipeline seems stuck",
        description: `"${job.mangaTitle}" is still waiting to start. The pipeline service may be offline — try clicking Retry.`,
      });
      // Don't re-notify — set a flag via extending the pendingSince to avoid repeat
      pendingSinceRef.current = Date.now() + 60_000; // Push ahead so the 30s check won't re-trigger
    } else {
      stuckTimerRef.current = setTimeout(() => {
        toast({
          title: "⏳ Pipeline seems stuck",
          description: `"${job.mangaTitle}" is still waiting to start. The pipeline service may be offline — try clicking Retry.`,
        });
        pendingSinceRef.current = Date.now() + 60_000;
      }, remaining);
    }

    return () => {
      if (stuckTimerRef.current) {
        clearTimeout(stuckTimerRef.current);
        stuckTimerRef.current = null;
      }
    };
  }, [job?.status, job?.mangaTitle]);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    let cancelled = false;

    // Bootstrap from REST first.
    fetch(`/api/jobs/${jobId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled || !data) return;
        setJob(data.job);
        setLogs(data.logs ?? []);
      })
      .catch(() => {});

    const socket = getSocket();

    const onConnect = () => {
      setConnected(true);
      socket.emit("subscribe", { jobId });
    };
    const onDisconnect = () => setConnected(false);
    const onSubscribed = () => {
      // Receiving the subscribed ack means the socket is live.
      setConnected(true);
    };
    const onStatus = (payload: ServerEvent) => {
      // Filter by jobId — the socket is a shared singleton, so events from
      // other jobs (if the user switches jobs) must not leak into this view.
      if (payload.type === "status" && payload.job && payload.job.id === jobId) setJob(payload.job);
    };
    const onLog = (payload: ServerEvent) => {
      if (payload.type === "log" && payload.log) {
        // JobLogEntry has a jobId field — filter to avoid cross-job leakage.
        if (payload.log.jobId && payload.log.jobId !== jobId) return;
        setLogs((prev) => {
          const next = [...prev, payload.log as JobLogEntry];
          // Cap at 500 lines to avoid memory bloat.
          return next.length > 500 ? next.slice(-500) : next;
        });
      }
    };
    const onProgress = (payload: ServerEvent) => {
      if (payload.type !== "progress") return;
      // Filter by jobId to prevent events from a different job updating this view.
      if (payload.jobId !== jobId) return;
      setJob((prev) =>
        prev
          ? {
              ...prev,
              progress: payload.progress ?? prev.progress,
              doneChapters: payload.doneChapters ?? prev.doneChapters,
              totalChapters: payload.totalChapters ?? prev.totalChapters,
              doneImages: payload.doneImages ?? prev.doneImages,
              totalImages: payload.totalImages ?? prev.totalImages,
              stage: payload.stage ?? prev.stage,
              message: payload.message ?? prev.message,
            }
          : prev
      );
    };
    const onChapter = (payload: ServerEvent) => {
      if (payload.type === "chapter" && payload.chapter) {
        mergeChapter(payload.chapter as ChapterInfo);
      }
    };
    const onDone = (payload: ServerEvent) => {
      if (payload.type !== "done") return;
      if (payload.jobId !== jobId) return;
      setJob((prev) =>
        prev
          ? {
              ...prev,
              status: "done",
              progress: 100,
              outputVideo: payload.outputVideo ?? prev.outputVideo,
              message: "Pipeline complete.",
            }
          : prev
      );
    };
    const onError = (payload: ServerEvent) => {
      if (payload.type !== "error") return;
      if (payload.jobId !== jobId) return;
      setJob((prev) =>
        prev
          ? { ...prev, status: "error", error: payload.error ?? "Unknown error" }
          : prev
      );
    };
    const onCancelled = (payload: ServerEvent) => {
      if (payload.type !== "cancelled") return;
      if (payload.jobId !== jobId) return;
      setJob((prev) => (prev ? { ...prev, status: "cancelled" } : prev));
    };

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("subscribed", onSubscribed);
    socket.on("status", onStatus);
    socket.on("log", onLog);
    socket.on("progress", onProgress);
    socket.on("chapter", onChapter);
    socket.on("done", onDone);
    socket.on("error", onError);
    socket.on("cancelled", onCancelled);

    if (socket.connected) {
      // Already connected (persistent socket) — just subscribe.
      // connected state will be set by the `subscribed` ack.
      socket.emit("subscribe", { jobId });
    }

    // REST POLLING FALLBACK: when the socket is disconnected, poll the REST
    // API every 5 seconds so the UI still shows job progress. This handles
    // the case where the pipeline-service restarts and the socket is briefly
    // down — the user still sees updates instead of a stale "reconnecting…"
    // state forever.
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    const startPolling = () => {
      if (pollTimer) return;
      pollTimer = setInterval(async () => {
        if (cancelled) return;
        try {
          const r = await fetch(`/api/jobs/${jobId}`);
          if (!r.ok) return;
          const data = await r.json();
          if (cancelled || !data) return;
          // A successful poll means the app is still getting live data from
          // the server even though the socket transport is down — surface
          // that as "connected" too, otherwise the UI shows a permanent
          // "Offline" badge while the job is visibly still updating every
          // 5s, which is exactly backwards for a tool meant to keep working
          // unattended (the socket dying doesn't mean the job died).
          setConnected(true);
          // Only update if the REST data is newer (different progress/message)
          setJob((prev) => {
            if (!prev) return data.job;
            // Prefer the REST data if progress changed or status changed
            if (data.job.progress !== prev.progress || data.job.status !== prev.status) {
              return data.job;
            }
            return prev;
          });
          // Merge logs (add any new ones not already in state)
          if (data.logs && data.logs.length > 0) {
            setLogs((prev) => {
              const prevIds = new Set(prev.map((l) => l.id));
              const newLogs = data.logs.filter((l: JobLogEntry) => !prevIds.has(l.id));
              if (newLogs.length === 0) return prev;
              const next = [...prev, ...newLogs];
              return next.length > 500 ? next.slice(-500) : next;
            });
          }
        } catch {
          // ignore — will retry next interval
        }
      }, 5000);
    };
    const stopPolling = () => {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };

    // Start polling when disconnected, stop when connected.
    const onConnectPolling = () => stopPolling();
    const onDisconnectPolling = () => startPolling();
    socket.on("connect", onConnectPolling);
    socket.on("disconnect", onDisconnectPolling);

    // If not connected on mount, start polling immediately.
    if (!socket.connected) {
      startPolling();
    }

    return () => {
      cancelled = true;
      stopPolling();
      socket.emit("unsubscribe", { jobId });
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("subscribed", onSubscribed);
      socket.off("status", onStatus);
      socket.off("log", onLog);
      socket.off("progress", onProgress);
      socket.off("chapter", onChapter);
      socket.off("done", onDone);
      socket.off("error", onError);
      socket.off("cancelled", onCancelled);
      socket.off("connect", onConnectPolling);
      socket.off("disconnect", onDisconnectPolling);
    };
  }, [jobId, mergeChapter]);

  return { job, logs, connected };
}
