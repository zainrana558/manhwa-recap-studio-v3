"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import { Loader2, Check, X, ImageOff, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface Frame {
  id: string;      // "chap_001/frame_00003.jpg"
  url: string;
  chapter: number;
  name: string;
  text: string;
  visual: string;
  status: string;
}

/**
 * Manual panel review. Shown while a job is `awaiting_review` (opt-in per job).
 * Every sliced panel is kept by default; the user clicks to drop covers, ads
 * and junk, then renders. Excluded ids are POSTed to /api/jobs/:id/review and
 * the pipeline resumes straight to render.
 */
export function PanelReview({
  jobId,
  onSubmitted,
}: {
  jobId: string;
  onSubmitted: () => void;
}) {
  const { toast } = useToast();
  const [frames, setFrames] = useState<Frame[] | null>(null);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancel = false;
    (async () => {
      try {
        const r = await fetch(`/api/frames/${jobId}?all=1`, { cache: "no-store" });
        const d = (await r.json()) as { frames: Frame[] };
        if (!cancel) setFrames(d.frames || []);
      } catch {
        if (!cancel) setFrames([]);
      }
    })();
    return () => {
      cancel = true;
    };
  }, [jobId]);

  const byChapter = useMemo(() => {
    const m = new Map<number, Frame[]>();
    for (const f of frames || []) {
      if (!m.has(f.chapter)) m.set(f.chapter, []);
      m.get(f.chapter)!.push(f);
    }
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }, [frames]);

  const toggle = useCallback((id: string) => {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const submit = useCallback(async () => {
    setSubmitting(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/review`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ excluded: [...excluded] }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.error || `Failed (${res.status})`);
      }
      toast({
        title: "Rendering started",
        description:
          excluded.size > 0
            ? `Dropped ${excluded.size} panel${excluded.size === 1 ? "" : "s"} — rendering the rest.`
            : "Keeping all panels — rendering.",
      });
      onSubmitted();
    } catch (e) {
      toast({
        title: "Couldn't start render",
        description: e instanceof Error ? e.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  }, [jobId, excluded, onSubmitted, toast]);

  if (frames === null) {
    return (
      <div className="p-8 rounded-xl border border-border bg-card flex items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading panels…
      </div>
    );
  }
  if (frames.length === 0) {
    return (
      <div className="p-8 rounded-xl border border-border bg-card flex flex-col items-center gap-2 text-sm text-muted-foreground">
        <ImageOff className="h-5 w-5" />
        No sliced panels found for this job.
        <Button size="sm" className="mt-2" onClick={submit} disabled={submitting}>
          Render anyway
        </Button>
      </div>
    );
  }

  const kept = frames.length - excluded.size;

  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 sm:p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-amber-300">Review panels before rendering</h3>
          <p className="text-xs text-muted-foreground">
            Click any panel to drop it (covers, ads, credits, blanks). {kept} of{" "}
            {frames.length} panels will be rendered.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExcluded(new Set())}
            disabled={excluded.size === 0 || submitting}
          >
            Reset
          </Button>
          <Button size="sm" onClick={submit} disabled={submitting}>
            {submitting ? (
              <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-1.5" />
            )}
            Render {kept} panel{kept === 1 ? "" : "s"}
          </Button>
        </div>
      </div>

      <div className="max-h-[62vh] overflow-y-auto pr-1 space-y-5">
        {byChapter.map(([ch, list]) => (
          <div key={ch} className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground sticky top-0 bg-amber-500/5 backdrop-blur py-1 z-10">
              Chapter {ch} — {list.filter((f) => !excluded.has(f.id)).length}/{list.length} kept
            </div>
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {list.map((f) => {
                const dropped = excluded.has(f.id);
                return (
                  <button
                    key={f.id}
                    onClick={() => toggle(f.id)}
                    title={f.visual || f.text || f.name}
                    className={`group relative aspect-[3/4] rounded-lg overflow-hidden border transition-all ${
                      dropped
                        ? "border-rose-500/60 opacity-40 grayscale"
                        : "border-border hover:border-primary/60"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={f.url}
                      alt={f.name}
                      loading="lazy"
                      className="w-full h-full object-cover bg-muted"
                    />
                    <span
                      className={`absolute top-1 right-1 h-5 w-5 rounded-full flex items-center justify-center ${
                        dropped ? "bg-rose-500 text-white" : "bg-black/50 text-white/70 group-hover:bg-primary group-hover:text-primary-foreground"
                      }`}
                    >
                      {dropped ? <X className="h-3 w-3" /> : <Check className="h-3 w-3" />}
                    </span>
                    {(f.visual || f.text) && (
                      <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-1 text-[9px] leading-tight text-white/90 line-clamp-2 text-left">
                        {f.visual || `“${f.text}”`}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
