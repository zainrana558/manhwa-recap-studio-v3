"use client";

import { useEffect, useRef, useState } from "react";
import { Eye, Loader2 } from "lucide-react";
import type { JobStatus } from "@/types/pipeline";

interface Frame {
  url: string;
  chapter: number;
  name: string;
  text: string;
  visual: string;
  status: string;
}

const ACTIVE: JobStatus[] = [
  "pending",
  "scraping",
  "transcribing",
  "translating",
  "rendering",
  "merging",
] as never[];

/**
 * Live filmstrip of the panels the pipeline is working through right now.
 * Polls /api/frames/{id} while the job is active and shows the most recent
 * frames with their transcription + visual caption underneath.
 */
export function LiveFrames({
  jobId,
  status,
}: {
  jobId: string;
  status: JobStatus;
}) {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [total, setTotal] = useState(0);
  const stripRef = useRef<HTMLDivElement | null>(null);
  const active = ACTIVE.includes(status);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const r = await fetch(`/api/frames/${jobId}?limit=14`, { cache: "no-store" });
        if (!r.ok) return;
        const d = (await r.json()) as { frames: Frame[]; total: number };
        if (stop) return;
        setFrames(d.frames || []);
        setTotal(d.total || 0);
      } catch {
        /* transient */
      }
    };
    tick();
    // Poll while active; one final poll shortly after it stops so the last
    // frames land, then leave the strip as-is.
    const iv = setInterval(tick, active ? 2500 : 8000);
    return () => {
      stop = true;
      clearInterval(iv);
    };
  }, [jobId, active]);

  // keep the newest frame in view
  useEffect(() => {
    const el = stripRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [frames.length]);

  if (!active && frames.length === 0) return null;

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Eye className="h-4 w-4 text-primary" />
          Live panel view
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">
          {active && <Loader2 className="inline h-3 w-3 animate-spin mr-1" />}
          {total > 0 ? `${total} panels sliced` : "waiting for panels…"}
        </span>
      </div>

      {frames.length === 0 ? (
        <p className="text-xs text-muted-foreground py-6 text-center">
          Panels appear here as the pipeline slices and reads them.
        </p>
      ) : (
        <div
          ref={stripRef}
          className="flex gap-3 overflow-x-auto pb-2 scroll-smooth"
        >
          {frames.map((f) => (
            <figure
              key={`${f.chapter}-${f.name}`}
              className="shrink-0 w-40 space-y-1.5"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={f.url}
                alt={f.name}
                loading="lazy"
                className="w-40 h-40 object-cover rounded-lg border border-border bg-muted"
              />
              <figcaption className="text-[10px] leading-tight text-muted-foreground line-clamp-3">
                <span className="text-foreground/70">ch {f.chapter}</span>
                {f.visual && (
                  <span className="block text-primary/90">👁 {f.visual}</span>
                )}
                {f.text ? (
                  <span className="block">“{f.text}”</span>
                ) : (
                  <span className="block italic opacity-60">no dialogue</span>
                )}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
