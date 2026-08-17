"use client";

import { useEffect, useRef, useState } from "react";
import { Terminal, ArrowDown, Circle } from "lucide-react";
import type { JobLogEntry } from "@/types/pipeline";
import { cn } from "@/lib/utils";

interface LogStreamProps {
  logs: JobLogEntry[];
}

const levelConfig: Record<string, { color: string; dot: string }> = {
  info: { color: "text-zinc-300", dot: "bg-zinc-500" },
  success: { color: "text-emerald-400", dot: "bg-emerald-500" },
  warn: { color: "text-amber-400", dot: "bg-amber-500" },
  error: { color: "text-rose-400", dot: "bg-rose-500" },
};

const stageConfig: Record<string, { color: string; bg: string; label: string }> = {
  search: { color: "text-sky-400", bg: "bg-sky-500/10", label: "SEARCH" },
  scrape: { color: "text-amber-400", bg: "bg-amber-500/10", label: "SCRAPE" },
  slice: { color: "text-cyan-400", bg: "bg-cyan-500/10", label: "SLICE" },
  transcribe: { color: "text-orange-400", bg: "bg-orange-500/10", label: "OCR" },
  translate: { color: "text-purple-400", bg: "bg-purple-500/10", label: "TRANSLATE" },
  render: { color: "text-emerald-400", bg: "bg-emerald-500/10", label: "RENDER" },
  merge: { color: "text-teal-400", bg: "bg-teal-500/10", label: "MERGE" },
  bgm: { color: "text-pink-400", bg: "bg-pink-500/10", label: "FINALIZE" },
  done: { color: "text-emerald-300", bg: "bg-emerald-500/10", label: "DONE" },
  cancel: { color: "text-rose-400", bg: "bg-rose-500/10", label: "CANCEL" },
};

export function LogStream({ logs }: LogStreamProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const [showJumpDown, setShowJumpDown] = useState(false);
  const [isAutoScroll, setIsAutoScroll] = useState(true);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (autoScrollRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    autoScrollRef.current = atBottom;
    setIsAutoScroll(atBottom);
    setShowJumpDown(!atBottom && logs.length > 10);
  };

  const jumpToBottom = () => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    autoScrollRef.current = true;
    setIsAutoScroll(true);
    setShowJumpDown(false);
  };

  return (
    <div className="overflow-hidden relative">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-muted/20">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground">Live log</span>
        </div>
        <div className="flex items-center gap-2 ml-auto">
          {isAutoScroll && logs.length > 0 && (
            <div className="flex items-center gap-1">
              <Circle className="h-2 w-2 fill-emerald-500 text-emerald-500 animate-pulse" />
              <span className="text-[10px] text-emerald-400/70">live</span>
            </div>
          )}
          <span className="text-xs text-muted-foreground/70 tabular-nums">{logs.length} lines</span>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="h-64 overflow-y-auto scrollbar-thin p-3 font-mono text-xs space-y-1"
      >
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground/40">
            <Terminal className="h-6 w-6" />
            <p className="text-xs">Waiting for logs…</p>
          </div>
        ) : (
          logs.map((log) => {
            const lvl = levelConfig[log.level] ?? levelConfig.info;
            const stg = log.stage ? (stageConfig[log.stage] ?? null) : null;
            return (
              <div key={log.id} className="flex gap-2 leading-relaxed group hover:bg-muted/20 rounded px-1 -mx-1 transition-colors">
                <span className="text-muted-foreground/40 flex-shrink-0 tabular-nums">
                  {new Date(log.createdAt).toLocaleTimeString("en-US", { hour12: false })}
                </span>
                <span className={cn("flex-shrink-0 mt-1.5 h-1.5 w-1.5 rounded-full", lvl.dot)} />
                {stg && (
                  <span className={cn("flex-shrink-0 px-1 rounded text-[9px] font-bold tracking-wide", stg.bg, stg.color)}>
                    {stg.label}
                  </span>
                )}
                <span className={cn("flex-1 break-words", lvl.color)}>
                  {log.message}
                </span>
              </div>
            );
          })
        )}
      </div>

      {showJumpDown && (
        <button
          onClick={jumpToBottom}
          className="absolute bottom-3 right-3 flex items-center gap-1 px-2 py-1 rounded-lg bg-primary/20 border border-primary/30 text-primary text-[10px] font-medium hover:bg-primary/30 transition animate-fade-in-up"
        >
          <ArrowDown className="h-3 w-3" />
          Jump to bottom
        </button>
      )}
    </div>
  );
}
