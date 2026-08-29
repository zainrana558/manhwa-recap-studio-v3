"use client";

import { useEffect, useState } from "react";
import { Film, BookOpen, ImageIcon, CheckCircle2, Cloud, Sparkles, Clock, Play, Zap, Users, Database, Search } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

interface Stats {
  totalJobs: number;
  completedJobs: number;
  totalChapters: number;
  totalImages: number;
  archivedJobs?: number;
}

interface PlatformStat {
  icon: typeof Film;
  label: string;
  value: string;
  color: string;
  barColor: string;
}

const PLATFORM_STATS: PlatformStat[] = [
  { icon: Search, label: "Sources", value: "6", color: "text-amber-400", barColor: "bg-amber-400" },
  { icon: Zap, label: "TTS Voices", value: "55+", color: "text-emerald-400", barColor: "bg-emerald-400" },
  { icon: Users, label: "Languages", value: "9+", color: "text-sky-400", barColor: "bg-sky-400" },
  { icon: Database, label: "API Providers", value: "3", color: "text-fuchsia-400", barColor: "bg-fuchsia-400" },
];

function StatCard({ item }: { item: PlatformStat }) {
  const Icon = item.icon;
  const miniBarHeights = [40, 70, 100, 60, 85];
  return (
    <div className="relative flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-card/60 border border-border overflow-hidden group hover:border-primary/20 hover:bg-card/90 transition-all duration-300 hover-glow-sm">
      <div
        className={`absolute inset-y-0 left-0 ${item.barColor} opacity-[0.06] transition-all duration-1000 group-hover:opacity-[0.12]`}
        style={{ width: "100%" }}
      />
      <div className="relative flex items-center gap-2.5">
        <div className="p-1.5 rounded-lg bg-primary/8 group-hover:bg-primary/15 transition-colors">
          <Icon className={`h-3.5 w-3.5 ${item.color}`} />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-bold tabular-nums leading-none">{item.value}</span>
          <span className="text-[10px] text-muted-foreground/70 leading-tight mt-0.5">{item.label}</span>
        </div>
        {/* Decorative mini-bars */}
        <div className="flex items-end gap-[2px] ml-1 h-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          {miniBarHeights.map((h, i) => (
            <div
              key={i}
              className={`w-[3px] rounded-full ${item.barColor}`}
              style={{ height: `${h}%`, opacity: 0.4 }}
            />
          ))}
        </div>
      </div>
      {/* Animated gradient line at bottom */}
      <div
        className="absolute bottom-0 left-0 h-[1px] animate-draw-line"
        style={{
          background: `linear-gradient(90deg, transparent, var(--primary), transparent)`,
          width: "100%",
        }}
      />
    </div>
  );
}

export function StatsBar() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && !data.error) setStats(data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4 py-3">
        {PLATFORM_STATS.map((s) => (
          <div key={s.label} className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl border border-border bg-card/40">
            <Skeleton className="h-3.5 w-3.5 rounded-lg" />
            <div className="space-y-1">
              <Skeleton className="h-4 w-8 rounded" />
              <Skeleton className="h-2.5 w-14 rounded" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Platform stats when no jobs
  if (!stats || stats.totalJobs === 0) {
    return (
      <div className="flex flex-col items-center gap-3 animate-fade-in-up">
        <div className="flex items-center gap-2 px-4 py-2 rounded-xl border border-primary/20 bg-primary/5">
          <Sparkles className="h-3.5 w-3.5 text-primary" />
          <p className="text-xs text-primary font-medium">Platform Ready</p>
          <span className="text-muted-foreground/30">·</span>
          <span className="text-xs text-muted-foreground">Search a manhwa to create your first video</span>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2.5 sm:gap-3">
          {PLATFORM_STATS.map((item) => (
            <StatCard key={item.label} item={item} />
          ))}
        </div>
      </div>
    );
  }

  const jobItems: PlatformStat[] = [
    { icon: Film, label: "Videos", value: String(stats.completedJobs), color: "text-emerald-400", barColor: "bg-emerald-400" },
    { icon: Play, label: "Jobs", value: String(stats.totalJobs), color: "text-sky-400", barColor: "bg-sky-400" },
    { icon: ImageIcon, label: "Images", value: stats.totalImages.toLocaleString(), color: "text-amber-400", barColor: "bg-amber-400" },
    { icon: Clock, label: "In Progress", value: String(stats.totalJobs - stats.completedJobs), color: "text-fuchsia-400", barColor: "bg-fuchsia-400" },
  ];

  if (stats.archivedJobs && stats.archivedJobs > 0) {
    jobItems.push({ icon: Cloud, label: "Archived", value: String(stats.archivedJobs), color: "text-sky-400", barColor: "bg-sky-400" });
  }

  return (
    <div className="flex flex-col items-center gap-3 animate-fade-in-up">
      <div className="flex items-center gap-2 px-4 py-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5">
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
        <p className="text-xs text-emerald-400 font-medium">
          {stats.completedJobs} video{stats.completedJobs !== 1 ? "s" : ""} created · {stats.totalImages.toLocaleString()} images processed
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2.5 sm:gap-3">
        {jobItems.map((item) => (
          <StatCard key={item.label} item={item} />
        ))}
      </div>
    </div>
  );
}