"use client";

import { useState, type JSX } from "react";
import {
  ChevronDown,
  HelpCircle,
  Workflow,
  VolumeX,
  Zap,
  Mic,
  HardDrive,
  AlertTriangle,
  CloudUpload,
  Layers,
  Gift,
  Globe,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSectionObserver } from "@/hooks/use-section-observer";

interface FAQItem {
  question: string;
  answer: string;
  icon: LucideIcon;
  iconClass: string;
  borderStyle: string;
  gradientFrom: string;
  gradientTo: string;
}

const FAQ_ITEMS: FAQItem[] = [
  {
    question: "How does the manhwa recap pipeline work?",
    answer:
      "The pipeline has 5 stages: (1) Search across 6 sources (MangaHere, FanFox, Webtoons, AsuraScans, MAL, AniList). " +
      "(2) Download every chapter image. (3) Transcribe speech bubble text using VLM (vision language model). " +
      "(4) Generate narration with edge-tts text-to-speech. (5) Render the final MP4 video with ffmpeg — " +
      "panels synced to narration audio. The whole process takes ~6 minutes per chapter.",
    icon: Workflow,
    iconClass: "text-sky-400",
    borderStyle: "border-l-sky-400",
    gradientFrom: "oklch(0.7 0.15 200)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "Why is there no voice in my video?",
    answer:
      "This happens when all VLM providers are rate-limited (429 errors). The pipeline transcribes panel text " +
      "using z-ai, Groq, and Gemini — if all three are rate-limited simultaneously, panels get empty text and " +
      "no narration is generated. The system automatically retries with backoff. Wait 30-60 minutes for the " +
      "rate limit to reset, then run the job again. The **VLM cache** also reuses successful transcriptions from " +
      "previous runs, so re-running a failed job is faster.",
    icon: VolumeX,
    iconClass: "text-rose-400",
    borderStyle: "border-l-rose-400",
    gradientFrom: "oklch(0.65 0.2 25)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "How do I speed up transcription?",
    answer:
      "VLM transcription is the slowest stage. To speed it up: (1) Set `GROQ_API_KEY` — Groq's LPU hardware is " +
      "3-5x faster than other providers. Get a free key at console.groq.com/keys. (2) Set `GEMINI_API_KEY` as a " +
      "second provider. (3) Adjust `VLM_CONCURRENCY` in .env (default 2, max 4) — higher = faster but more 429 errors. " +
      "(4) The VLM cache reuses transcriptions from previous runs of the same manga, so re-runs are instant.",
    icon: Zap,
    iconClass: "text-amber-400",
    borderStyle: "border-l-amber-400",
    gradientFrom: "oklch(0.78 0.17 65)",
    gradientTo: "oklch(0.65 0.22 50)",
  },
  {
    question: "Can I use a different narration voice?",
    answer:
      "Yes! There are 55 voices available across 8 English accents (US, UK, AU, CA, IE, IN, ZA) plus 8 other " +
      "languages (Japanese, Korean, Spanish, French, German, Portuguese, Hindi, Chinese). Click the speaker icon " +
      "next to the voice dropdown to preview any voice before starting the pipeline.",
    icon: Mic,
    iconClass: "text-emerald-400",
    borderStyle: "border-l-emerald-400",
    gradientFrom: "oklch(0.7 0.15 160)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "Where are my videos stored?",
    answer:
      "By default, videos are stored locally in the `data/jobs/{jobId}/output/` directory. If you configure Mega " +
      "cloud archive (`MEGA_EMAIL` + `MEGA_PASSWORD` in .env), finished videos automatically upload to Mega (20 GB free) " +
      "and the local file is deleted to free disk space. When you watch a video, it's transparently restored from " +
      "Mega to a 1-hour temp cache and streamed with seek support.",
    icon: HardDrive,
    iconClass: "text-fuchsia-400",
    borderStyle: "border-l-fuchsia-400",
    gradientFrom: "oklch(0.65 0.2 280)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "Why did my job fail or get stuck?",
    answer:
      "Common causes: (1) The pipeline-service crashed or restarted mid-job — the service auto-requeues stuck jobs " +
      "on restart, so just refresh the page. (2) Network issues during scraping — try again or pick a different manga. " +
      "(3) Disk space full — check available space. (4) All VLM providers rate-limited — wait 30-60 min and retry. " +
      "You can click the **Retry** button on any failed/stuck job to restart it without re-entering config.",
    icon: AlertTriangle,
    iconClass: "text-orange-400",
    borderStyle: "border-l-orange-400",
    gradientFrom: "oklch(0.72 0.18 45)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "What does 'Archive to cloud' do?",
    answer:
      "The 'Archive to cloud' button manually uploads a completed video to Mega cloud storage and deletes the local " +
      "file. This frees disk space while keeping the video accessible — when you click Download or Play, the video is " +
      "automatically fetched from Mega. **Auto-archive** is enabled by default (set `AUTO_ARCHIVE=false` in .env to disable).",
    icon: CloudUpload,
    iconClass: "text-sky-400",
    borderStyle: "border-l-sky-400",
    gradientFrom: "oklch(0.7 0.15 200)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "Can I process multiple chapters at once?",
    answer:
      "Yes! In the Pipeline Configuration, set 'Chapters to process' to the number of chapters you want (or 0 for all). " +
      "Each chapter is processed sequentially — scraping, transcribing, and rendering one at a time. More chapters = " +
      "longer processing time (~6 min per chapter). The progress bar shows overall completion across all chapters.",
    icon: Layers,
    iconClass: "text-teal-400",
    borderStyle: "border-l-teal-400",
    gradientFrom: "oklch(0.7 0.15 195)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "Is this free to use?",
    answer:
      "Yes, completely free. The pipeline uses: z-ai VLM (free tier), edge-tts (free, unlimited), ffmpeg (open source), " +
      "and YOLO panel detection (open source, runs locally). Optional free enhancements: Groq for faster VLM + narration " +
      "rewriting, Gemini as a second VLM provider, Mega for 20 GB cloud storage. No paid resources are required.",
    icon: Gift,
    iconClass: "text-emerald-400",
    borderStyle: "border-l-emerald-400",
    gradientFrom: "oklch(0.7 0.15 160)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
  {
    question: "How do I deploy this online?",
    answer:
      "See `DEPLOYMENT.md` for the full guide. The app can run as a single Docker container (Dockerfile included) on any " +
      "free Docker host. For Vercel deployment, set `PIPELINE_SERVICE_URL` to point to your laptop running the pipeline-service " +
      "(exposed via Cloudflare Tunnel). The Next.js frontend goes on Vercel (free), the database on Turso (free 9 GB), " +
      "and videos on Mega (free 20 GB) or local storage.",
    icon: Globe,
    iconClass: "text-violet-400",
    borderStyle: "border-l-violet-400",
    gradientFrom: "oklch(0.65 0.2 280)",
    gradientTo: "oklch(0.78 0.17 65)",
  },
];

/** Parse basic markdown: **bold** and `code` */
function parseMarkdown(text: string): (string | JSX.Element)[] {
  const parts: (string | JSX.Element)[] = [];
  const regex = /(\*\*(.+?)\*\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(<strong key={match.index}>{match[2]}</strong>);
    } else if (match[3]) {
      parts.push(
        <code
          key={match.index}
          className="px-1 py-0.5 rounded bg-muted text-[0.85em] font-mono"
        >
          {match[3]}
        </code>
      );
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

export function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);
  const { ref, isVisible } = useSectionObserver(0.05);

  return (
    <section ref={ref} className={`max-w-3xl mx-auto space-y-6 transition-all duration-700 ${isVisible ? "animate-section-in" : "opacity-0"}`}>
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 mb-3">
          <HelpCircle className="h-3.5 w-3.5 text-primary" />
          <span className="text-[11px] font-semibold uppercase tracking-widest text-primary">
            FAQ
          </span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Frequently Asked Questions</h2>
        <p className="text-sm text-muted-foreground mt-1.5 max-w-lg mx-auto leading-relaxed">
          Everything you need to know about the pipeline
        </p>
      </div>

      <div className="space-y-2">
        {FAQ_ITEMS.map((item, i) => {
          const isOpen = openIndex === i;
          const Icon = item.icon;
          return (
            <div
              key={i}
              className={cn(
                "rounded-xl border overflow-hidden transition-all duration-500 ease-out relative",
                isOpen
                  ? `border-l-2 ${item.borderStyle} shadow-lg shadow-primary/5 bg-card/90`
                  : "border-border hover:bg-card/80 hover:border-primary/20 bg-card/60"
              )}
            >
              {/* Gradient left border overlay when opened */}
              {isOpen && (
                <div
                  className="absolute left-0 top-0 bottom-0 w-1 pointer-events-none"
                  style={{
                    background: `linear-gradient(to bottom, ${item.gradientFrom}, ${item.gradientTo})`,
                    opacity: 0.6,
                  }}
                />
              )}
              {/* Subtle background gradient when expanded */}
              {isOpen && (
                <div
                  className="absolute inset-0 pointer-events-none"
                  style={{
                    background: `linear-gradient(135deg, ${item.gradientFrom} / 0.03, ${item.gradientTo} / 0.02)`,
                  }}
                />
              )}
              <button
                onClick={() => setOpenIndex(isOpen ? null : i)}
                className="w-full flex items-center gap-3 p-4 text-left hover:bg-muted/30 transition-colors relative z-10"
                aria-expanded={isOpen}
              >
                {/* Number badge */}
                <span className={cn(
                  "text-[10px] font-mono font-bold w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 transition-all duration-500",
                  isOpen
                    ? "bg-primary/15 text-primary"
                    : "bg-muted/50 text-muted-foreground/50"
                )}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className={cn(
                  "p-1.5 rounded-lg transition-all duration-500 flex-shrink-0",
                  isOpen ? "bg-primary/15" : "bg-muted/50"
                )}>
                  <Icon className={cn("h-4 w-4", item.iconClass, isOpen && "animate-icon-bounce")} />
                </div>
                <span className="text-sm font-medium flex-1">{item.question}</span>
                <ChevronDown
                  className={cn(
                    "h-4 w-4 flex-shrink-0 text-muted-foreground transition-transform duration-500 ease-out",
                    isOpen && "rotate-180"
                  )}
                />
              </button>
              <div
                className={cn(
                  "overflow-hidden transition-all duration-500 ease-out",
                  isOpen ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
                )}
              >
                <div className="px-4 pb-4 pl-[5.5rem] text-sm text-muted-foreground leading-relaxed relative z-10">
                  {parseMarkdown(item.answer)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
