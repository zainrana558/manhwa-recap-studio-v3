"use client";

import { useEffect, useState, useRef, type ReactNode } from "react";
import { motion, useMotionValue, useSpring, useTransform, type MotionValue } from "framer-motion";
import {
  Sparkles,
  Layers,
  ChevronLeft,
  ChevronRight,
  Bookmark,
  BookmarkCheck,
  Info,
  BookOpen,
  Flame,
  Star,
} from "lucide-react";
import type { MangadexManga } from "@/types/pipeline";

/**
 * Kinetic Typed-Out Editorial Title Component.
 */
const TYPED_PHRASES = [
  "Into Cinema.",
  "Into Narrated Epics.",
  "Into Living Stories.",
  "Into Studio Editions.",
];

export function TypedCinemaTitle() {
  const [phraseIdx, setPhraseIdx] = useState(0);
  const [charCount, setCharCount] = useState(TYPED_PHRASES[0].length);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const current = TYPED_PHRASES[phraseIdx];
    let delay = deleting ? 36 : 68;
    if (!deleting && charCount === current.length) delay = 2200;
    else if (deleting && charCount === 0) delay = 360;

    const timer = setTimeout(() => {
      if (!deleting) {
        if (charCount < current.length) setCharCount((c) => c + 1);
        else setDeleting(true);
      } else {
        if (charCount > 0) setCharCount((c) => c - 1);
        else {
          setDeleting(false);
          setPhraseIdx((i) => (i + 1) % TYPED_PHRASES.length);
        }
      }
    }, delay);
    return () => clearTimeout(timer);
  }, [phraseIdx, charCount, deleting]);

  return (
    <span className="inline-flex items-baseline justify-center min-h-[1.12em]">
      <span className="font-serif italic font-medium gold-foil-live">
        {TYPED_PHRASES[phraseIdx].slice(0, charCount)}
      </span>
      <span
        aria-hidden="true"
        className="inline-block w-[3px] h-[0.82em] ml-1.5 align-baseline bg-gradient-to-b from-[#FFF0B8] to-[#E11D48] rounded-full animate-pulse"
      />
    </span>
  );
}

/**
 * Lightweight GPU-friendly floating champagne-gold & cherry embers (8 motes for top Core Web Vitals).
 */
const EMBER_SPECS = [
  { left: "8%", size: 4, dur: 15, delay: 0, gold: true },
  { left: "21%", size: 3, dur: 19, delay: 2.8, gold: false },
  { left: "36%", size: 4, dur: 17, delay: 5.5, gold: true },
  { left: "51%", size: 3, dur: 20, delay: 1.5, gold: false },
  { left: "66%", size: 4, dur: 16, delay: 6.4, gold: true },
  { left: "79%", size: 3, dur: 18, delay: 3.7, gold: false },
  { left: "91%", size: 4, dur: 16, delay: 0.9, gold: true },
  { left: "44%", size: 3, dur: 22, delay: 9.0, gold: true },
];

export function AtelierAmbientParticles() {
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      {EMBER_SPECS.map((p, idx) => (
        <span
          key={idx}
          className="atelier-ember absolute bottom-[-24px] rotate-45 rounded-[1px]"
          style={{
            left: p.left,
            width: `${p.size}px`,
            height: `${p.size}px`,
            background: p.gold ? "#F5D27A" : "#F43F5E",
            boxShadow: p.gold
              ? "0 0 8px 1px rgba(245, 210, 122, 0.45)"
              : "0 0 8px 1px rgba(244, 63, 94, 0.55)",
            animationDuration: `${p.dur}s`,
            animationDelay: `${p.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

// Keep CursorVelvetGlow as a zero-overhead CSS-only spotlight (no React re-render on mousemove!)
export function CursorVelvetGlow() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const onMove = (e: MouseEvent) => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        el.style.setProperty("--cx", `${e.clientX}px`);
        el.style.setProperty("--cy", `${e.clientY}px`);
        el.style.opacity = "1";
        raf = 0;
      });
    };
    const onLeave = () => {
      el.style.opacity = "0";
    };
    window.addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", onLeave);
    return () => {
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div
      ref={ref}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0 transition-opacity duration-500 opacity-0"
      style={{
        mixBlendMode: "screen",
        background:
          "radial-gradient(620px circle at var(--cx, -600px) var(--cy, -600px), rgba(255, 234, 175, 0.28) 0%, rgba(244, 63, 94, 0.30) 28%, rgba(225, 29, 72, 0.12) 52%, transparent 74%)",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Unique Editorial Genre Atelier — Sculpted Roman-Numeral Medallions (NO ICONS!)
// ---------------------------------------------------------------------------
export interface GenrePreset {
  id: string;
  roman: string;
  label: string;
  query: string;
}

export const GENRE_PRESETS: GenrePreset[] = [
  { id: "all", roman: "00", label: "All Genres", query: "" },
  { id: "murim", roman: "I", label: "Murim", query: "Murim" },
  { id: "regression", roman: "II", label: "Regression", query: "Return" },
  { id: "system", roman: "III", label: "System", query: "Leveling" },
  { id: "fantasy", roman: "IV", label: "Fantasy", query: "Magic" },
  { id: "tower", roman: "V", label: "Tower", query: "Tower" },
  { id: "necromancer", roman: "VI", label: "Necromancer", query: "Necromancer" },
  { id: "revenge", roman: "VII", label: "Revenge", query: "Revenge" },
  { id: "action", roman: "VIII", label: "Action", query: "Sword" },
  { id: "comedy", roman: "IX", label: "Comedy", query: "Greatest Estate" },
];

export function EditorialGenreAtelier({
  activeGenre,
  onSelectGenre,
}: {
  activeGenre: string;
  onSelectGenre: (genre: GenrePreset) => void;
}) {
  return (
    <div className="max-w-4xl mx-auto pt-2 space-y-3">
      <div className="flex items-center justify-center gap-3">
        <span className="h-px w-8 bg-gradient-to-r from-transparent to-[#F5D27A]/60" />
        <span className="text-[9.5px] font-mono uppercase tracking-[0.3em] text-[#F5D27A]/90">
          Curated Genre Atelier
        </span>
        <span className="h-px w-8 bg-gradient-to-l from-transparent to-[#F5D27A]/60" />
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-2.5">
        {GENRE_PRESETS.map((g) => {
          const isActive = activeGenre === g.id;
          return (
            <button
              key={g.id}
              type="button"
              onClick={() => onSelectGenre(g)}
              className={`group relative overflow-hidden inline-flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-medium cursor-pointer transition-all duration-200 hover:-translate-y-0.5 active:scale-95 ${
                isActive
                  ? "text-[#FFF8F5] border-[#F5D27A]/80 shadow-[0_12px_28px_-8px_rgba(225,29,72,0.68)]"
                  : "bg-[#170910] dark:bg-[#16080F] text-foreground/85 border-primary/25 hover:border-[#F5D27A]/65"
              }`}
            >
              {isActive && (
                <motion.span
                  layoutId="activeGenreMedallion"
                  transition={{ type: "spring", stiffness: 320, damping: 28 }}
                  className="absolute inset-0 bg-gradient-to-r from-[#E11D48] via-[#BE123C] to-[#9F1239]"
                />
              )}
              <span
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-gradient-to-r from-transparent via-[#FFF0B8]/20 to-transparent"
              />
              <span
                className={`relative z-10 text-[9px] font-mono tracking-widest ${
                  isActive ? "text-[#FFF0B8]" : "text-[#F5D27A]/70 group-hover:text-[#F5D27A]"
                }`}
              >
                {g.roman}
              </span>
              <span className="relative z-10 tracking-wide">{g.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 9 Verified Scrapeable Hero Recommendations
// ---------------------------------------------------------------------------
const HERO_MANGA: MangadexManga[] = [
  {
    id: "vs-161--tower-of-god",
    title: "Tower of God",
    source: "vortexscans",
    coverUrl: "https://storage.vortexscans.org/upload/2024/10/05/14c892be-717c-40ea-acb7-fe1cded34396.webp",
    lastChapter: "652",
    status: "ongoing",
    year: 2010,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Bam enters the mysterious Tower to chase after Rachel and uncovers a world of tests, factions, and godlike power.",
    tags: ["Action", "Adventure", "Fantasy"],
  },
  {
    id: "as-return-of-the-mount-hua-sect",
    title: "Return of the Mount Hua Sect",
    source: "asurascans",
    coverUrl: "https://meo.comick.pictures/g6BDzb.jpg",
    lastChapter: "195",
    status: "ongoing",
    year: 2021,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Chung Myung, the Plum Blossom Sword Saint, reincarnates 100 years later to restore the fallen Mount Hua Sect.",
    tags: ["Action", "Comedy", "Murim"],
  },
  {
    id: "as-solo-leveling",
    title: "Solo Leveling",
    source: "asurascans",
    coverUrl: "https://meo.comick.pictures/0GRgd.jpg",
    lastChapter: "201",
    status: "completed",
    year: 2018,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "The weakest hunter of all mankind awakens a mysterious System that lets him level up without limits.",
    tags: ["Action", "Fantasy", "System"],
  },
  {
    id: "as-the-greatest-estate-developer",
    title: "The Greatest Estate Developer",
    source: "asurascans",
    coverUrl: "https://meo.comick.pictures/R8ynbb.jpg",
    lastChapter: "223",
    status: "completed",
    year: 2021,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Civil engineering student Suho Kim wakes up as lazy noble Lloyd Frontera and reshapes a fantasy kingdom.",
    tags: ["Action", "Comedy", "Fantasy"],
  },
  {
    id: "as-nano-machine",
    title: "Nano Machine",
    source: "asurascans",
    coverUrl: "https://meo.comick.pictures/v8YXRJ.jpg",
    lastChapter: "339",
    status: "ongoing",
    year: 2020,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Nanotechnology meets martial arts at the Mashin Academy as Cheon Yeo-woon rises to become Heavenly Demon.",
    tags: ["Action", "Fantasy", "Murim"],
  },
  {
    id: "wt-2154",
    title: "Omniscient Reader",
    source: "webtoons",
    coverUrl: "https://webtoon-phinf.pstatic.net/20261124_249/1795497294345Rpz4W_JPEG/7Thumb_Poster.jpg?type=q90",
    lastChapter: "317",
    status: "ongoing",
    year: 2020,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Kim Dokja is the only person who knows how the world ends when his favorite web novel becomes reality.",
    tags: ["Action", "Fantasy", "Survival"],
  },
  {
    id: "ts-the-beginning-after-the-end",
    title: "The Beginning After The End",
    source: "thunderscans",
    coverUrl: "https://en-thunderscans.com/wp-content/uploads/2024/11/2024-11-04-01-44-20-1730727860533-200x300.jpeg",
    lastChapter: "259",
    status: "ongoing",
    year: 2018,
    contentRating: "safe",
    originalLanguage: "en",
    description: "King Grey is reincarnated into a world of magic and monsters with a second chance to protect those he loves.",
    tags: ["Reincarnation", "Action", "Fantasy"],
  },
  {
    id: "as-eleceed",
    title: "Eleceed",
    source: "asurascans",
    coverUrl: "https://meo.comick.pictures/W8z8e4.jpg",
    lastChapter: "423",
    status: "ongoing",
    year: 2018,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Kind-hearted speedster Jiwoo teams up with Kayden, an overpowered awakener trapped inside a chubby cat.",
    tags: ["Action", "Comedy", "Supernatural"],
  },
  {
    id: "as-lookism",
    title: "Lookism",
    source: "asurascans",
    coverUrl: "https://meo.comick.pictures/0Zo6aq.jpg",
    lastChapter: "626",
    status: "ongoing",
    year: 2014,
    contentRating: "safe",
    originalLanguage: "ko",
    description: "Daniel wakes up with a second body and gets pulled into the underground world of Seoul's four major crews.",
    tags: ["Action", "Drama", "Martial Arts"],
  },
];

// ---------------------------------------------------------------------------
// Shared Reusable Liquid 3D Card (Idle: Cover + Ch. Count + Name ONLY;
// Hover: Pop-up Description + Tags + Shine + 3D Tilt; Click Anywhere Opens!)
// ---------------------------------------------------------------------------
function Liquid3DDeckCard({
  item,
  index,
  smoothPos,
  isCenter,
  badgeLeft,
  isResolving = false,
  bookmarked = false,
  onCardClick,
  onExpandInfo,
  onBookmark,
  score,
}: {
  item: MangadexManga;
  index: number;
  smoothPos: MotionValue<number>;
  isCenter: boolean;
  badgeLeft?: string;
  isResolving?: boolean;
  bookmarked?: boolean;
  onCardClick: () => void;
  onExpandInfo?: () => void;
  onBookmark?: (e: React.MouseEvent) => void;
  score?: number | null;
}) {
  const [imgFailed, setImgFailed] = useState(false);
  const [hovering, setHovering] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const mx = useMotionValue(0.5);
  const my = useMotionValue(0.5);
  const sx = useSpring(mx, { stiffness: 200, damping: 24 });
  const sy = useSpring(my, { stiffness: 200, damping: 24 });
  const tiltX = useTransform(sy, [0, 1], [10, -10]);
  const tiltY = useTransform(sx, [0, 1], [-10, 10]);
  const shineBg = useTransform(
    [sx, sy],
    ([px, py]: number[]) =>
      `radial-gradient(280px circle at ${(px as number) * 100}% ${(py as number) * 100}%, rgba(255, 244, 198, 0.36), rgba(244, 63, 94, 0.12) 45%, transparent 72%)`
  );

  const x = useTransform(smoothPos, (p) => (index - p) * 150);
  const y = useTransform(smoothPos, (p) => Math.abs(index - p) * 14 - (1 - Math.min(1, Math.abs(index - p))) * 8);
  const z = useTransform(smoothPos, (p) => -Math.abs(index - p) * 90);
  const baseRotateY = useTransform(smoothPos, (p) => Math.max(-36, Math.min(36, (index - p) * -19)));
  const rotateZ = useTransform(smoothPos, (p) => (index - p) * 2.8);
  const scale = useTransform(smoothPos, (p) => Math.max(0.7, 1.05 - Math.abs(index - p) * 0.12));
  const opacity = useTransform(smoothPos, (p) => {
    const dist = Math.abs(index - p);
    if (dist > 3.6) return 0;
    return Math.max(0.3, 1 - dist * 0.21);
  });
  const zIndex = useTransform(smoothPos, (p) => 50 - Math.round(Math.abs(index - p) * 10));

  const onCardMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isCenter) return;
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;
    mx.set((e.clientX - rect.left) / rect.width);
    my.set((e.clientY - rect.top) / rect.height);
  };

  const onCardMouseLeave = () => {
    setHovering(false);
    mx.set(0.5);
    my.set(0.5);
  };

  return (
    <motion.div
      ref={cardRef}
      style={{
        x,
        y,
        z,
        rotateY: isCenter && hovering ? tiltY : baseRotateY,
        rotateX: isCenter && hovering ? tiltX : 0,
        rotateZ,
        scale,
        opacity,
        zIndex,
        backfaceVisibility: "hidden",
        WebkitBackfaceVisibility: "hidden",
        willChange: "transform, opacity",
        borderColor: isCenter ? "rgba(245, 210, 122, 0.9)" : "rgba(244, 63, 94, 0.28)",
        boxShadow: isCenter
          ? "0 34px 72px -14px rgba(225, 29, 72, 0.58), 0 0 0 1px rgba(245, 210, 122, 0.48)"
          : "0 22px 46px -16px rgba(0, 0, 0, 0.88)",
      }}
      onMouseEnter={() => isCenter && setHovering(true)}
      onMouseMove={onCardMouseMove}
      onMouseLeave={onCardMouseLeave}
      onClick={onCardClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onCardClick();
        }
      }}
      className="group absolute w-[220px] sm:w-[255px] h-[330px] sm:h-[380px] rounded-2xl overflow-hidden text-left cursor-pointer border bg-[#160810] select-none"
      aria-label={`${item.title}${item.lastChapter ? ` — Ch. ${item.lastChapter}` : ""}`}
    >
      {/* Full-Bleed Optimized WebP Cover */}
      {item.coverUrl && !imgFailed ? (
        <img
          src={`/api/cover?url=${encodeURIComponent(item.coverUrl)}&w=420`}
          alt={item.title}
          referrerPolicy="no-referrer"
          draggable={false}
          decoding="async"
          onError={() => setImgFailed(true)}
          className="w-full h-full object-cover pointer-events-none"
        />
      ) : (
        <div
          className="w-full h-full flex flex-col items-center justify-center p-6 text-center"
          style={{
            background: "radial-gradient(circle at 50% 30%, rgba(225,29,72,0.4) 0%, #1C0812 60%, #0E050A 100%)",
          }}
        >
          <Sparkles className="h-7 w-7 text-[#F5D27A] mb-2" />
          <p className="font-serif italic text-lg text-[#FFF8F5] line-clamp-3">{item.title}</p>
        </div>
      )}

      {/* Pointer-Tracking Specular Champagne Shine */}
      <motion.div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-20 transition-opacity duration-300"
        style={{
          opacity: isCenter && hovering ? 1 : 0,
          background: shineBg,
        }}
      />

      {/* Base bottom gradient scrim */}
      <div
        className={`pointer-events-none absolute inset-0 z-10 transition-all duration-300 ${
          isCenter && hovering
            ? "bg-gradient-to-t from-[#0B0408] via-[#0E050A]/85 to-[#0E050A]/35"
            : "bg-gradient-to-t from-[#0D0509] via-[#0D0509]/25 to-transparent"
        }`}
      />

      {/* Top Row: Idle shows ONLY Chapter Count Jewel; Hover reveals Left Tag + Quick Icons */}
      <div className="absolute top-3 inset-x-3 flex items-center justify-between gap-2 z-20">
        <div
          className={`flex items-center gap-1.5 transition-opacity duration-200 ${
            isCenter && hovering ? "opacity-100" : "opacity-0"
          }`}
        >
          {badgeLeft && (
            <span className="px-2 py-0.5 rounded-full text-[9px] font-mono uppercase tracking-[0.14em] bg-[#16070F]/90 text-[#FFE8EC] border border-[#F43F5E]/50">
              {isResolving ? "Matching…" : badgeLeft}
            </span>
          )}
          {score && (
            <span className="inline-flex items-center gap-0.5 px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#16070F]/90 text-[#F5D27A] border border-[#F5D27A]/40">
              <Star className="h-2.5 w-2.5 fill-[#F5D27A] text-[#F5D27A]" />
              {score}%
            </span>
          )}
        </div>

        {item.lastChapter ? (
          <span className="ml-auto inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-gradient-to-r from-[#E11D48] to-[#BE123C] text-[#FFF8F5] border border-[#F5D27A]/75 shadow-[0_6px_16px_rgba(225,29,72,0.58)]">
            <BookOpen className="h-2.5 w-2.5 text-[#FFF0B8]" />
            Ch. {item.lastChapter}
          </span>
        ) : item.year ? (
          <span className="ml-auto px-2 py-0.5 rounded-full text-[9.5px] font-mono bg-[#16070F]/85 text-[#F5D27A] border border-[#F5D27A]/40">
            {item.year}
          </span>
        ) : null}
      </div>

      {/* Bottom Info Plate: Idle = Name ONLY; Hover = Smoothly Pop-Up Description + Genres + Actions */}
      <div className="absolute bottom-0 inset-x-0 p-4 z-20 space-y-2">
        <h3 className="font-serif font-semibold text-lg sm:text-xl leading-tight text-[#FFF8F5] line-clamp-2 drop-shadow">
          {item.title}
        </h3>

        {/* Hover Pop-Up Details Drawer */}
        <div
          className={`grid transition-all duration-300 ease-out ${
            isCenter && hovering ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
          }`}
        >
          <div className="overflow-hidden space-y-2.5">
            {item.description && (
              <p className="text-[11.5px] text-[#FFE8EC]/85 leading-relaxed line-clamp-3">
                {item.description}
              </p>
            )}
            {item.tags && item.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {item.tags.slice(0, 3).map((t) => (
                  <span
                    key={t}
                    className="px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#240A16] text-[#F5D27A] border border-[#F5D27A]/30"
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-center justify-between gap-2 pt-1" onPointerDown={(e) => e.stopPropagation()}>
              <span className="text-[10px] font-mono uppercase tracking-[0.16em] text-[#FFF0B8] font-semibold">
                Click Card For Recap →
              </span>
              <div className="flex items-center gap-1.5">
                {onExpandInfo && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onExpandInfo();
                    }}
                    aria-label={`Quick info for ${item.title}`}
                    className="h-7 w-7 rounded-full bg-[#1C0A13] border border-[#F5D27A]/45 text-[#F5D27A] flex items-center justify-center cursor-pointer"
                  >
                    <Info className="h-3 w-3" />
                  </button>
                )}
                {onBookmark && (
                  <button
                    type="button"
                    onClick={onBookmark}
                    aria-label={`Bookmark ${item.title}`}
                    className={`h-7 w-7 rounded-full bg-[#1C0A13] border flex items-center justify-center cursor-pointer ${
                      bookmarked ? "border-[#F43F5E] text-[#F43F5E]" : "border-[#F5D27A]/45 text-[#FFE8EC]"
                    }`}
                  >
                    {bookmarked ? <BookmarkCheck className="h-3 w-3" /> : <Bookmark className="h-3 w-3" />}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Generic Smooth Drag-Slidable 3D Deck Container (Hero 9-Card + Search + Trending)
// ---------------------------------------------------------------------------
function Base3DDeck({
  items,
  initialIndex = 0,
  onSelectCenter,
  onExpandInfo,
  isBookmarked,
  onBookmark,
  resolvingId,
  sourceLabelMap,
  hintText,
  scores,
}: {
  items: MangadexManga[];
  initialIndex?: number;
  onSelectCenter: (m: MangadexManga) => void;
  onExpandInfo?: (m: MangadexManga) => void;
  isBookmarked?: (id: string) => boolean;
  onBookmark?: (e: React.MouseEvent, m: MangadexManga) => void;
  resolvingId?: string | null;
  sourceLabelMap?: Record<string, string>;
  hintText: string;
  scores?: Record<string, number | null | undefined>;
}) {
  const startIdx = Math.min(initialIndex, Math.max(0, items.length - 1));
  const [activeIdx, setActiveIdx] = useState(startIdx);
  const rawPos = useMotionValue(startIdx);
  const smoothPos = useSpring(rawPos, { stiffness: 120, damping: 24, mass: 0.85 });

  const dragRef = useRef({
    dragging: false,
    startX: 0,
    startPos: startIdx,
    lastX: 0,
    lastTime: 0,
    velocity: 0,
    movedPx: 0,
  });

  useEffect(() => {
    const idx = Math.min(initialIndex, Math.max(0, items.length - 1));
    setActiveIdx(idx);
    rawPos.set(idx);
  }, [items.length, items[0]?.id, initialIndex, rawPos]);

  if (!items.length) return null;

  const safeIdx = Math.min(activeIdx, items.length - 1);
  const activeItem = items[safeIdx];

  const snapTo = (nextIdx: number) => {
    const clamped = Math.max(0, Math.min(items.length - 1, nextIdx));
    setActiveIdx(clamped);
    rawPos.set(clamped);
  };

  const goPrev = () => snapTo(safeIdx > 0 ? safeIdx - 1 : items.length - 1);
  const goNext = () => snapTo(safeIdx < items.length - 1 ? safeIdx + 1 : 0);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    dragRef.current = {
      dragging: true,
      startX: e.clientX,
      startPos: rawPos.get(),
      lastX: e.clientX,
      lastTime: performance.now(),
      velocity: 0,
      movedPx: 0,
    };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.dragging) return;
    const now = performance.now();
    const dt = Math.max(1, now - dragRef.current.lastTime);
    const dx = e.clientX - dragRef.current.startX;
    dragRef.current.velocity = (e.clientX - dragRef.current.lastX) / dt;
    dragRef.current.lastX = e.clientX;
    dragRef.current.lastTime = now;
    dragRef.current.movedPx = Math.max(dragRef.current.movedPx, Math.abs(dx));
    if (dragRef.current.movedPx > 6 && !(e.currentTarget as HTMLDivElement).hasPointerCapture(e.pointerId)) {
      try {
        (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
      } catch {}
    }

    let next = dragRef.current.startPos - dx / 160;
    if (next < 0) next = next * 0.25;
    if (next > items.length - 1) next = items.length - 1 + (next - (items.length - 1)) * 0.25;
    rawPos.set(next);
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.dragging) return;
    dragRef.current.dragging = false;
    try {
      (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
    } catch {}
    const projected = rawPos.get() - dragRef.current.velocity * 1.2;
    snapTo(Math.round(projected));
  };

  const wheelLockRef = useRef(0);
  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.shiftKey ? e.deltaY : 0;
    if (Math.abs(delta) < 8) return;
    e.preventDefault();
    const now = performance.now();
    if (now - wheelLockRef.current < 165) return;
    wheelLockRef.current = now;
    snapTo(safeIdx + (delta > 0 ? 1 : -1));
  };

  const start = Math.max(0, safeIdx - 4);
  const end = Math.min(items.length - 1, safeIdx + 4);
  const windowItems: Array<{ item: MangadexManga; idx: number }> = [];
  for (let i = start; i <= end; i++) {
    windowItems.push({ item: items[i], idx: i });
  }

  return (
    <div className="relative py-3">
      <div
        className="relative h-[395px] sm:h-[440px] w-full max-w-5xl mx-auto flex items-center justify-center select-none touch-pan-y cursor-grab active:cursor-grabbing"
        style={{ perspective: "1450px" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
      >
        {items.length > 1 && (
          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={goPrev}
            aria-label="Previous card"
            className="absolute left-1 sm:left-3 z-50 h-10 w-10 rounded-full bg-[#190811] border border-[#F5D27A]/45 text-[#F5D27A] hover:border-[#F43F5E] hover:text-white hover:bg-[#E11D48] flex items-center justify-center shadow-lg transition-all hover:scale-105 active:scale-95 cursor-pointer"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
        )}

        {items.length > 1 && (
          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={goNext}
            aria-label="Next card"
            className="absolute right-1 sm:right-3 z-50 h-10 w-10 rounded-full bg-[#190811] border border-[#F5D27A]/45 text-[#F5D27A] hover:border-[#F43F5E] hover:text-white hover:bg-[#E11D48] flex items-center justify-center shadow-lg transition-all hover:scale-105 active:scale-95 cursor-pointer"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        )}

        {windowItems.map(({ item, idx }) => {
          const src = item.source ?? "mangahere";
          return (
            <Liquid3DDeckCard
              key={item.id}
              item={item}
              index={idx}
              smoothPos={smoothPos}
              isCenter={idx === safeIdx}
              badgeLeft={sourceLabelMap?.[src] ?? src}
              isResolving={resolvingId === item.id}
              bookmarked={isBookmarked?.(item.id) ?? false}
              score={scores?.[item.id]}
              onCardClick={() => {
                if (dragRef.current.movedPx > 8) return;
                if (idx === safeIdx) {
                  if (resolvingId !== item.id) onSelectCenter(item);
                } else {
                  snapTo(idx);
                }
              }}
              onExpandInfo={onExpandInfo ? () => onExpandInfo(item) : undefined}
              onBookmark={onBookmark ? (e) => onBookmark(e, item) : undefined}
            />
          );
        })}
      </div>

      {/* Footer Dots + Counter */}
      <div className="mt-2 flex flex-col items-center gap-2">
        <div className="flex items-center gap-1.5 flex-wrap justify-center max-w-md">
          {items.slice(0, 14).map((it, i) => (
            <button
              key={it.id}
              type="button"
              onClick={() => snapTo(i)}
              aria-label={`Focus ${it.title}`}
              className={`h-1.5 rounded-full transition-all duration-300 cursor-pointer ${
                i === safeIdx
                  ? "w-7 bg-gradient-to-r from-[#E11D48] to-[#F5D27A]"
                  : "w-1.5 bg-muted-foreground/35 hover:bg-[#F5D27A]/70"
              }`}
            />
          ))}
        </div>
        <div className="flex items-center gap-2.5 text-[10px] font-mono uppercase tracking-[0.2em] text-[#F5D27A]">
          <Layers className="h-2.5 w-2.5 text-[#F43F5E]" />
          <span>{hintText}</span>
          <span className="text-muted-foreground">·</span>
          <span className="text-foreground/80">
            {String(safeIdx + 1).padStart(2, "0")} / {String(items.length).padStart(2, "0")}
            {activeItem.lastChapter ? ` · Ch. ${activeItem.lastChapter}` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}

export function AtelierCinemaDeck({
  onSelectManga,
}: {
  onSelectManga?: (manga: MangadexManga) => void;
}) {
  return (
    <Base3DDeck
      items={HERO_MANGA}
      initialIndex={4}
      onSelectCenter={(m) => onSelectManga?.(m)}
      hintText="Drag To Slide · Click Card To Open Overview"
    />
  );
}

export function SearchResults3DCoverflow({
  items,
  onSelect,
  onExpandInfo,
  isBookmarked,
  onBookmark,
  resolvingId,
  sourceLabelMap,
}: {
  items: MangadexManga[];
  onSelect: (m: MangadexManga) => void;
  onExpandInfo: (m: MangadexManga) => void;
  isBookmarked?: (id: string) => boolean;
  onBookmark?: (e: React.MouseEvent, m: MangadexManga) => void;
  resolvingId?: string | null;
  sourceLabelMap: Record<string, string>;
}) {
  return (
    <Base3DDeck
      items={items}
      initialIndex={0}
      onSelectCenter={onSelect}
      onExpandInfo={onExpandInfo}
      isBookmarked={isBookmarked}
      onBookmark={onBookmark}
      resolvingId={resolvingId}
      sourceLabelMap={sourceLabelMap}
      hintText="Drag To Slide · Click Card For Overview"
    />
  );
}

/**
 * Real-Time Everyday Trending Korean Manhwa 3D Showcase!
 */
export function Trending3DShowcase({
  onSelectManga,
  isBookmarked,
  onBookmarkToggle,
}: {
  onSelectManga: (m: MangadexManga) => void;
  isBookmarked?: (id: string) => boolean;
  onBookmarkToggle?: (m: MangadexManga) => void;
}) {
  const [items, setItems] = useState<Array<MangadexManga & { averageScore?: number | null }>>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/trending")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled) return;
        setItems(Array.isArray(d?.items) ? d.items : []);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!loading && items.length === 0) return null;

  const scores: Record<string, number | null | undefined> = {};
  for (const it of items) scores[it.id] = it.averageScore;

  return (
    <section className="max-w-6xl mx-auto space-y-4">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2.5 px-3.5 py-1 rounded-full bg-[#1B0812] border border-[#F5D27A]/35 text-[10px] font-mono uppercase tracking-[0.26em] text-[#F5D27A]">
          <Flame className="h-3 w-3 text-[#F43F5E]" />
          Live Everyday Trending · Real-Time
        </div>
        <h2 className="font-serif text-3xl sm:text-4xl font-semibold text-foreground">
          Trending <span className="italic gold-foil-live">Today</span>
        </h2>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-md mx-auto">
          Hover any card to reveal details, or click anywhere on the focused cover to open its overview &amp; start a recap.
        </p>
      </div>

      {loading ? (
        <div className="h-[395px] flex items-center justify-center">
          <div className="flex items-center gap-3 px-5 py-2.5 rounded-full bg-card border border-primary/30 text-xs font-mono text-[#F5D27A]">
            <Sparkles className="h-4 w-4 animate-pulse text-[#F43F5E]" />
            Loading live daily trending…
          </div>
        </div>
      ) : (
        <Base3DDeck
          items={items}
          initialIndex={Math.min(3, Math.floor(items.length / 2))}
          onSelectCenter={onSelectManga}
          isBookmarked={isBookmarked}
          onBookmark={
            onBookmarkToggle
              ? (e, m) => {
                  e.stopPropagation();
                  onBookmarkToggle(m);
                }
              : undefined
          }
          scores={scores}
          hintText="Live Daily Trending · Click Card For Overview"
        />
      )}
    </section>
  );
}

/**
 * ULTRA-FAST Zero-Lag CSS-Hardware-Accelerated 3D Tilt + Dual-Layer Shine Card!
 * Uses pure CSS variables (--rx, --ry, --mx, --my) instead of Framer Motion hooks
 * so rendering 24–100+ search result cards stays locked at 60fps with top Core Web Vitals!
 */
export function Tilt3DResultCard({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
  delayIndex?: number;
}) {
  const innerRef = useRef<HTMLDivElement>(null);

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = innerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    el.style.setProperty("--rx", `${(0.5 - py) * 18}deg`);
    el.style.setProperty("--ry", `${(px - 0.5) * 18}deg`);
    el.style.setProperty("--mx", `${px * 100}%`);
    el.style.setProperty("--my", `${py * 100}%`);
  };

  const handleLeave = () => {
    const el = innerRef.current;
    if (!el) return;
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
  };

  return (
    <div
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      className={`tilt-card-outer ${className}`}
    >
      <div ref={innerRef} className="tilt-card-inner group/shine relative rounded-2xl">
        {children}
        {/* Layer 1: Pointer-tracking champagne-gold specular shine */}
        <div aria-hidden="true" className="tilt-card-sheen pointer-events-none absolute inset-0 rounded-2xl z-30" />
        {/* Layer 2: Diagonal metallic foil ribbon sweep on hover */}
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 rounded-2xl overflow-hidden z-30">
          <div className="absolute inset-y-0 -left-full w-2/3 -skew-x-12 bg-gradient-to-r from-transparent via-[#FFF4C6]/28 to-transparent group-hover/shine:left-[160%] transition-all duration-700 ease-out" />
        </div>
      </div>
    </div>
  );
}

/**
 * High-fashion kinetic cinema ticker — pauses smoothly on hover.
 */
export function EditorialMarqueeStrip() {
  const items = [
    "YOLO11N-SEG PANEL SCULPTING",
    "1920×1080 CINEMA MASTER",
    "KOKORO NEURAL NARRATION",
    "13 LIVE MANHWA ARCHIVES",
    "RAPIDOCR SPEECH ISOLATION",
    "ZERO-CLIP VERTICAL PANS",
  ];
  const repeated = [...items, ...items];

  return (
    <div className="relative w-full overflow-hidden border-y border-[#F5D27A]/25 bg-[#1B0711] dark:bg-[#16060E] py-2.5 select-none group">
      <div className="flex w-max animate-editorial-marquee group-hover:[animation-play-state:paused]">
        {repeated.map((text, i) => (
          <div
            key={i}
            className="flex items-center gap-6 px-6 text-[10px] font-mono uppercase tracking-[0.28em] text-[#FFE8EC]/85"
          >
            <span>{text}</span>
            <Sparkles className="h-3 w-3 text-[#F5D27A] shrink-0" />
          </div>
        ))}
      </div>
    </div>
  );
}
