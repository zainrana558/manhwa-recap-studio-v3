"use client";

import { useState, useCallback, useEffect } from "react";

const STORAGE_KEY = "mrs-recently-viewed";
const MAX_ITEMS = 20;

export interface RecentManga {
  id: string;
  title: string;
  coverUrl: string;
  viewedAt: number;
}

function loadItems(): RecentManga[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveItems(items: RecentManga[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
  } catch {
    // localStorage may be full or unavailable
  }
}

function needsCoverHeal(url: string | null | undefined): boolean {
  return !url || url.includes("cdn.asurascans.com");
}

export function useRecentlyViewed() {
  const [recentlyViewed, setRecentlyViewed] = useState<RecentManga[]>(() => {
    if (typeof window === "undefined") return [];
    return loadItems().sort((a, b) => b.viewedAt - a.viewedAt);
  });

  // Self-heal cached items that have missing or Cloudflare-blocked cover URLs.
  useEffect(() => {
    const broken = recentlyViewed.filter((item) => needsCoverHeal(item.coverUrl));
    if (broken.length === 0) return;
    let cancelled = false;
    Promise.all(
      broken.slice(0, 5).map(async (item) => {
        try {
          const res = await fetch(`/api/search?q=${encodeURIComponent(item.title)}&limit=5`);
          if (!res.ok) return null;
          const data = (await res.json()) as {
            manga?: Array<{ title?: string; coverUrl?: string | null }>;
          };
          const good = (data.manga ?? []).find(
            (m) => m.coverUrl && !needsCoverHeal(m.coverUrl)
          );
          return good?.coverUrl ? { id: item.id, coverUrl: good.coverUrl } : null;
        } catch {
          return null;
        }
      })
    ).then((fixes) => {
      if (cancelled) return;
      const fixMap = new Map<string, string>();
      for (const f of fixes) {
        if (f) fixMap.set(f.id, f.coverUrl);
      }
      if (fixMap.size === 0) return;
      setRecentlyViewed((prev) => {
        const next = prev.map((it) =>
          fixMap.has(it.id) ? { ...it, coverUrl: fixMap.get(it.id)! } : it
        );
        saveItems(next);
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
    // Run once on mount to repair previously cached items.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const addViewed = useCallback(
    (manga: { id: string; title: string; coverUrl: string }) => {
      setRecentlyViewed((prev) => {
        const filtered = prev.filter((item) => item.id !== manga.id);
        const next = [
          { ...manga, viewedAt: Date.now() },
          ...filtered,
        ].slice(0, MAX_ITEMS);
        saveItems(next);
        return next;
      });
    },
    []
  );

  const removeItem = useCallback((id: string) => {
    setRecentlyViewed((prev) => {
      const next = prev.filter((item) => item.id !== id);
      saveItems(next);
      return next;
    });
  }, []);

  const clearAll = useCallback(() => {
    setRecentlyViewed([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  return { recentlyViewed, addViewed, removeItem, clearAll };
}
