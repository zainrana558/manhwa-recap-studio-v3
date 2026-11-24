import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { workDir, datasetDir } from "@/lib/paths";

export const dynamic = "force-dynamic";

type NarrationEntry = { text: string; status: string; visual?: string };

async function readChapterNarration(
  id: string,
  chapter: number,
): Promise<Record<string, NarrationEntry> | null> {
  try {
    const raw = await fs.readFile(
      path.join(datasetDir(id), `chapter_${String(chapter).padStart(3, "0")}`, "narration.json"),
      "utf8",
    );
    const list = JSON.parse(raw) as Array<{ image: string; text: string; status: string; visual?: string }>;
    const map: Record<string, NarrationEntry> = {};
    for (const n of list) map[n.image] = { text: n.text || "", status: n.status || "", visual: n.visual };
    return map;
  } catch {
    return null;
  }
}

/**
 * GET /api/frames/{id}?limit=16
 *
 * Live view of the panels the pipeline is currently slicing / captioning /
 * rendering. Reads sliced frames under work/temp_slices/chap_XXX/ and pairs
 * each with its transcription + visual caption from narration.json.
 *
 * For live strip polls (without ?all=1), stats ONLY the directory metadata
 * across chapters (~N stats) rather than every individual frame file (~27k+
 * stats on 300+ chapter jobs), then stats individual frames only inside the
 * newest active chapters.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
    return NextResponse.json({ error: "bad id" }, { status: 400 });
  }
  const all = req.nextUrl.searchParams.get("all") === "1";
  const limit = all
    ? 100000
    : Math.min(48, Math.max(1, Number(req.nextUrl.searchParams.get("limit")) || 16));

  const slicesRoot = path.join(workDir(id), "temp_slices");
  let chapDirs: string[] = [];
  try {
    chapDirs = (await fs.readdir(slicesRoot, { withFileTypes: true }))
      .filter((e) => e.isDirectory() && /^chap_\d+$/i.test(e.name))
      .map((e) => e.name);
  } catch {
    return NextResponse.json({ frames: [], ready: false });
  }

  type ChapMeta = { cd: string; chapter: number; entries: string[]; dirMtime: number };
  const metasRaw = await Promise.all(
    chapDirs.map(async (cd): Promise<ChapMeta | null> => {
      const chapter = parseInt(cd.replace(/\D/g, ""), 10);
      if (!Number.isFinite(chapter)) return null;
      const dirPath = path.join(slicesRoot, cd);
      try {
        const [files, st] = await Promise.all([fs.readdir(dirPath), fs.stat(dirPath)]);
        const entries = files.filter((f) => /^frame_\d+\.jpe?g$/i.test(f));
        if (entries.length === 0) return null;
        return { cd, chapter, entries, dirMtime: st.mtimeMs };
      } catch {
        return null;
      }
    }),
  );
  const metas = metasRaw.filter((m): m is ChapMeta => m !== null);
  const totalFrames = metas.reduce((acc, m) => acc + m.entries.length, 0);

  type Row = { chapter: number; name: string; mtime: number };
  let selectedRows: Row[] = [];
  const narrationCache = new Map<number, Record<string, NarrationEntry>>();

  if (all) {
    const num = (s: string) => parseInt(s.replace(/\D/g, ""), 10) || 0;
    metas.sort((a, b) => a.chapter - b.chapter);
    for (const m of metas) {
      m.entries.sort((a, b) => num(a) - num(b));
      for (const f of m.entries) {
        selectedRows.push({ chapter: m.chapter, name: f, mtime: m.dirMtime });
        if (selectedRows.length >= limit) break;
      }
      if (selectedRows.length >= limit) break;
    }
    const neededChaps = [...new Set(selectedRows.map((r) => r.chapter))];
    await Promise.all(
      neededChaps.map(async (ch) => {
        const map = await readChapterNarration(id, ch);
        if (map) narrationCache.set(ch, map);
      }),
    );
  } else {
    // Newest-touched chapters first; stat only individual frames inside those
    // recent chapters (typically 1-3 dirs) instead of scanning all 300+ dirs.
    metas.sort((a, b) => b.dirMtime - a.dirMtime || b.chapter - a.chapter);
    const recentRows: Row[] = [];
    for (const m of metas) {
      const dirPath = path.join(slicesRoot, m.cd);
      const fileRows = await Promise.all(
        m.entries.map(async (f): Promise<Row | null> => {
          try {
            const st = await fs.stat(path.join(dirPath, f));
            return { chapter: m.chapter, name: f, mtime: st.mtimeMs };
          } catch {
            return null;
          }
        }),
      );
      for (const r of fileRows) {
        if (r) recentRows.push(r);
      }
      if (recentRows.length >= limit * 2) break;
    }
    recentRows.sort((a, b) => b.mtime - a.mtime);
    selectedRows = recentRows.slice(0, limit);

    const neededChaps = [...new Set(selectedRows.map((r) => r.chapter))];
    await Promise.all(
      neededChaps.map(async (ch) => {
        const map = await readChapterNarration(id, ch);
        if (map) narrationCache.set(ch, map);
      }),
    );
  }

  const frames = selectedRows.map((r) => {
    const n = narrationCache.get(r.chapter)?.[r.name];
    return {
      url: `/api/frames/${id}/${r.chapter}/${r.name}`,
      id: `chap_${String(r.chapter).padStart(3, "0")}/${r.name}`,
      chapter: r.chapter,
      name: r.name,
      text: n?.text ?? "",
      visual: n?.visual ?? "",
      status: n?.status ?? "",
    };
  });
  // live strip: newest-last so it reads left->right in processing order
  if (!all) frames.reverse();

  return NextResponse.json({
    frames,
    ready: true,
    total: totalFrames,
  });
}
