import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { workDir, datasetDir } from "@/lib/paths";

export const dynamic = "force-dynamic";

/**
 * GET /api/frames/{id}?limit=16
 *
 * Live view of the panels the pipeline is currently slicing / captioning /
 * rendering. Reads the sliced frames under work/temp_slices/chap_XXX/ and pairs
 * each with its transcription + visual caption from that chapter's
 * dataset/chapter_XXX/narration.json.
 *
 * Returns the most recently written frames (by mtime) so the UI shows a live
 * "what's being processed right now" filmstrip. Empty once a finished job's
 * work/ dir is cleaned up — that's expected; the final video is the artifact.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
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

  type Row = { chapter: number; name: string; mtime: number };
  const rows: Row[] = [];
  const narrationCache = new Map<number, Record<string, { text: string; status: string; visual?: string }>>();

  for (const cd of chapDirs) {
    const chapter = parseInt(cd.replace(/\D/g, ""), 10);
    if (!Number.isFinite(chapter)) continue;
    let entries: string[] = [];
    try {
      entries = (await fs.readdir(path.join(slicesRoot, cd))).filter((f) =>
        /^frame_\d+\.jpe?g$/i.test(f)
      );
    } catch {
      continue;
    }
    for (const f of entries) {
      try {
        const st = await fs.stat(path.join(slicesRoot, cd, f));
        rows.push({ chapter, name: f, mtime: st.mtimeMs });
      } catch {
        /* ignore */
      }
    }
    // load narration.json once per chapter
    try {
      const raw = await fs.readFile(
        path.join(datasetDir(id), `chapter_${String(chapter).padStart(3, "0")}`, "narration.json"),
        "utf8"
      );
      const list = JSON.parse(raw) as Array<{ image: string; text: string; status: string; visual?: string }>;
      const map: Record<string, { text: string; status: string; visual?: string }> = {};
      for (const n of list) map[n.image] = { text: n.text || "", status: n.status || "", visual: n.visual };
      narrationCache.set(chapter, map);
    } catch {
      /* narration.json not written yet */
    }
  }

  if (all) {
    // natural reading order: chapter then frame number
    const num = (s: string) => parseInt(s.replace(/\D/g, ""), 10) || 0;
    rows.sort((a, b) => a.chapter - b.chapter || num(a.name) - num(b.name));
  } else {
    rows.sort((a, b) => b.mtime - a.mtime);
  }
  const frames = rows.slice(0, limit).map((r) => {
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
  // live strip: newest-last so it reads left→right in processing order
  if (!all) frames.reverse();

  return NextResponse.json({
    frames,
    ready: true,
    total: rows.length,
  });
}
