import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { db } from "@/lib/db";
import { workDir } from "@/lib/paths";

export const dynamic = "force-dynamic";

const PIPELINE_SERVICE_URL =
  process.env.PIPELINE_SERVICE_URL || "http://localhost:3001";
const PIPELINE_SECRET = process.env.PIPELINE_SECRET || "";

/**
 * POST /api/jobs/{id}/review   body: { excluded: string[] }
 *
 * Finish the manual panel-review step: persist the list of panels the user
 * dropped ("chap_XXX/frame_NNNNN.jpg"), drop a marker so the pipeline won't
 * pause here again, flip the job back to pending, and tell pipeline-service
 * to resume. The render phase reads work/excluded_frames.json and skips
 * those panels. Everything else (scrape, slices, OCR) is already on disk, so
 * the resume goes straight to render.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
    return NextResponse.json({ error: "bad id" }, { status: 400 });
  }

  const job = await db.job.findUnique({ where: { id } });
  if (!job) return NextResponse.json({ error: "job not found" }, { status: 404 });
  if (job.status !== "awaiting_review") {
    return NextResponse.json(
      { error: `job is ${job.status}, not awaiting_review` },
      { status: 409 }
    );
  }

  let excluded: string[] = [];
  try {
    const body = (await req.json()) as { excluded?: unknown };
    if (Array.isArray(body.excluded)) {
      excluded = body.excluded
        .filter((x): x is string => typeof x === "string")
        .map((x) => x.trim())
        .filter((x) => /^chap_\d+\/frame_\d+\.jpe?g$/i.test(x));
    }
  } catch {
    /* empty body = keep everything */
  }

  const wd = workDir(id);
  await fs.mkdir(wd, { recursive: true });
  await fs.writeFile(
    path.join(wd, "excluded_frames.json"),
    JSON.stringify({ excluded, reviewedAt: new Date().toISOString() }, null, 2)
  );
  await fs.writeFile(path.join(wd, ".review_done"), new Date().toISOString());

  await db.job.update({
    where: { id },
    data: {
      status: "pending",
      stage: null,
      message:
        excluded.length > 0
          ? `Review done — dropping ${excluded.length} panel(s), rendering…`
          : "Review done — rendering all panels…",
    },
  });

  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 5000);
    await fetch(`${PIPELINE_SERVICE_URL}/internal/start`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(PIPELINE_SECRET ? { authorization: `Bearer ${PIPELINE_SECRET}` } : {}),
      },
      body: JSON.stringify({ jobId: id }),
      signal: controller.signal,
    });
    clearTimeout(t);
  } catch {
    /* best-effort; the job is pending and can be retried from the UI */
  }

  return NextResponse.json({ ok: true, excluded: excluded.length });
}
