import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  getSourceFromId,
  getSlugFromId,
  getChaptersForSource,
} from "@/lib/scrapers";
import { outputDir } from "@/lib/paths";
import { mapJob } from "@/lib/serialize";
import type { CreateJobInput, JobDetail } from "@/types/pipeline";

export const dynamic = "force-dynamic";

// URL of the pipeline mini-service (socket.io + Python pipeline).
// - Local dev / sandbox: defaults to http://localhost:3001
// - Vercel / production: set PIPELINE_SERVICE_URL to your laptop's public
//   tunnel URL (e.g. https://your-laptop.trycloudflare.com)
const PIPELINE_SERVICE_URL = process.env.PIPELINE_SERVICE_URL || "http://localhost:3001";

/** Fire-and-forget POST to the pipeline service with a short timeout. */
async function notifyPipeline(
  path: string,
  body: Record<string, unknown>
): Promise<void> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    await fetch(`${PIPELINE_SERVICE_URL}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch {
    // Pipeline service is allowed to be down; job remains in DB and can be started later.
  } finally {
    clearTimeout(timeout);
  }
}

/** GET /api/jobs — list all jobs (newest first). */
export async function GET() {
  try {
    const jobs = await db.job.findMany({
      orderBy: { createdAt: "desc" },
      include: { _count: { select: { chapters: true } } },
    });

    const jobs_: JobDetail[] = jobs.map((j) =>
      mapJob({
        ...j,
        // List view: omit chapter detail, but expose totalChapters for UI.
        chapters: [],
      })
    );

    return NextResponse.json({ jobs: jobs_ });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to list jobs: ${message}` },
      { status: 500 }
    );
  }
}

/** POST /api/jobs — create a new job and kick off the pipeline. */
export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as CreateJobInput;

    if (!body?.mangaId || !body?.mangaTitle) {
      return NextResponse.json(
        { error: "mangaId and mangaTitle are required." },
        { status: 400 }
      );
    }

    const language = body.language || "en";
    const chapterLimit = Math.max(0, body.chapterLimit || 0);
    const chapterIds = body.chapterIds ?? null;
    const voice = body.voice || "en-US-AndrewNeural";
    const translate = body.translate === true;
    const bgmPath = body.bgmPath ?? null;
    const useBgm = body.useBgm !== false;

    // 1) Fetch chapter list from the appropriate scraping source.
    const source = getSourceFromId(body.mangaId);
    if (!source) {
      return NextResponse.json(
        { error: `Invalid manga id: ${body.mangaId}. Expected prefix mh-, ff-, or wt-.` },
        { status: 400 }
      );
    }
    const slug = getSlugFromId(body.mangaId);

    let chapterFeed: Array<{
      id: string;
      chapterNum: string;
      title: string | null;
    }> = [];
    try {
      const scraped = await getChaptersForSource(source, slug);
      chapterFeed = scraped.map((c) => ({
        id: c.id,
        chapterNum: c.chapterNum,
        title: c.title,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      return NextResponse.json(
        { error: `Failed to fetch chapters from ${source}: ${message}` },
        { status: 502 }
      );
    }

    // Apply chapter selection: specific IDs take priority, then chapterLimit, then all.
    let selected: typeof chapterFeed;
    if (chapterIds && chapterIds.length > 0) {
      const idSet = new Set(chapterIds);
      selected = chapterFeed.filter((c) => idSet.has(c.id));
    } else {
      selected =
        chapterLimit > 0 ? chapterFeed.slice(0, chapterLimit) : chapterFeed;
    }

    if (selected.length === 0) {
      return NextResponse.json(
        {
          error: `No chapters found for manga "${body.mangaTitle}" on MangaHere.`,
        },
        { status: 400 }
      );
    }

    const totalImages = 0; // MangaHere page count not known until scraping

    // 2) Create Job + Chapter rows.
    const job = await db.job.create({
      data: {
        mangaId: body.mangaId,
        mangaTitle: body.mangaTitle,
        coverUrl: body.coverUrl ?? null,
        language,
        sourceLang: null, // pipeline service fills this in based on actual scraped chapters
        status: "pending",
        progress: 0,
        stage: null,
        message: "Job created, waiting for pipeline to start.",
        totalChapters: selected.length,
        doneChapters: 0,
        totalImages,
        doneImages: 0,
        outputDir: null,
        outputVideo: null,
        error: null,
        groqKey: body.groqKey ?? null,
        geminiKey: body.geminiKey ?? null,
        openRouterKey: body.openRouterKey ?? null,
        zhipuKey: body.zhipuKey ?? null,
        siliconFlowKey: body.siliconFlowKey ?? null,
        openaiKey: body.openaiKey ?? null,
        megaEmail: body.megaEmail ?? null,
        megaPassword: body.megaPassword ?? null,
        r2AccountId: body.r2AccountId ?? null,
        r2AccessKeyId: body.r2AccessKeyId ?? null,
        r2SecretAccessKey: body.r2SecretAccessKey ?? null,
        r2Bucket: body.r2Bucket ?? null,
        autoArchive: body.autoArchive ?? false,
        voice,
        chapterLimit,
        translate,
        bgmPath,
        useBgm,
        chapters: {
          create: selected.map((c, i) => ({
            index: i + 1,
            mangadexId: c.id, // chapter id from the scraping source
            chapterNum: c.chapterNum,
            title: c.title,
            language: "en",
            pageCount: 0,
            folder: `chapter_${String(i + 1).padStart(3, "0")}`,
            status: "pending",
          })),
        },
        logs: {
          create: {
            level: "info",
            stage: "search",
            message: `Job created for "${body.mangaTitle}" — ${selected.length} chapter(s), ${totalImages} image(s).`,
          },
        },
      },
      include: { chapters: { orderBy: { index: "asc" } } },
    });

    // Mark outputDir now that we have a jobId.
    await db.job.update({
      where: { id: job.id },
      data: { outputDir: outputDir(job.id) },
    });

    const detail = mapJob({
      ...job,
      outputDir: outputDir(job.id),
    });

    // 3) Fire-and-forget: tell the pipeline service to start the job.
    //    Non-blocking — pipeline runs in the background and pushes progress over WS.
    void notifyPipeline("/internal/start", { jobId: job.id });

    return NextResponse.json({ job: detail }, { status: 201 });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to create job: ${message}` },
      { status: 500 }
    );
  }
}
