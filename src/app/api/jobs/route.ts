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
// pipeline-service's /internal/* endpoints require this (checkAuth in
// index.ts, added by a security-hardening pass that never wired the
// Next.js side to send it). Without it every call here got a 401,
// notifyPipeline returned false, and the job was reported as "Pipeline
// service is not running" even when the service was up and healthy.
//
// start.sh writes the shared secret to $PROJECT_ROOT/.pipeline-secret and
// exports PIPELINE_SECRET for both processes — but the env var doesn't always
// survive the setsid/systemd hop to the standalone server. Fall back to the
// file (same value pipeline-service uses) so the two sides can't drift.
function resolvePipelineSecret(): string {
  const env = process.env.PIPELINE_SECRET;
  if (env) return env;
  try {
    const root = process.env.PROJECT_ROOT || process.cwd();
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const fs = require("fs") as typeof import("fs");
    const p = `${root}/.pipeline-secret`;
    if (fs.existsSync(p)) return fs.readFileSync(p, "utf8").trim();
  } catch {
    /* ignore */
  }
  return "";
}
const PIPELINE_SECRET = resolvePipelineSecret();

/**
 * POST to the pipeline service with a short timeout.
 * Returns true if the pipeline accepted the request, false if unreachable
 * or rejected (auth failure, bad request, etc. — see server logs for why).
 */
async function notifyPipeline(
  path: string,
  body: Record<string, unknown>
): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const res = await fetch(`${PIPELINE_SERVICE_URL}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(PIPELINE_SECRET ? { authorization: `Bearer ${PIPELINE_SECRET}` } : {}),
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) {
      const errBody = await res.text().catch(() => "");
      console.warn(`[notifyPipeline] ${path} -> HTTP ${res.status}: ${errBody.slice(0, 200)}`);
    }
    return res.ok;
  } catch (err) {
    clearTimeout(timeout);
    console.warn(`[notifyPipeline] ${path} unreachable: ${err instanceof Error ? err.message : String(err)}`);
    return false;
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
    // Narration is on by default; pass narrate:false to speak the raw
    // transcribed panel text verbatim instead of an LLM-rewritten recap.
    const narrate = body.narrate !== false;
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

    // M25 FIX: Sort chapters by chapter number before selecting.
    chapterFeed.sort((a, b) => {
      const na = parseFloat(a.chapterNum) || 0;
      const nb = parseFloat(b.chapterNum) || 0;
      return na - nb;
    });

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
          // H35 FIX: Use actual source name instead of hardcoded "MangaHere"
          error: `No chapters found for manga "${body.mangaTitle}" on ${source}.`,
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
        narrate,
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

    // 3) Tell the pipeline service to start the job.
    const pipelineOk = await notifyPipeline("/internal/start", { jobId: job.id });

    if (!pipelineOk) {
      // Pipeline service call failed — could genuinely be down, OR up but
      // rejecting the request (e.g. PIPELINE_SECRET mismatch/unset — see
      // the [notifyPipeline] warning logged just above with the real HTTP
      // status/error). "not running" was misleading callers into
      // restarting a service that was actually fine. Also: this repo's
      // launcher is start.sh, not start-services.sh — check
      // logs/pipeline.log either way before assuming a restart is needed.
      const hint = "Pipeline service call failed — check logs/pipeline.log (server not running, or PIPELINE_SECRET misconfigured between it and the web app). Restart with: bash start.sh";
      await db.job.update({
        where: { id: job.id },
        data: {
          status: "error",
          error: hint,
          message: "Pipeline service unreachable",
        },
      });
      await db.jobLog.create({
        data: {
          jobId: job.id,
          level: "error",
          stage: "start",
          message: hint,
        },
      });
    }

    return NextResponse.json({ job: detail }, { status: 201 });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to create job: ${message}` },
      { status: 500 }
    );
  }
}
