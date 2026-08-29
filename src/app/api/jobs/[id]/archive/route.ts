import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { db } from "@/lib/db";
import { outputVideoPath, fileExists } from "@/lib/paths";
import { archiveVideo, isArchiveConfigured } from "@/lib/archive";

export const dynamic = "force-dynamic";
export const maxDuration = 300; // 5 min — large uploads can take a while

/**
 * POST /api/jobs/{id}/archive
 *
 * Manually upload a completed job's video to cloud storage (Mega) and
 * delete the local file to free disk space.
 *
 * If the job is already archived, returns the existing archive info.
 * If the local file is already gone (archived + deleted), returns 409.
 */
export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    if (!id) {
      return NextResponse.json({ error: "Job id is required." }, { status: 400 });
    }

    const job = await db.job.findUnique({
      where: { id },
      select: {
        id: true,
        mangaTitle: true,
        status: true,
        outputDir: true,
        outputVideo: true,
        archiveProvider: true,
        archiveFileId: true,
      },
    });

    if (!job) {
      return NextResponse.json({ error: "Job not found." }, { status: 404 });
    }

    if (!isArchiveConfigured()) {
      return NextResponse.json(
        { error: "Mega is not configured. Set MEGA_EMAIL and MEGA_PASSWORD env vars." },
        { status: 400 }
      );
    }

    if (job.status !== "done") {
      return NextResponse.json(
        { error: `Job is not finished (status: ${job.status}). Wait for it to complete first.` },
        { status: 409 }
      );
    }

    // Already archived — return existing info.
    if (job.archiveProvider && job.archiveFileId) {
      return NextResponse.json({
        ok: true,
        alreadyArchived: true,
        provider: job.archiveProvider,
        fileId: job.archiveFileId,
      });
    }

    // Resolve local file path.
    const candidatePaths: string[] = [];
    if (job.outputDir && job.outputVideo) {
      candidatePaths.push(path.join(job.outputDir, job.outputVideo));
    }
    candidatePaths.push(outputVideoPath(job.id));

    let filePath: string | null = null;
    for (const p of candidatePaths) {
      if (await fileExists(p)) {
        filePath = p;
        break;
      }
    }

    if (!filePath) {
      return NextResponse.json(
        { error: "Local video file not found — it may have already been archived and deleted." },
        { status: 409 }
      );
    }

    const safeTitle = (job.mangaTitle || "recap").replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "");
    const filename = `${safeTitle}_recap.mp4`;

    // Upload to Mega.
    const result = await archiveVideo(filePath, filename);

    // Store archive info in DB.
    await db.job.update({
      where: { id },
      data: {
        archiveProvider: result.provider,
        archiveFileId: result.fileId,
      },
    });

    // Delete the local file to free disk space.
    try {
      await fs.unlink(filePath);
    } catch {
      // non-fatal — the file might be in use
    }

    return NextResponse.json({
      ok: true,
      archived: true,
      provider: result.provider,
      fileId: result.fileId,
    });
  } catch (err) {
    console.error("[archive] error:", err);
    return NextResponse.json(
      {
        error: "Failed to archive video.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}
