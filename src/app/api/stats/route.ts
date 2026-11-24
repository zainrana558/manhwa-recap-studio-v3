import { NextResponse } from "next/server";
import { db } from "@/lib/db";

export const dynamic = "force-dynamic";

/** GET /api/stats — aggregate job metrics for the homepage stats bar. */
export async function GET() {
  try {
    const totalJobs = await db.job.count();
    const completedJobs = await db.job.count({ where: { status: "done" } });
    const activeJobs = await db.job.count({
      where: {
        status: {
          in: [
            "pending",
            "awaiting_review",
            "scraping",
            "transcribing",
            "translating",
            "rendering",
            "merging",
          ],
        },
      },
    });
    const totalChapters = await db.chapter.count();
    const totalImages = await db.job.aggregate({ _sum: { totalImages: true } });
    const archivedJobs = await db.job.count({ where: { status: "done", NOT: { archiveProvider: null } } });

    return NextResponse.json({
      totalJobs,
      completedJobs,
      activeJobs,
      totalChapters,
      totalImages: totalImages._sum.totalImages ?? 0,
      archivedJobs,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to load stats: ${message}` },
      { status: 500 }
    );
  }
}
