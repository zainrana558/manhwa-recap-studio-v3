import { NextResponse } from "next/server";
import { db } from "@/lib/db";

export const dynamic = "force-dynamic";

const DEMO_JOBS = [
  {
    mangaId: "a1c7c817-4e59-43b7-9365-09675a149a6f",
    mangaTitle: "Solo Leveling",
    coverUrl: "https://uploads.mangadex.org/covers/a1c7c817-4e59-43b7-9365-09675a149a6f/cover.jpg",
    status: "done",
    progress: 100,
    stage: "merged",
    message: "Video rendered successfully",
    totalChapters: 5,
    doneChapters: 5,
    totalImages: 847,
    doneImages: 847,
    outputVideo: "solo-leveling-ch1-5-recap.mp4",
    voice: "en-US-AndrewNeural",
    chapterLimit: 5,
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
  },
  {
    mangaId: "32d76d19-8a05-4db0-9fc2-e0b0648fe9d0",
    mangaTitle: "Tower of God",
    coverUrl: "https://uploads.mangadex.org/covers/32d76d19-8a05-4db0-9fc2-e0b0648fe9d0/cover.jpg",
    status: "done",
    progress: 100,
    stage: "merged",
    message: "Video rendered successfully",
    totalChapters: 3,
    doneChapters: 3,
    totalImages: 412,
    doneImages: 412,
    outputVideo: "tower-of-god-ch1-3-recap.mp4",
    voice: "en-US-JennyNeural",
    chapterLimit: 3,
    createdAt: new Date(Date.now() - 5 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 5 * 60 * 60 * 1000),
  },
  {
    mangaId: "7f0c7f8f-7c1f-45b8-a42f-5c4523e8d4e4",
    mangaTitle: "The Beginning After The End",
    coverUrl: "https://uploads.mangadex.org/covers/7f0c7f8f-7c1f-45b8-a42f-5c4523e8d4e4/cover.jpg",
    status: "done",
    progress: 100,
    stage: "merged",
    message: "Video rendered and archived to Mega",
    totalChapters: 10,
    doneChapters: 10,
    totalImages: 2103,
    doneImages: 2103,
    outputVideo: "tbate-ch1-10-recap.mp4",
    archiveProvider: "mega",
    voice: "en-US-AndrewNeural",
    chapterLimit: 10,
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 24 * 60 * 60 * 1000),
  },
  {
    mangaId: "763b4b2e-d59a-4f9a-9e30-5a4847b70a6e",
    mangaTitle: "Omniscient Reader",
    coverUrl: "https://uploads.mangadex.org/covers/763b4b2e-d59a-4f9a-9e30-5a4847b70a6e/cover.jpg",
    status: "done",
    progress: 100,
    stage: "merged",
    message: "Video rendered successfully",
    totalChapters: 2,
    doneChapters: 2,
    totalImages: 290,
    doneImages: 290,
    outputVideo: "omniscient-reader-ch1-2-recap.mp4",
    voice: "en-GB-RyanNeural",
    chapterLimit: 2,
    createdAt: new Date(Date.now() - 48 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 48 * 60 * 60 * 1000),
  },
  {
    mangaId: "f4c73680-dc7e-4d32-85c5-e0b0648fe9d1",
    mangaTitle: "Nano Machine",
    coverUrl: "https://uploads.mangadex.org/covers/f4c73680-dc7e-4d32-85c5-e0b0648fe9d1/cover.jpg",
    status: "rendering",
    progress: 72,
    stage: "rendering",
    message: "Rendering chapter 4/5 — merging panels with audio",
    totalChapters: 5,
    doneChapters: 3,
    totalImages: 510,
    doneImages: 380,
    voice: "en-US-GuyNeural",
    chapterLimit: 5,
    createdAt: new Date(Date.now() - 0.5 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 0.5 * 60 * 60 * 1000),
  },
];

// Demo-data seeding/wiping endpoint — for local development only. Unguarded,
// DELETE below wipes every job (and all its chapters/logs) with no auth, so
// this must never be reachable in a real deployment.
function devOnlyGuard(): NextResponse | null {
  if (process.env.NODE_ENV === "production") {
    return NextResponse.json({ error: "Not available in production." }, { status: 403 });
  }
  return null;
}

export async function POST() {
  const blocked = devOnlyGuard();
  if (blocked) return blocked;
  try {
    const existing = await db.job.count();
    if (existing > 0) {
      return NextResponse.json({ message: "Database already has data. Skipping seed.", existing });
    }

    let created = 0;
    for (const jobData of DEMO_JOBS) {
      await db.job.create({ data: jobData });
      created++;
    }

    return NextResponse.json({ message: `Seeded ${created} demo jobs`, created });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: `Seed failed: ${message}` }, { status: 500 });
  }
}

export async function DELETE() {
  const blocked = devOnlyGuard();
  if (blocked) return blocked;
  try {
    const { count } = await db.job.deleteMany();
    await db.chapter.deleteMany();
    await db.jobLog.deleteMany();
    return NextResponse.json({ message: `Deleted ${count} jobs and related data` });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: `Clear failed: ${message}` }, { status: 500 });
  }
}