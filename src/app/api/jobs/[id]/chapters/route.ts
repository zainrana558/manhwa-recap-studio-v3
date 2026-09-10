import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { outputDir } from "@/lib/paths";

export const dynamic = "force-dynamic";

/** GET /api/jobs/{id}/chapters — scrubber markers written by the merge step. */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
    return NextResponse.json({ error: "bad id" }, { status: 400 });
  }
  try {
    const raw = await fs.readFile(path.join(outputDir(id), "chapters.json"), "utf8");
    return new NextResponse(raw, {
      headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=60" },
    });
  } catch {
    return NextResponse.json({ chapters: [], total: 0 });
  }
}
