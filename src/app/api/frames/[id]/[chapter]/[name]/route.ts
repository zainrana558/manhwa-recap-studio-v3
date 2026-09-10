import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { workDir } from "@/lib/paths";

export const dynamic = "force-dynamic";

/** GET /api/frames/{id}/{chapter}/{name} — stream one sliced panel frame. */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string; chapter: string; name: string }> }
) {
  const { id, chapter, name } = await params;
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(id || "")) {
    return NextResponse.json({ error: "bad id" }, { status: 400 });
  }
  const chNum = parseInt(chapter, 10);
  if (!Number.isFinite(chNum) || chNum < 1 || chNum > 9999) {
    return NextResponse.json({ error: "bad chapter" }, { status: 400 });
  }
  if (!/^frame_\d{1,7}\.jpe?g$/i.test(name || "")) {
    return NextResponse.json({ error: "bad name" }, { status: 400 });
  }

  const file = path.join(
    workDir(id),
    "temp_slices",
    `chap_${String(chNum).padStart(3, "0")}`,
    name
  );
  try {
    const buf = await fs.readFile(file);
    return new NextResponse(new Uint8Array(buf), {
      headers: {
        "Content-Type": "image/jpeg",
        // frames are immutable once written; the list route drives refresh
        "Cache-Control": "public, max-age=300",
      },
    });
  } catch {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
}
