import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import crypto from "crypto";
import { DATA_DIR } from "@/lib/paths";

export const dynamic = "force-dynamic";

// Was hardcoded to process.cwd()/data/bgm, which silently diverged from
// DATA_DIR whenever the DATA_DIR env var pointed elsewhere (e.g. an external
// volume) — uploads would land somewhere the pipeline-service never looks.
const BGM_DIR = path.join(DATA_DIR, "bgm");

/** GET /api/bgm — list all available BGM tracks. */
export async function GET() {
  try {
    await fs.mkdir(BGM_DIR, { recursive: true });
    const entries = await fs.readdir(BGM_DIR);
    const tracks = await Promise.all(
      entries
        .filter((f) => /\.(mp3|wav|ogg|m4a)$/i.test(f))
        .map(async (f) => {
          const stat = await fs.stat(path.join(BGM_DIR, f));
          return {
            name: f,
            size: stat.size,
            isDefault: f === "default_cinematic.mp3",
          };
        })
    );
    return NextResponse.json({ tracks });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to list BGM" },
      { status: 500 }
    );
  }
}

/** POST /api/bgm — upload a new BGM track (multipart form-data, field "file"). */
export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");
    if (!file || !(file instanceof File)) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Validate it's an audio file
    if (file.type && !file.type.startsWith("audio/") && !file.name.match(/\.(mp3|wav|ogg|m4a)$/i)) {
      return NextResponse.json({ error: "File must be an audio file (mp3, wav, ogg, m4a)" }, { status: 400 });
    }

    await fs.mkdir(BGM_DIR, { recursive: true });

    // Generate a safe filename
    const ext = path.extname(file.name).toLowerCase() || ".mp3";
    const safeName = file.name
      .replace(/[^a-zA-Z0-9._-]/g, "_")
      .replace(/\.(mp3|wav|ogg|m4a)$/i, "")
      .slice(0, 60);
    const hash = crypto.randomBytes(4).toString("hex");
    const filename = `${safeName}_${hash}${ext}`;

    const buf = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(path.join(BGM_DIR, filename), buf);

    return NextResponse.json({ name: filename, size: buf.length });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Upload failed" },
      { status: 500 }
    );
  }
}

/** DELETE /api/bgm?name={filename} — delete a BGM track (except the default). */
export async function DELETE(req: NextRequest) {
  try {
    const name = req.nextUrl.searchParams.get("name");
    if (!name) {
      return NextResponse.json({ error: "Name is required" }, { status: 400 });
    }
    if (name === "default_cinematic.mp3") {
      return NextResponse.json({ error: "Cannot delete the default track" }, { status: 400 });
    }
    // Prevent path traversal
    if (name.includes("/") || name.includes("..")) {
      return NextResponse.json({ error: "Invalid filename" }, { status: 400 });
    }
    const filePath = path.join(BGM_DIR, name);
    await fs.unlink(filePath);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Delete failed" },
      { status: 500 }
    );
  }
}
