import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const {
      mangaTitle,
      mangaId,
      language,
      sourceLang,
      voice,
      chapterLimit,
      translate,
      useBgm,
      bgmPath,
      totalChapters,
    } = body;

    if (!mangaId || !mangaTitle) {
      return NextResponse.json(
        { error: "mangaId and mangaTitle are required." },
        { status: 400 },
      );
    }

    const config = {
      mangaId,
      mangaTitle,
      language: language || "en",
      sourceLang: sourceLang ?? null,
      voice: voice || "en-US-ChristopherNeural",
      chapterLimit: chapterLimit ?? 0,
      translate: translate !== false,
      useBgm: useBgm !== false,
      bgmPath: bgmPath ?? null,
      totalChapters: totalChapters ?? 0,
    };

    return NextResponse.json({ config });
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body." },
      { status: 400 },
    );
  }
}
