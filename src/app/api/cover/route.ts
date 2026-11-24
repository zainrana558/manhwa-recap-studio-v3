import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const raw = req.nextUrl.searchParams.get("url") || "";
  if (!/^https?:\/\//i.test(raw)) {
    return new NextResponse("Invalid url", { status: 400 });
  }
  try {
    const parsed = new URL(raw);
    const res = await fetch(raw, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        Referer: parsed.hostname.endsWith("pstatic.net")
          ? "https://www.webtoons.com/"
          : `${parsed.origin}/`,
      },
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) {
      return new NextResponse("Upstream error", { status: 502 });
    }
    const buf = await res.arrayBuffer();
    const ct = res.headers.get("content-type") || "image/jpeg";
    return new NextResponse(buf, {
      status: 200,
      headers: {
        "Content-Type": ct,
        "Cache-Control": "public, max-age=86400, immutable",
      },
    });
  } catch {
    return new NextResponse("Fetch failed", { status: 502 });
  }
}
