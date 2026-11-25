import { NextRequest, NextResponse } from "next/server";
import sharp from "sharp";

export const dynamic = "force-dynamic";

interface CacheEntry {
  buf: Buffer;
  ct: string;
}

const MAX_CACHE = 240;
const coverCache = new Map<string, CacheEntry>();

function getCached(key: string): CacheEntry | undefined {
  const hit = coverCache.get(key);
  if (!hit) return undefined;
  // Refresh LRU order
  coverCache.delete(key);
  coverCache.set(key, hit);
  return hit;
}

function setCached(key: string, entry: CacheEntry) {
  if (coverCache.size >= MAX_CACHE) {
    const oldest = coverCache.keys().next().value;
    if (oldest) coverCache.delete(oldest);
  }
  coverCache.set(key, entry);
}

export async function GET(req: NextRequest) {
  const raw = req.nextUrl.searchParams.get("url") || "";
  const wParam = parseInt(req.nextUrl.searchParams.get("w") || "420", 10);
  const targetWidth = Number.isFinite(wParam) ? Math.max(160, Math.min(800, wParam)) : 420;

  if (!/^https?:\/\//i.test(raw)) {
    return new NextResponse("Invalid url", { status: 400 });
  }

  const cacheKey = `${targetWidth}:${raw}`;
  const cached = getCached(cacheKey);
  if (cached) {
    return new NextResponse(new Uint8Array(cached.buf), {
      status: 200,
      headers: {
        "Content-Type": cached.ct,
        "Cache-Control": "public, max-age=604800, immutable",
        "X-Cover-Cache": "HIT",
      },
    });
  }

  try {
    const parsed = new URL(raw);
    const res = await fetch(raw, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        Referer: parsed.hostname.endsWith("pstatic.net")
          ? "https://www.webtoons.com/"
          : parsed.hostname.endsWith("mangadex.org")
            ? "https://mangadex.org/"
            : `${parsed.origin}/`,
      },
      signal: AbortSignal.timeout(9000),
    });
    if (!res.ok) {
      return new NextResponse("Upstream error", { status: 502 });
    }
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (!ct.startsWith("image/") && !ct.includes("octet-stream")) {
      return new NextResponse("Not an image", { status: 404 });
    }
    const rawBuf = Buffer.from(await res.arrayBuffer());
    if (rawBuf.byteLength < 100) {
      return new NextResponse("Empty image", { status: 404 });
    }

    let outBuf = rawBuf;
    let outCt = ct.startsWith("image/") ? ct : "image/jpeg";
    try {
      outBuf = await sharp(rawBuf)
        .resize({ width: targetWidth, withoutEnlargement: true })
        .webp({ quality: 80, effort: 2 })
        .toBuffer();
      outCt = "image/webp";
    } catch {
      // Fallback to raw image buffer if sharp cannot decode an exotic format
    }

    setCached(cacheKey, { buf: outBuf, ct: outCt });
    return new NextResponse(new Uint8Array(outBuf), {
      status: 200,
      headers: {
        "Content-Type": outCt,
        "Cache-Control": "public, max-age=604800, immutable",
        "X-Cover-Cache": "MISS",
      },
    });
  } catch {
    return new NextResponse("Fetch failed", { status: 502 });
  }
}
