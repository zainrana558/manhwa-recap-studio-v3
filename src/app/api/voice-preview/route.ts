import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

/**
 * GET /api/voice-preview?voice={voiceId}
 *
 * Proxies to the pipeline mini-service's /preview/voice endpoint, which
 * generates (or serves from cache) a short edge-tts preview MP3.
 *
 * Why proxy? edge-tts requires Python, which isn't available on serverless
 * hosts like Vercel. The mini-service (which has Python) does the actual
 * generation. This keeps the Next.js layer Python-free and deployable anywhere.
 *
 * Connection target:
 *   - PIPELINE_SERVICE_URL env var (set on Vercel to your laptop's tunnel URL)
 *   - defaults to http://localhost:3001 for local dev / sandbox
 */
const PIPELINE_SERVICE_URL =
  process.env.PIPELINE_SERVICE_URL || "http://localhost:3001";

// edge-tts IDs ("en-US-AndrewNeural") or local Kokoro IDs ("am_michael", "af_bella").
const VOICE_ID_RE = /^([a-z]{2}-[A-Z]{2}-[A-Za-z0-9]+Neural|[abhijpef][fm]_[a-z][a-z_]*)$/;

export async function GET(req: NextRequest) {
  const voice = req.nextUrl.searchParams.get("voice");

  if (!voice || !VOICE_ID_RE.test(voice)) {
    return NextResponse.json(
      { error: "Invalid or missing 'voice' parameter. Expected format like 'en-US-AndrewNeural'." },
      { status: 400 }
    );
  }

  try {
    const upstream = `${PIPELINE_SERVICE_URL}/preview/voice?voice=${encodeURIComponent(voice)}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);

    try {
      const upstreamRes = await fetch(upstream, { signal: controller.signal });

      if (!upstreamRes.ok) {
        const text = await upstreamRes.text().catch(() => "");
        return NextResponse.json(
          { error: "Pipeline service could not generate the preview.", detail: text.slice(0, 200) },
          { status: upstreamRes.status }
        );
      }

      const audioBuffer = Buffer.from(await upstreamRes.arrayBuffer());
      return new NextResponse(audioBuffer, {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "Content-Length": String(audioBuffer.length),
          "Cache-Control": "public, max-age=86400, immutable",
        },
      });
    } finally {
      clearTimeout(timeout);
    }
  } catch (err) {
    const isAbort = err instanceof Error && err.name === "AbortError";
    return NextResponse.json(
      {
        error: isAbort
          ? "Voice preview timed out — the pipeline service may be busy or offline."
          : "Failed to reach the pipeline service for voice preview.",
      },
      { status: isAbort ? 504 : 502 }
    );
  }
}
