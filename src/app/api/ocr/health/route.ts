import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/**
 * GET /api/ocr/health
 *
 * Proxies to the PaddleOCR mini-service's /health endpoint.
 * Returns the OCR service status and model info.
 */
const OCR_SERVICE_URL = process.env.OCR_SERVICE_URL || "http://localhost:3002";

export async function GET() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    try {
      const res = await fetch(`${OCR_SERVICE_URL}/health`, {
        signal: controller.signal,
      });

      if (!res.ok) {
        return NextResponse.json(
          { ready: false, model: "unavailable", status: res.status },
          { status: 200 }
        );
      }

      const data = await res.json();
      return NextResponse.json(data, { status: 200 });
    } finally {
      clearTimeout(timeout);
    }
  } catch {
    return NextResponse.json(
      { ready: false, model: "unavailable" },
      { status: 200 }
    );
  }
}
