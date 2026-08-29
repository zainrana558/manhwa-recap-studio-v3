"use client";

import { io, type Socket } from "socket.io-client";

let socket: Socket | null = null;

/**
 * Singleton socket.io client.
 *
 * Connection target (in priority order):
 *   1. NEXT_PUBLIC_PIPELINE_SERVICE_URL — set this when the frontend is deployed
 *      (e.g. to Vercel) and the pipeline-service runs elsewhere (e.g. your
 *      laptop exposed via Cloudflare Tunnel). The socket connects directly to
 *      that public URL with CORS enabled.
 *   2. Local dev / sandbox default — connect through the Caddy gateway using
 *      XTransformPort=3001 so the request is forwarded to the pipeline
 *      mini-service. Path MUST be "/".
 *
 * Reconnection settings are tuned for reliability:
 * - reconnectionDelay starts at 1000ms (was 1500ms) for faster reconnect
 * - reconnectionAttempts is Infinity (never give up)
 * - reconnectionDelayMax caps at 5000ms so it doesn't slow down too much
 *
 * pipeline-service's socket.io `allowRequest` (see index.ts) rejects any
 * connection whose `secret` query param doesn't match PIPELINE_SECRET —
 * added by a security-hardening pass that never updated this client to
 * actually send one, so every socket connection was silently refused.
 * This is a client component, so only a NEXT_PUBLIC_-prefixed var reaches
 * the browser (baked in at `bun run build` time, not read at runtime —
 * the build must happen with NEXT_PUBLIC_PIPELINE_SECRET set to the same
 * value pipeline-service uses for PIPELINE_SECRET).
 */
export function getSocket(): Socket {
  if (!socket) {
    const serviceUrl = process.env.NEXT_PUBLIC_PIPELINE_SERVICE_URL;
    const secret = process.env.NEXT_PUBLIC_PIPELINE_SECRET;

    if (serviceUrl) {
      // Deployed mode: connect directly to the remote pipeline-service URL.
      socket = io(serviceUrl, {
        path: "/",
        query: secret ? { secret } : undefined,
        transports: ["polling", "websocket"],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: Infinity,
        timeout: 10000,
      });
    } else {
      // Local dev / sandbox: route through the Caddy gateway.
      socket = io({
        path: "/",
        query: secret ? { XTransformPort: "3001", secret } : { XTransformPort: "3001" },
        transports: ["polling", "websocket"],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: Infinity,
        timeout: 10000,
      });
    }
  }
  return socket;
}

export function disconnectSocket() {
  if (socket) {
    socket.removeAllListeners();
    socket.disconnect();
    socket = null;
  }
}
