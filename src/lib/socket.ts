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
 */
export function getSocket(): Socket {
  if (!socket) {
    const serviceUrl = process.env.NEXT_PUBLIC_PIPELINE_SERVICE_URL;

    if (serviceUrl) {
      // Deployed mode: connect directly to the remote pipeline-service URL.
      socket = io(serviceUrl, {
        path: "/",
        transports: ["polling", "websocket"],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 50,
        timeout: 10000,
      });
    } else {
      // Local dev / sandbox: route through the Caddy gateway.
      socket = io({
        path: "/",
        query: { XTransformPort: "3001" },
        transports: ["polling", "websocket"],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 50,
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
