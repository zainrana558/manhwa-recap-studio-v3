import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // The /api/download/[id] route does `fs.open(<db-driven path>)` — Next 16's
  // Turbopack file-tracer can't bound that, decides the route may read ANY
  // file under the project root, and copies ~233k files (node_modules, the
  // whole pipeline/ tree, generated .mp4s, datasets) into .next/standalone.
  // That turned "Finalizing page optimization" into a 6 GB, many-minute step
  // and left the standalone dir containing the entire repo. Exclude the heavy
  // non-code trees from every route's trace.
  outputFileTracingExcludes: {
    "*": [
      "pipeline/**",
      "mini-services/*/models/**",
      "mini-services/*/.venv/**",
      ".venv/**",
      "data/**",
      "db/**",
      "logs/**",
      "examples/**",
      "agent-ctx/**",
      "**/*.mp4",
      "**/*.onnx",
      "**/*.log",
      "**/*.tar",
      "**/*.zip",
      ".next/cache/**",
      "node_modules/@next/swc-*/**",
    ],
  },
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  allowedDevOrigins: ["*.daytonaproxy01.net"],
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "uploads.mangadex.org" },
      { protocol: "https", hostname: "mangadex.org" },
      { protocol: "https", hostname: "***.mangahere.cc" },
      { protocol: "https", hostname: "***.fanfox.net" },
      { protocol: "https", hostname: "***.webtoons.com" },
      { protocol: "https", hostname: "***.asurascans.com" },
      { protocol: "https", hostname: "cdn.mangaimage.co" },
      { protocol: "https", hostname: "***.mangapill.com" },
      { protocol: "https", hostname: "***.toonily.com" },
    ],
  },
};

export default nextConfig;
