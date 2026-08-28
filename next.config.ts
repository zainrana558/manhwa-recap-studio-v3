import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
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
