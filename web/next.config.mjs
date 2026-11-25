// The browser talks only to Next; /api/* is proxied to the local DeltaQuery API.
const API = process.env.DELTAQUERY_API ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Local LLM compile + generation can exceed the dev proxy's ~30s default.
  experimental: { proxyTimeout: 300_000 },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};
export default nextConfig;
