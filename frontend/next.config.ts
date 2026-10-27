import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // STATIC_EXPORT=1 writes plain files to out/ for a free static host; the API URL comes from NEXT_PUBLIC_API_BASE.
  output: process.env.STATIC_EXPORT === "1" ? "export" : undefined,
};

export default config;
