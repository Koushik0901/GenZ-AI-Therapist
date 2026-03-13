import { existsSync } from "node:fs";
import path from "node:path";
import type { NextConfig } from "next";
import { config as loadEnv } from "dotenv";

const workspaceRoot = path.resolve(process.cwd(), "..");

for (const fileName of [".env.local", ".env"]) {
  const filePath = path.join(workspaceRoot, fileName);
  if (existsSync(filePath)) {
    loadEnv({ path: filePath, override: false, quiet: true });
  }
}

const nextConfig: NextConfig = {
  experimental: {
    optimizePackageImports: ["lucide-react", "recharts"],
  },
  env: {
    NEXT_PUBLIC_APP_URL:
      process.env.NEXT_PUBLIC_APP_URL ?? process.env.APP_URL ?? "http://localhost:3000",
    NEXT_PUBLIC_SUPABASE_URL:
      process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL ?? "",
    NEXT_PUBLIC_SUPABASE_ANON_KEY:
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
      process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY ??
      process.env.SUPABASE_ANON_KEY ??
      process.env.SUPABASE_PUBLISHABLE_DEFAULT_KEY ??
      "",
  },
};

export default nextConfig;
