"use client";

import { createBrowserClient } from "@supabase/ssr";

import { appEnv, isSupabaseConfigured } from "@/lib/env";

export function createBrowserSupabase() {
  if (!isSupabaseConfigured) {
    return null;
  }

  return createBrowserClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey);
}
