function readEnv(...keys: string[]) {
  for (const key of keys) {
    const value = process.env[key];
    if (value) {
      return value;
    }
  }

  return "";
}

export const appEnv = {
  appUrl: readEnv("NEXT_PUBLIC_APP_URL", "APP_URL") || "http://localhost:3000",
  openRouterApiKey: readEnv("OPENROUTER_API_KEY"),
  openRouterModel: readEnv("OPENROUTER_MODEL") || "openrouter/openai/gpt-oss-120b",
  openRouterBaseUrl:
    readEnv("OPENROUTER_BASE_URL") || "https://openrouter.ai/api/v1",
  serperApiKey: readEnv("SERPER_API_KEY"),
  supabaseUrl: readEnv("NEXT_PUBLIC_SUPABASE_URL", "SUPABASE_URL"),
  supabaseAnonKey: readEnv(
    "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY",
    "SUPABASE_ANON_KEY",
    "SUPABASE_PUBLISHABLE_DEFAULT_KEY",
  ),
};

export const isSupabaseConfigured =
  Boolean(appEnv.supabaseUrl) && Boolean(appEnv.supabaseAnonKey);
export const isOpenRouterConfigured = Boolean(appEnv.openRouterApiKey);
export const isSerperConfigured = Boolean(appEnv.serperApiKey);

export function getMissingEnv() {
  return [
    ...(!appEnv.supabaseUrl ? ["NEXT_PUBLIC_SUPABASE_URL"] : []),
    ...(!appEnv.supabaseAnonKey ? ["NEXT_PUBLIC_SUPABASE_ANON_KEY"] : []),
    ...(!appEnv.openRouterApiKey ? ["OPENROUTER_API_KEY"] : []),
  ];
}
