import { cookies } from "next/headers";
import { cache } from "react";
import { createServerClient } from "@supabase/ssr";

import { appEnv, isSupabaseConfigured } from "@/lib/env";

export const createServerSupabase = cache(async function createServerSupabase() {
  if (!isSupabaseConfigured) {
    return null;
  }

  const cookieStore = await cookies();

  return createServerClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet) {
        try {
          cookiesToSet.forEach(({ name, value, options }) => {
            cookieStore.set(name, value, options);
          });
        } catch {
          // Server Components can read cookies but may not be allowed to write them.
        }
      },
    },
  });
});
