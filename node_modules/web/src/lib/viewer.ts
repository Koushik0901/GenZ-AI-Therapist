import { cache } from "react";

import { isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

export const getViewer = cache(async function getViewer() {
  if (!isSupabaseConfigured) {
    return {
      supabase: null,
      user: null,
    };
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return {
      supabase: null,
      user: null,
    };
  }

  const {
    data: { user },
  } = await supabase.auth.getUser();

  return {
    supabase,
    user,
  };
});
