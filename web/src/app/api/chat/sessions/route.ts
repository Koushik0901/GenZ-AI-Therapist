import { NextResponse } from "next/server";

import { getSessionSummaries } from "@/lib/chat";
import { isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

export async function GET() {
  if (!isSupabaseConfigured) {
    return NextResponse.json({
      ok: true,
      demoMode: true,
      sessions: [],
    });
  }

  const supabase = await createServerSupabase();
  const {
    data: { user },
  } = supabase ? await supabase.auth.getUser() : { data: { user: null } };

  if (!supabase || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const sessions = await getSessionSummaries(user.id);

  return NextResponse.json({
    ok: true,
    sessions,
  });
}
