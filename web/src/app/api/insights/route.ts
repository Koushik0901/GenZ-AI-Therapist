import { NextResponse } from "next/server";

import { getInsights } from "@/lib/insights";
import { createServerSupabase } from "@/lib/supabase/server";

export async function GET() {
  const supabase = await createServerSupabase();
  const {
    data: { user },
  } = supabase ? await supabase.auth.getUser() : { data: { user: null } };

  return NextResponse.json(await getInsights(user?.id));
}
