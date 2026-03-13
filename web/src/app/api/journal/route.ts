import { NextResponse } from "next/server";
import { z } from "zod";

import { isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

const journalSchema = z.object({
  title: z.string().min(1).max(120),
  body: z.string().min(1).max(10000),
  mood: z.string().min(1).max(40).optional(),
});

export async function POST(request: Request) {
  const parsed = journalSchema.safeParse(await request.json());

  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
  }

  if (isSupabaseConfigured) {
    const supabase = await createServerSupabase();
    const {
      data: { user },
    } = supabase ? await supabase.auth.getUser() : { data: { user: null } };

    if (!supabase || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { data, error } = await supabase
      .from("journal_entries")
      .insert({
        user_id: user.id,
        title: parsed.data.title,
        body: parsed.data.body,
        mood: parsed.data.mood ?? null,
      })
      .select("id,title,body,mood,created_at")
      .single();

    if (error) {
      return NextResponse.json({ error: "Journal save failed" }, { status: 500 });
    }

    return NextResponse.json({
      ok: true,
      savedAt: data.created_at,
      payload: {
        title: data.title,
        body: data.body,
        mood: data.mood ?? undefined,
      },
    });
  }

  return NextResponse.json({
    ok: true,
    savedAt: new Date().toISOString(),
    payload: parsed.data,
  });
}
