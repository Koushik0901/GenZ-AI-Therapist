import { NextResponse } from "next/server";
import { z } from "zod";

import { isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

const checkInSchema = z.object({
  mood: z.number().min(0).max(100),
  energy: z.number().min(0).max(100),
  stress: z.number().min(0).max(100),
  note: z.string().max(2000).optional(),
});

export async function POST(request: Request) {
  const parsed = checkInSchema.safeParse(await request.json());

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
      .from("daily_checkins")
      .insert({
        user_id: user.id,
        mood_score: parsed.data.mood,
        energy_score: parsed.data.energy,
        stress_score: parsed.data.stress,
        note: parsed.data.note ?? null,
      })
      .select("id,mood_score,energy_score,stress_score,note,created_at")
      .single();

    if (error) {
      return NextResponse.json({ error: "Check-in save failed" }, { status: 500 });
    }

    return NextResponse.json({
      ok: true,
      savedAt: data.created_at,
      payload: {
        mood: data.mood_score,
        energy: data.energy_score,
        stress: data.stress_score,
        note: data.note ?? undefined,
      },
    });
  }

  return NextResponse.json({
    ok: true,
    savedAt: new Date().toISOString(),
    payload: parsed.data,
  });
}
