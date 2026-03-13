import { NextResponse } from "next/server";
import { z } from "zod";

import { isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

const paramsSchema = z.object({
  id: z.string().uuid(),
});

const patchSchema = z.object({
  status: z.enum(["pending", "approved", "hidden"]),
});

export async function PATCH(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  if (!isSupabaseConfigured) {
    return NextResponse.json({ ok: true, demoMode: true });
  }

  const parsedParams = paramsSchema.safeParse(await context.params);
  const parsedBody = patchSchema.safeParse(await request.json());

  if (!parsedParams.success || !parsedBody.success) {
    return NextResponse.json(
      {
        error: {
          params: parsedParams.success ? null : parsedParams.error.flatten(),
          body: parsedBody.success ? null : parsedBody.error.flatten(),
        },
      },
      { status: 400 },
    );
  }

  const supabase = await createServerSupabase();
  const {
    data: { user },
  } = supabase ? await supabase.auth.getUser() : { data: { user: null } };

  if (!supabase || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { data, error } = await supabase
    .from("memory_items")
    .update({ status: parsedBody.data.status })
    .eq("id", parsedParams.data.id)
    .eq("user_id", user.id)
    .select("id,status")
    .maybeSingle();

  if (error) {
    return NextResponse.json({ error: "Memory update failed" }, { status: 500 });
  }

  if (!data) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return NextResponse.json({
    ok: true,
    payload: data,
  });
}
