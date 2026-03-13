import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";

export async function GET(request: Request) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get("code");

  if (!code) {
    return NextResponse.redirect(new URL("/auth?error=missing-code", request.url));
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return NextResponse.redirect(new URL("/auth?error=supabase-not-configured", request.url));
  }

  const { error } = await supabase.auth.exchangeCodeForSession(code);
  if (error) {
    return NextResponse.redirect(
      new URL(`/auth?error=${encodeURIComponent(error.message)}`, request.url),
    );
  }

  return NextResponse.redirect(new URL("/app/chat", request.url));
}
