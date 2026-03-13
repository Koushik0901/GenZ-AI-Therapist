"use server";

import { redirect } from "next/navigation";

import { appEnv, isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

export async function signInAction(formData: FormData) {
  if (!isSupabaseConfigured) {
    redirect("/auth?error=supabase-not-configured");
  }

  const email = String(formData.get("email") ?? "").trim();
  if (!email) {
    redirect("/auth?error=missing-email");
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    redirect("/auth?error=supabase-client-unavailable");
  }

  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: {
      emailRedirectTo: `${appEnv.appUrl}/auth/callback`,
    },
  });

  if (error) {
    redirect(`/auth?error=${encodeURIComponent(error.message)}`);
  }

  redirect(`/auth?sent=1&email=${encodeURIComponent(email)}`);
}

export async function signOutAction() {
  const supabase = await createServerSupabase();
  if (supabase) {
    await supabase.auth.signOut();
  }

  redirect("/auth");
}
