"use server";

import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { appEnv, isSupabaseConfigured } from "@/lib/env";
import { createServerSupabase } from "@/lib/supabase/server";

async function getAuthRedirectBaseUrl() {
  const requestHeaders = await headers();
  const origin = requestHeaders.get("origin");
  if (origin) {
    return origin;
  }

  const forwardedProto = requestHeaders.get("x-forwarded-proto");
  const forwardedHost = requestHeaders.get("x-forwarded-host");
  if (forwardedProto && forwardedHost) {
    return `${forwardedProto}://${forwardedHost}`;
  }

  const host = requestHeaders.get("host");
  if (host) {
    const protocol = host.includes("localhost") ? "http" : "https";
    return `${protocol}://${host}`;
  }

  return appEnv.appUrl;
}

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

  const baseUrl = await getAuthRedirectBaseUrl();

  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: {
      emailRedirectTo: `${baseUrl}/auth/callback`,
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
