import Link from "next/link";
import { redirect } from "next/navigation";
import { ArrowLeft } from "lucide-react";

import { signInAction } from "@/app/auth/actions";
import { isSupabaseConfigured } from "@/lib/env";
import { getViewer } from "@/lib/viewer";

export default async function AuthPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { user } = await getViewer();

  if (isSupabaseConfigured && user) {
    redirect("/app/chat");
  }

  const params = await searchParams;
  const sent = params.sent === "1";
  const error = typeof params.error === "string" ? params.error : "";
  const email = typeof params.email === "string" ? params.email : "";

  return (
    <main className="hero-grid mx-auto flex min-h-screen max-w-[1240px] items-center px-4 py-8 lg:px-6">
      <div className="grid w-full gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="panel-clay dashboard-orb rounded-[2.2rem] p-8">
          <p className="theme-kicker">Welcome back</p>
          <h1 className="mt-4 font-display text-6xl leading-[0.9]">No password mess. Just get back in.</h1>
          <p className="mt-4 max-w-xl text-base leading-8 text-[var(--muted)]">
            Use your email to get back to your saved chats, journal dumps, vibe
            checks, and the memory choices you said were okay to keep inside
            GenZ AI Therapist.
          </p>
          <form action={signInAction} className="mt-8 rounded-[1.6rem] border border-[rgba(91,58,38,0.1)] bg-white/84 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.42)]">
            <label className="block">
              <span className="text-sm font-semibold">Email</span>
              <input
                type="email"
                name="email"
                defaultValue={email}
                placeholder="you@example.com"
                className="theme-input mt-2 w-full rounded-[1rem] px-4 py-3 text-sm"
              />
            </label>
            <button
              type="submit"
              disabled={!isSupabaseConfigured}
              className="button-forest mt-4 rounded-full px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60"
            >
              Send magic link
            </button>
            <p className="mt-4 text-sm leading-7 text-[var(--muted)]">
              {isSupabaseConfigured
                ? "You’ll get a one-time sign-in link by email. No password drama."
                : "Sign-in is still getting wired up, but the app shell is here."}
            </p>
            {sent ? (
              <p className="mt-3 text-sm leading-7 text-[var(--forest)]">
                Check your inbox. The magic link got sent to {email || "your email"}.
              </p>
            ) : null}
            {error ? (
              <p className="mt-3 text-sm leading-7 text-[var(--coral)]">
                {decodeURIComponent(error)}
              </p>
            ) : null}
          </form>
        </section>

        <aside className="panel-dark dashboard-orb rounded-[2.2rem] p-8 text-[var(--paper)]">
          <p className="theme-kicker text-[rgba(255,238,220,0.72)]">Why sign in</p>
          <p className="mt-4 text-sm leading-7 text-[rgba(255,238,220,0.84)]">
            Signing in keeps your whole vibe intact. Your chats, journals,
            check-ins, and approved memory do not have to reset every time you
            bounce. This space is still non-clinical and not a replacement for
            therapy.
          </p>
          <Link href="/" className="mt-8 inline-flex items-center gap-2 text-sm underline-offset-4 hover:underline">
            <ArrowLeft size={16} />
            Back home
          </Link>
        </aside>
      </div>
    </main>
  );
}
