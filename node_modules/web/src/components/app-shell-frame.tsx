"use client";

import { usePathname } from "next/navigation";
import { House, LogOut } from "lucide-react";

import type { SessionSummary } from "@/lib/chat";
import { AppShellNav } from "@/components/app-shell-nav";

type AppShellFrameProps = {
  children: React.ReactNode;
  userLabel: string;
  recentSessions: SessionSummary[];
  signOutAction?: () => Promise<void>;
};

export function AppShellFrame({
  children,
  userLabel,
  recentSessions,
  signOutAction,
}: AppShellFrameProps) {
  const pathname = usePathname();
  const showHistory = pathname === "/app/chat";

  return (
    <div
      className={`mx-auto grid h-[100dvh] max-w-[1600px] gap-4 overflow-hidden px-4 py-4 lg:px-6 ${
        showHistory
          ? "lg:grid-cols-[78px_320px_minmax(0,1fr)]"
          : "lg:grid-cols-[78px_minmax(0,1fr)]"
      }`}
    >
      <aside className="panel-dark relative z-20 flex h-full min-h-0 flex-col items-center justify-between rounded-[2rem] px-3 py-4 text-[var(--paper)]">
        <AppShellNav variant="rail" />
        <div className="flex flex-col items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-[1.1rem] border border-[rgba(255,240,225,0.1)] bg-[rgba(255,255,255,0.08)] text-[var(--paper)]">
            <House size={18} />
          </div>
          {signOutAction ? (
            <form action={signOutAction}>
              <button
                type="submit"
                title="Log out"
                className="flex h-12 w-12 items-center justify-center rounded-[1.1rem] border border-[rgba(255,240,225,0.1)] bg-[rgba(255,255,255,0.08)] text-[rgba(255,236,219,0.72)] transition hover:bg-[rgba(255,255,255,0.12)] hover:text-[var(--paper)]"
              >
                <LogOut size={18} />
              </button>
            </form>
          ) : null}
        </div>
      </aside>

      {showHistory ? (
        <aside className="panel-plum dashboard-orb relative z-10 flex h-full min-h-0 flex-col rounded-[2rem] p-4">
          <AppShellNav variant="history" recentSessions={recentSessions} />

          <div className="mt-4 shrink-0 space-y-3 border-t border-[rgba(91,58,38,0.08)] pt-4">
            <div className="panel-dark rounded-[1.5rem] p-4 text-[var(--paper)]">
              <p className="theme-kicker text-[rgba(255,239,219,0.66)]">Signed in as</p>
              <p className="mt-2 truncate text-sm font-medium text-[var(--paper)]">
                {userLabel}
              </p>
              <p className="mt-3 text-sm leading-6 text-[rgba(255,239,219,0.82)]">
                GenZ AI Therapist is non-clinical. Think listener, reflection
                space, and resource guidance, not actual therapy.
              </p>
            </div>
            <div className="panel-clay rounded-[1.2rem] px-4 py-3 text-sm leading-6 text-[var(--muted)]">
              If you are in immediate danger or think you may act on thoughts of
              self-harm, contact local emergency services or a crisis line right
              away.
            </div>
          </div>
        </aside>
      ) : null}

      <main className="relative z-0 h-full min-h-0 overflow-hidden">{children}</main>
    </div>
  );
}
