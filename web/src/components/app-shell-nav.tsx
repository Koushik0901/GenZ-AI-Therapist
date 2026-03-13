"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import {
  BrainCircuit,
  BookHeart,
  LineChart,
  MessageSquare,
  Plus,
  Settings2,
  Sparkles,
} from "lucide-react";

import type { SessionSummary } from "@/lib/chat";

type AppShellNavProps = {
  recentSessions?: SessionSummary[];
  variant: "rail" | "history";
};

type SessionUpsertDetail = {
  id: string;
  title: string;
  updatedAt: string;
  remove?: boolean;
};

const SESSION_UPSERT_EVENT = "genz-ai-therapist:session-upsert";

const nav = [
  { href: "/app/chat", label: "Yap", icon: Sparkles },
  { href: "/app/journal", label: "Journal", icon: BookHeart },
  { href: "/app/check-in", label: "Vibe check", icon: BrainCircuit },
  { href: "/app/insights", label: "Pattern tea", icon: LineChart },
  { href: "/app/settings", label: "Settings", icon: Settings2 },
];

const freshChatHref = "/app/chat?fresh=1";

function mergeSessionLists(
  current: SessionSummary[],
  incoming: SessionSummary[],
) {
  const merged = [
    ...incoming,
    ...current.filter(
      (session) => !incoming.some((incomingSession) => incomingSession.id === session.id),
    ),
  ].slice(0, 8);

  const unchanged =
    current.length === merged.length &&
    current.every((session, index) => {
      const next = merged[index];
      return (
        next &&
        next.id === session.id &&
        next.title === session.title &&
        next.updatedAt === session.updatedAt
      );
    });

  return unchanged ? current : merged;
}

function applySessionPatch(
  sessions: SessionSummary[],
  patch: SessionUpsertDetail,
) {
  if (patch.remove) {
    return sessions.filter((session) => session.id !== patch.id);
  }

  const nextSession = {
    id: patch.id,
    title: patch.title,
    updatedAt: patch.updatedAt,
  };

  return [
    nextSession,
    ...sessions.filter((session) => session.id !== patch.id),
  ].slice(0, 8);
}

export function AppShellNav({
  recentSessions = [],
  variant,
}: AppShellNavProps) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeSession = searchParams.get("session");
  const [sessionPatches, setSessionPatches] = useState<SessionUpsertDetail[]>([]);

  useEffect(() => {
    const handleSessionUpsert = (event: Event) => {
      const detail = (event as CustomEvent<SessionUpsertDetail>).detail;
      if (!detail?.id) {
        return;
      }

      setSessionPatches((current) => {
        const next = [
          detail,
          ...current.filter((patch) => patch.id !== detail.id),
        ];

        return next.slice(0, 12);
      });
    };

    window.addEventListener(SESSION_UPSERT_EVENT, handleSessionUpsert);
    return () => {
      window.removeEventListener(SESSION_UPSERT_EVENT, handleSessionUpsert);
    };
  }, []);

  useEffect(() => {
    if (variant !== "history") {
      return;
    }

    let cancelled = false;

    const hydrateSessions = async () => {
      const response = await fetch("/api/chat/sessions", {
        credentials: "same-origin",
        cache: "no-store",
      });

      if (!response.ok) {
        return;
      }

      const payload = (await response.json()) as {
        ok?: boolean;
        sessions?: SessionSummary[];
      };

      const sessions = Array.isArray(payload.sessions) ? payload.sessions : [];

      if (cancelled || !payload.ok) {
        return;
      }

      setSessionPatches((current) => {
        const hydrated = sessions.filter(
          (session) =>
            !current.some((patch) => patch.id === session.id && patch.remove),
        );

        const next = [
          ...hydrated.map((session) => ({
            id: session.id,
            title: session.title,
            updatedAt: session.updatedAt,
          })),
          ...current,
        ].slice(0, 12);

        return next;
      });
    };

    void hydrateSessions();

    const handleFocus = () => {
      void hydrateSessions();
    };

    window.addEventListener("focus", handleFocus);
    return () => {
      cancelled = true;
      window.removeEventListener("focus", handleFocus);
    };
  }, [variant]);

  const visibleSessions = useMemo(() => {
    const patched = sessionPatches.reduce(applySessionPatch, recentSessions);
    return mergeSessionLists(recentSessions, patched);
  }, [recentSessions, sessionPatches]);

  if (variant === "rail") {
    return (
      <div className="flex flex-col items-center gap-3">
        <Link
          href="/"
          className="flex h-12 w-12 items-center justify-center rounded-[1.3rem] border border-[rgba(255,240,225,0.12)] bg-[rgba(255,255,255,0.08)] text-[var(--paper)] shadow-[0_16px_30px_rgba(39,21,14,0.24)]"
        >
          <Sparkles size={18} />
        </Link>
        <div className="h-px w-8 bg-[rgba(255,240,225,0.12)]" />
        <nav className="flex flex-col gap-2">
          {nav.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);

            return (
              <Link
                key={href}
                href={href}
                title={label}
                className={`group flex h-12 w-12 items-center justify-center rounded-[1.1rem] border transition ${
                  active
                    ? "border-[rgba(255,240,225,0.18)] bg-[rgba(255,255,255,0.14)] text-[var(--paper)] shadow-[0_12px_24px_rgba(245,228,218,0.08)]"
                    : "border-[rgba(255,240,225,0.08)] bg-[rgba(255,255,255,0.06)] text-[rgba(255,236,219,0.72)] hover:-translate-y-0.5 hover:border-[rgba(255,240,225,0.18)] hover:bg-[rgba(255,255,255,0.1)] hover:text-[var(--paper)]"
                }`}
              >
                <Icon size={18} />
              </Link>
            );
          })}
        </nav>
      </div>
    );
  }

  return (
    <>
      <div className="flex items-start justify-between gap-3 border-b border-[rgba(91,58,38,0.08)] pb-4">
        <div>
          <p className="theme-kicker">chat history</p>
          <p className="mt-2 font-display text-3xl leading-none">
            GenZ AI Therapist
          </p>
          <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
            Pick an old convo back up or start a fresh yap. This is support and
            reflection, not clinical care.
          </p>
        </div>
        <Link
          href={freshChatHref}
          className={`inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-[1.1rem] transition ${
            pathname === "/app/chat" && !activeSession
              ? "button-forest text-[var(--paper)]"
              : "button-coral text-white"
          }`}
          title="New chat"
        >
          <Plus size={18} />
        </Link>
      </div>

      <div className="app-scrollbar mt-4 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {visibleSessions.length ? (
          visibleSessions.map((session) => {
            const active = pathname === "/app/chat" && activeSession === session.id;

            return (
              <Link
                key={session.id}
                href={`/app/chat?session=${session.id}`}
                className={`block rounded-[1.3rem] border px-4 py-3 transition ${
                  active
                    ? "border-[rgba(95,102,114,0.18)] bg-[rgba(95,102,114,0.1)] shadow-[0_14px_28px_rgba(95,102,114,0.1)]"
                    : "border-[rgba(91,58,38,0.08)] bg-white/62 hover:-translate-y-0.5 hover:bg-white"
                }`}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${
                      active
                        ? "bg-[rgba(95,102,114,0.16)] text-[var(--plum-deep)]"
                        : "bg-[rgba(53,88,78,0.1)] text-[var(--forest)]"
                    }`}
                  >
                    <MessageSquare size={15} />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-[var(--ink)]">
                      {session.title}
                    </p>
                    <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                      {new Date(session.updatedAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </Link>
            );
          })
        ) : (
          <div className="rounded-[1.3rem] border border-dashed border-[rgba(91,58,38,0.16)] bg-white/52 px-4 py-4 text-sm leading-6 text-[var(--muted)]">
            Your saved lore shows up here after the first real convo.
          </div>
        )}
      </div>
    </>
  );
}
