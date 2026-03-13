"use client";

import { useEffect, useMemo, useRef, useState, useTransition } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import {
  Archive,
  ArrowUp,
  Compass,
  Link2,
  NotebookText,
  Sparkles,
  Trash2,
} from "lucide-react";
import remarkGfm from "remark-gfm";

import type { ThreadReplyMeta } from "@/lib/chat";

type ResourceCard = {
  url: string;
  description: string;
};

const pendingBubbleLines = [
  "pondering for real...",
  "connecting the dots...",
  "lowkey cooking a reply...",
];

const SESSION_UPSERT_EVENT = "genz-ai-therapist:session-upsert";

function MarkdownBubbleContent() {
  return (
    <MarkdownTextPrimitive
      className="chat-markdown"
      remarkPlugins={[remarkGfm]}
    />
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="max-w-[85%]">
      <div className="rounded-[1.8rem] border border-[rgba(86,51,31,0.08)] bg-[linear-gradient(180deg,rgba(255,255,255,0.92),rgba(255,247,240,0.82))] px-4 py-3 shadow-[0_18px_36px_rgba(55,31,19,0.09)]">
        <MessagePrimitive.Content
          components={{
            Text: MarkdownBubbleContent,
          }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="ml-auto max-w-[82%]">
      <div className="rounded-[1.8rem] border border-[rgba(53,88,78,0.14)] bg-[linear-gradient(180deg,rgba(233,245,240,0.96),rgba(216,235,228,0.86))] px-4 py-3 text-[var(--forest-deep)] shadow-[0_18px_36px_rgba(29,57,52,0.08)]">
        <MessagePrimitive.Content
          components={{
            Text: MarkdownBubbleContent,
          }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

type AssistantThreadProps = {
  initialMessages: Array<{ role: "user" | "assistant"; content: string }>;
  initialSessionId?: string;
  demoMode: boolean;
  sessionTitle: string;
  initialReplyMeta: ThreadReplyMeta;
};

type ThreadRuntimeViewProps = {
  sessionId?: string;
  threadSeed: Array<{ role: "user" | "assistant"; content: string }>;
  isReplyPending: boolean;
  onConversationStart: () => void;
  onReplyPendingChange: (value: boolean) => void;
  onPayload: (payload: {
    sessionId?: string;
    title?: string;
    sentiment?: string;
    intent?: string;
    resources?: ResourceCard[];
  }) => void;
};

async function readJsonSafely(response: Response) {
  const text = await response.text();
  if (!text.trim()) {
    return null;
  }

  try {
    return JSON.parse(text) as {
      response?: string;
      sessionId?: string;
      title?: string;
      sentiment?: string;
      intent?: string;
      resources?: ResourceCard[];
      error?: string;
      detail?: string;
    };
  } catch {
    return null;
  }
}

function ThreadRuntimeView({
  sessionId,
  threadSeed,
  isReplyPending,
  onConversationStart,
  onReplyPendingChange,
  onPayload,
}: ThreadRuntimeViewProps) {
  const [pendingLineIndex, setPendingLineIndex] = useState(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!isReplyPending) {
      setPendingLineIndex(0);
      return;
    }

    const interval = window.setInterval(() => {
      setPendingLineIndex((current) => (current + 1) % pendingBubbleLines.length);
    }, 1400);

    return () => window.clearInterval(interval);
  }, [isReplyPending]);

  const runtime = useLocalRuntime(
    {
      async run({ messages }) {
        const normalizedHistory = messages
          .map((message) => ({
            role: message.role,
            content: message.content
              .filter((part) => part.type === "text")
              .map((part) => ("text" in part ? part.text : ""))
              .join("\n"),
          }))
          .filter((message) => message.content);

        const latestUser = [...normalizedHistory]
          .reverse()
          .find((message) => message.role === "user");

        onConversationStart();
        onReplyPendingChange(true);

        try {
          const response = await fetch("/api/chat/send", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            keepalive: true,
            body: JSON.stringify({
              sessionId,
              message: latestUser?.content ?? "",
              history: normalizedHistory.slice(-8),
            }),
          });

          const payload = await readJsonSafely(response);
          if (mountedRef.current) {
            onPayload({
              sessionId: payload?.sessionId,
              title: payload?.title,
              sentiment: payload?.sentiment,
              intent: payload?.intent,
              resources: payload?.resources,
            });
          }

          if (!response.ok) {
            const fallback =
              payload?.detail ||
              payload?.error ||
              "Something glitched while sending that. Try again in a second.";

            return {
              content: [
                {
                  type: "text",
                  text: fallback,
                },
              ],
            };
          }

          return {
            content: [
              {
                type: "text",
                text:
                  payload?.response ??
                  "I'm here. Say the unfiltered version and we'll sort it together.",
              },
            ],
          };
        } finally {
          if (mountedRef.current) {
            onReplyPendingChange(false);
          }
        }
      },
    },
    {
      initialMessages: threadSeed,
    },
  );

  const messageComponents = useMemo(
    () => ({
      UserMessage,
      AssistantMessage,
    }),
    [],
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="mt-5 flex min-h-0 flex-1 flex-col">
        <ThreadPrimitive.Viewport className="app-scrollbar flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
          <ThreadPrimitive.Empty>
            <div className="panel-clay rounded-[1.6rem] border border-dashed p-5 text-sm leading-7 text-[var(--muted)]">
              You do not need the perfect opener. Just drop the truest sentence you have right now.
            </div>
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages components={messageComponents} />
          {isReplyPending ? (
            <div className="max-w-[85%]">
              <div className="rounded-[1.8rem] border border-[rgba(95,102,114,0.12)] bg-[linear-gradient(180deg,rgba(248,247,244,0.96),rgba(235,236,231,0.9))] px-4 py-3 shadow-[0_18px_36px_rgba(95,102,114,0.1)]">
                <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[var(--plum)]" />
                    <span
                      className="h-2.5 w-2.5 animate-pulse rounded-full bg-[var(--plum)]"
                      style={{ animationDelay: "120ms" }}
                    />
                    <span
                      className="h-2.5 w-2.5 animate-pulse rounded-full bg-[var(--plum)]"
                      style={{ animationDelay: "240ms" }}
                    />
                  </div>
                  <span>{pendingBubbleLines[pendingLineIndex]}</span>
                </div>
              </div>
            </div>
          ) : null}
        </ThreadPrimitive.Viewport>

        <ThreadPrimitive.ViewportFooter className="mt-5 shrink-0">
          <ComposerPrimitive.Root className="rounded-[2rem] border border-[rgba(86,51,31,0.1)] bg-[linear-gradient(180deg,rgba(255,251,246,0.95),rgba(250,240,231,0.82))] p-3 shadow-[0_18px_40px_rgba(55,31,19,0.08)]">
            <ComposerPrimitive.Input
              rows={2}
              placeholder="what is feeling heavy, messy, loud, or lowkey too much rn?"
              className="min-h-14 w-full resize-none bg-transparent px-2 py-3 text-sm leading-7 outline-none placeholder:text-[var(--muted)]"
            />
            <div className="flex items-center justify-between gap-3 border-t border-[rgba(91,58,38,0.08)] pt-3">
              <p className="text-xs text-[var(--muted)]">
                GenZ AI Therapist is non-clinical. Think listener plus resource
                guidance, not a replacement for therapy. If you might act on
                self-harm or you are in immediate danger, hit emergency or
                crisis support right now.
              </p>
              <ComposerPrimitive.Send className="button-forest inline-flex min-w-[108px] shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-full px-4 py-2 text-sm font-medium transition">
                send it
                <ArrowUp size={16} />
              </ComposerPrimitive.Send>
            </div>
          </ComposerPrimitive.Root>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

export function AssistantThread({
  initialMessages,
  initialSessionId,
  demoMode,
  sessionTitle,
  initialReplyMeta,
}: AssistantThreadProps) {
  const initialHasConversation = initialMessages.some(
    (message) => message.role === "user",
  );
  const [threadSeed, setThreadSeed] = useState(initialMessages);
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [currentTitle, setCurrentTitle] = useState(sessionTitle);
  const [threadResetNonce, setThreadResetNonce] = useState(0);
  const [hasStartedConversation, setHasStartedConversation] = useState(
    initialHasConversation,
  );
  const [isReplyPending, setIsReplyPending] = useState(false);
  const [sessionStatus, setSessionStatus] = useState(
    demoMode
      ? "You can just yap in here. For now it stays in this browser session."
      : initialSessionId
        ? "This chat is saved, so you can dip and come back. Still, this is support and guidance, not therapy."
        : "Fresh thread ready. Say the real thing and the save starts on the first message.",
  );
  const [replyMeta, setReplyMeta] = useState<{
    sentiment?: string;
    intent?: string;
    resources: ResourceCard[];
  }>(initialReplyMeta);
  const [isSessionActionPending, startSessionAction] = useTransition();
  const router = useRouter();
  const searchParams = useSearchParams();
  const threadRef = useRef<HTMLDivElement>(null);

  const broadcastSessionUpsert = (payload: {
    id: string;
    title: string;
    updatedAt?: string;
    remove?: boolean;
  }) => {
    window.dispatchEvent(
      new CustomEvent(SESSION_UPSERT_EVENT, {
        detail: {
          id: payload.id,
          title: payload.title,
          updatedAt: payload.updatedAt ?? new Date().toISOString(),
          remove: payload.remove,
        },
      }),
    );
  };

  const resetToFreshThread = (nextStatus?: string) => {
    setThreadSeed([]);
    setSessionId(undefined);
    setCurrentTitle("Fresh thread");
    setReplyMeta({ resources: [] });
    setHasStartedConversation(false);
    setIsReplyPending(false);
    if (nextStatus) {
      setSessionStatus(nextStatus);
    }
    setThreadResetNonce((value) => value + 1);
    router.replace("/app/chat?fresh=1", { scroll: false });
  };

  useEffect(() => {
    const sessionQuery = searchParams.get("session");
    const freshQuery = searchParams.get("fresh");

    if (sessionId && !sessionQuery && freshQuery !== "1") {
      router.replace(`/app/chat?session=${sessionId}`, { scroll: false });
    }
  }, [router, searchParams, sessionId]);

  useEffect(() => {
    if (!sessionId || demoMode) {
      return;
    }

    let cancelled = false;

    const hydrateThreadMeta = async () => {
      const response = await fetch(`/api/chat/sessions/${sessionId}`, {
        credentials: "same-origin",
        cache: "no-store",
      });

      if (!response.ok) {
        return;
      }

      const payload = (await response.json()) as {
        ok?: boolean;
        thread?: {
          sessionId?: string;
          sessionTitle?: string;
          replyMeta?: ThreadReplyMeta;
        };
      };

      if (cancelled || !payload.ok || !payload.thread?.sessionId) {
        return;
      }

      if (payload.thread.sessionTitle) {
        setCurrentTitle(payload.thread.sessionTitle);
        broadcastSessionUpsert({
          id: payload.thread.sessionId,
          title: payload.thread.sessionTitle,
        });
      }

      if (payload.thread.replyMeta) {
        setReplyMeta((current) => ({
          sentiment: payload.thread?.replyMeta?.sentiment ?? current.sentiment,
          intent: payload.thread?.replyMeta?.intent ?? current.intent,
          resources:
            payload.thread?.replyMeta?.resources?.length
              ? payload.thread.replyMeta.resources
              : current.resources,
        }));
      }
    };

    void hydrateThreadMeta();

    return () => {
      cancelled = true;
    };
  }, [demoMode, sessionId]);

  const seedComposer = (text: string) => {
    const composer = threadRef.current?.querySelector("textarea");
    if (!(composer instanceof HTMLTextAreaElement)) {
      return;
    }

    const descriptor = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    );
    descriptor?.set?.call(composer, text);
    composer.dispatchEvent(new Event("input", { bubbles: true }));
    composer.focus();
  };

  const runSessionAction = (mode: "archive" | "delete") => {
    if (!sessionId || demoMode) {
      return;
    }

    if (
      mode === "delete" &&
      !window.confirm("Delete this session and its messages permanently?")
    ) {
      return;
    }

    startSessionAction(async () => {
      const response = await fetch(`/api/chat/sessions/${sessionId}`, {
        method: mode === "archive" ? "PATCH" : "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        body:
          mode === "archive"
            ? JSON.stringify({
                archived: true,
              })
            : undefined,
      });

      if (!response.ok) {
        setSessionStatus(
          mode === "archive"
            ? "Could not archive this session."
            : "Could not delete this session.",
        );
        return;
      }

      setSessionStatus(
        mode === "archive" ? "Session archived." : "Session deleted.",
      );
      if (mode === "delete") {
        broadcastSessionUpsert({
          id: sessionId,
          title: currentTitle,
          remove: true,
        });
        resetToFreshThread(
          "That thread is gone. Start fresh here. Still non-clinical, still here to listen, and still happy to point you toward real support if you need it.",
        );
        router.refresh();
        return;
      }
      broadcastSessionUpsert({
        id: sessionId,
        title: currentTitle,
        remove: true,
      });
      resetToFreshThread("Session archived. Fresh thread ready.");
      router.refresh();
    });
  };

  return (
    <div
      ref={threadRef}
      className="grid h-full min-h-0 gap-4 overflow-hidden xl:grid-cols-[minmax(0,1fr)_300px]"
    >
        <section className="glass dashboard-orb flex h-full min-h-0 min-w-0 flex-col rounded-[2rem] p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[rgba(91,58,38,0.08)] pb-4">
            <div>
              <p className="theme-kicker">Main yap</p>
              <h1 className="mt-2 font-display text-4xl leading-none">
                {currentTitle}
              </h1>
              <p className="mt-3 text-sm leading-6 text-[var(--muted)]">
                {sessionStatus}
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <button
                type="button"
                onClick={() =>
                  resetToFreshThread(
                    "Fresh thread ready. Say the real thing and we’ll start there.",
                  )
                }
                className="button-forest inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] transition"
              >
                <Sparkles size={14} />
                new chat
              </button>
              <div className="rounded-full bg-[rgba(53,88,78,0.12)] px-3 py-1 text-xs font-semibold text-[var(--forest)]">
                {demoMode
                  ? "private mode"
                  : sessionId
                    ? "saved session"
                    : "fresh draft"}
              </div>
              {sessionId && !demoMode ? (
                <>
                  <button
                    type="button"
                    disabled={isSessionActionPending}
                    onClick={() => runSessionAction("archive")}
                    className="button-ghost inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--ink)] transition disabled:opacity-60"
                  >
                    <Archive size={14} />
                    archive
                  </button>
                  <button
                    type="button"
                    disabled={isSessionActionPending}
                    onClick={() => runSessionAction("delete")}
                    className="button-coral inline-flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.16em] transition disabled:opacity-60"
                  >
                    <Trash2 size={14} />
                    Delete
                  </button>
                </>
              ) : null}
            </div>
          </div>

          {!hasStartedConversation ? (
            <div className="mt-5 flex flex-wrap gap-3">
              {[
                "ngl I feel weird and I can't even explain it",
                "help me figure out what is actually eating me rn",
                "can we make a tiny plan so I stop spiraling",
                "I just need to vent without getting judged",
              ].map((line) => (
                <button
                  key={line}
                  type="button"
                  onClick={() => seedComposer(line)}
                  className="theme-chip rounded-full px-4 py-2 text-sm text-[var(--ink)] transition hover:-translate-y-0.5"
                >
                  {line}
                </button>
              ))}
            </div>
          ) : null}
          <ThreadRuntimeView
            key={threadResetNonce}
            sessionId={sessionId}
            threadSeed={threadSeed}
            isReplyPending={isReplyPending}
            onConversationStart={() => setHasStartedConversation(true)}
            onReplyPendingChange={setIsReplyPending}
            onPayload={(payload) => {
              const nextSessionId = payload.sessionId ?? sessionId;
              const nextTitle = payload.title?.trim();
              const resolvedTitle =
                nextTitle ||
                (currentTitle && currentTitle !== "Fresh thread"
                  ? currentTitle
                  : undefined);

              if (nextSessionId && nextSessionId !== sessionId) {
                setSessionId(nextSessionId);
                if (!demoMode) {
                  setSessionStatus(
                    "This chat is saved, so you can dip and come back. Still, this is support and guidance, not therapy.",
                  );
                }
                router.replace(`/app/chat?session=${nextSessionId}`, {
                  scroll: false,
                });
              }

              if (nextTitle) {
                setCurrentTitle(nextTitle);
              }

              if (nextSessionId && resolvedTitle) {
                broadcastSessionUpsert({
                  id: nextSessionId,
                  title: resolvedTitle,
                });
              }

              setReplyMeta((current) => ({
                sentiment: payload.sentiment ?? current.sentiment,
                intent: payload.intent ?? current.intent,
                resources:
                  payload.resources && payload.resources.length
                    ? payload.resources
                    : current.resources,
              }));

            }}
          />
        </section>

        <aside className="flex min-h-0 min-w-0 flex-col gap-4 overflow-hidden pr-1">
          <section className="panel-moss dashboard-orb shrink-0 rounded-[1.8rem] p-4">
            <p className="theme-kicker">vibe read</p>
            <div className="mt-3 space-y-3 text-sm leading-6 text-[var(--muted)]">
              <div>
                <p className="font-semibold text-[var(--ink)]">Current mood</p>
                <p>
                  {replyMeta.sentiment
                    ? `${replyMeta.sentiment} energy`
                    : "No vibe read yet. Send a message and I’ll clock the energy."}
                </p>
              </div>
              <div>
                <p className="font-semibold text-[var(--ink)]">What you seem to need</p>
                <p>
                  {replyMeta.intent
                    ? `${replyMeta.intent} mode`
                    : "Could be support, a reality check, info, a small plan, or just space to yap."}
                </p>
              </div>
            </div>
          </section>

          <section className="panel-clay dashboard-orb flex min-h-0 flex-1 flex-col rounded-[1.8rem] p-4">
            <p className="theme-kicker">resource drops</p>
            <div className="app-scrollbar mt-3 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
              {replyMeta.resources.length ? (
                replyMeta.resources.map((resource) => (
                  <a
                    key={resource.url}
                    href={resource.url}
                    target="_blank"
                    rel="noreferrer"
                    className="block rounded-[1.2rem] border border-[rgba(182,103,67,0.12)] bg-white/72 p-3 transition hover:-translate-y-0.5 hover:bg-white"
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 rounded-full bg-[rgba(182,103,67,0.12)] p-2 text-[var(--clay)]">
                        <Link2 size={14} />
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-sm font-semibold text-[var(--ink)]">
                          {resource.url.replace(/^https?:\/\//, "")}
                        </p>
                        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                          {resource.description}
                        </p>
                      </div>
                    </div>
                  </a>
                ))
              ) : (
                <div className="rounded-[1.2rem] border border-dashed border-[rgba(182,103,67,0.18)] bg-white/58 p-3 text-sm leading-6 text-[var(--muted)]">
                  If something genuinely useful fits the convo, I’ll drop it here instead of link-dumping in your face.
                </div>
              )}
            </div>
          </section>

          <section className="panel-dark dashboard-orb shrink-0 rounded-[1.8rem] p-4 text-[var(--paper)]">
            <p className="theme-kicker text-[rgba(255,232,215,0.72)]">next move</p>
            <div className="mt-3 space-y-2.5 text-sm leading-5 text-[rgba(255,232,215,0.84)]">
              {[
                {
                  icon: Compass,
                  title: "vibe check",
                  body: "Log mood, energy, and stress so the pattern is obvious.",
                },
                {
                  icon: NotebookText,
                  title: "brain dump",
                  body: "Use the journal when the thought is too layered for chat bubbles.",
                },
                {
                  icon: Sparkles,
                  title: "your lore",
                  body: "Choose what this app is allowed to remember later.",
                },
              ].map(({ icon: Icon, title, body }) => (
                <div key={title} className="flex gap-3">
                  <Icon size={15} className="mt-0.5 shrink-0" />
                  <div>
                    <p className="font-semibold text-[var(--paper)]">{title}</p>
                    <p>{body}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </aside>
    </div>
  );
}
