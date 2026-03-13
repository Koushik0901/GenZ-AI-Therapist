import { AssistantThread } from "@/components/assistant-thread";
import { getSessionWithMessages } from "@/lib/chat";
import { getViewer } from "@/lib/viewer";

export default async function ChatPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const wantsFreshThread =
    params.fresh === "1" || (Array.isArray(params.fresh) && params.fresh.includes("1"));
  const sessionId =
    !wantsFreshThread && typeof params.session === "string" && params.session
      ? params.session
      : undefined;
  const { user } = await getViewer();
  const userId = user?.id;

  const thread = wantsFreshThread
    ? {
        sessionId: undefined,
        sessionTitle: "Fresh thread",
        messages: [],
        demoMode: !userId,
        replyMeta: {
          resources: [],
        },
      }
    : await getSessionWithMessages(userId, sessionId);

  return (
    <AssistantThread
      key={`${thread.sessionId ?? "fresh-thread"}-${thread.messages.length}`}
      initialMessages={thread.messages}
      initialSessionId={thread.sessionId}
      demoMode={thread.demoMode}
      sessionTitle={thread.sessionTitle}
      initialReplyMeta={thread.replyMeta}
    />
  );
}
