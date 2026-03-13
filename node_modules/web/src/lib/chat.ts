import { unstable_noStore as noStore } from "next/cache";

import { seedMessages } from "@/lib/demo-data";
import { createServerSupabase } from "@/lib/supabase/server";

type ChatSessionRow = {
  id: string;
  title: string;
  user_id: string;
  archived: boolean;
  created_at: string;
  updated_at: string;
};

type MessageRow = {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  sentiment: string | null;
  intent: string | null;
  resource_payload: unknown;
  created_at: string;
};

type ResourceCard = {
  url: string;
  description: string;
};

export type ThreadReplyMeta = {
  sentiment?: string;
  intent?: string;
  resources: ResourceCard[];
};

export type SessionSummary = {
  id: string;
  title: string;
  updatedAt: string;
};

export type ThreadSeedMessage = {
  role: "user" | "assistant";
  content: string;
};

function normalizeResources(value: unknown): ResourceCard[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(
      (item): item is { url: string; description: string } =>
        Boolean(
          item &&
            typeof item === "object" &&
            "url" in item &&
            "description" in item &&
            typeof item.url === "string" &&
            typeof item.description === "string",
        ),
    )
    .slice(0, 3)
    .map((item) => ({
      url: item.url,
      description: item.description,
    }));
}

function deriveSessionTitle(message: string) {
  const compact = message.replace(/\s+/g, " ").trim();
  if (!compact) {
    return "Untitled session";
  }

  return compact.split(" ").slice(0, 6).join(" ").slice(0, 72);
}

export async function getLatestSessionWithMessages(userId?: string) {
  noStore();
  return getSessionWithMessages(userId);
}

export async function getSessionSummaries(userId?: string) {
  noStore();

  if (!userId) {
    return [
      {
        id: "foundation-session",
        title: "Foundation session",
        updatedAt: new Date().toISOString(),
      },
    ] satisfies SessionSummary[];
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return [] satisfies SessionSummary[];
  }

  try {
    const { data: sessionRows } = await supabase
      .from("chat_sessions")
      .select("id,title,updated_at")
      .eq("user_id", userId)
      .eq("archived", false)
      .order("updated_at", { ascending: false })
      .limit(8);

    return ((sessionRows ?? []) as Array<{ id: string; title: string; updated_at: string }>).map(
      (row) => ({
        id: row.id,
        title: row.title,
        updatedAt: row.updated_at,
      }),
    );
  } catch {
    return [] satisfies SessionSummary[];
  }
}

export async function getSessionWithMessages(userId?: string, sessionId?: string) {
  noStore();

  if (!userId) {
    return {
      sessionId: undefined,
      sessionTitle: "Foundation session",
      messages: seedMessages,
      demoMode: true,
      replyMeta: {
        resources: [],
      } satisfies ThreadReplyMeta,
    };
  }

  const supabase = await createServerSupabase();
  if (!supabase) {
    return {
      sessionId: undefined,
      sessionTitle: "Fresh thread",
      messages: [],
      demoMode: false,
      replyMeta: {
        resources: [],
      } satisfies ThreadReplyMeta,
    };
  }

  try {
    let sessionRowsQuery = supabase
      .from("chat_sessions")
      .select("id,title,user_id,archived,created_at,updated_at")
      .eq("user_id", userId)
      .eq("archived", false);

    if (sessionId) {
      sessionRowsQuery = sessionRowsQuery.eq("id", sessionId).limit(1);
    } else {
      sessionRowsQuery = sessionRowsQuery.order("updated_at", { ascending: false }).limit(1);
    }

    const { data: sessionRows } = await sessionRowsQuery;

    const session = sessionRows?.[0] as ChatSessionRow | undefined;
    if (!session) {
      return {
        sessionId: undefined,
        sessionTitle: "Fresh thread",
        messages: [],
        demoMode: false,
        replyMeta: {
          resources: [],
        } satisfies ThreadReplyMeta,
      };
    }

    const { data: messageRows } = await supabase
      .from("messages")
      .select("id,session_id,role,content,sentiment,intent,resource_payload,created_at")
      .eq("session_id", session.id)
      .order("created_at", { ascending: true });

    const messages = ((messageRows ?? []) as MessageRow[]).map((row) => ({
      role: row.role,
      content: row.content,
    }));

    const orderedMessages = (messageRows ?? []) as MessageRow[];

    const latestAssistant = [...orderedMessages]
      .reverse()
      .find((row) => row.role === "assistant");
    const latestResourceAssistant = [...orderedMessages].reverse().find((row) => {
      if (row.role !== "assistant") {
        return false;
      }

      return normalizeResources(row.resource_payload).length > 0;
    });

    return {
      sessionId: session.id,
      sessionTitle: session.title,
      messages,
      demoMode: false,
      replyMeta: {
        sentiment: latestAssistant?.sentiment ?? undefined,
        intent: latestAssistant?.intent ?? undefined,
        resources: normalizeResources(latestResourceAssistant?.resource_payload),
      } satisfies ThreadReplyMeta,
    };
  } catch {
    return {
      sessionId: undefined,
      sessionTitle: "Fresh thread",
      messages: [],
      demoMode: false,
      replyMeta: {
        resources: [],
      } satisfies ThreadReplyMeta,
    };
  }
}

export async function ensureSessionForUser(
  userId: string,
  sessionId: string | undefined,
  firstMessage: string,
) {
  noStore();

  const supabase = await createServerSupabase();
  if (!supabase) {
    return null;
  }

  if (sessionId) {
    const { data: existing } = await supabase
      .from("chat_sessions")
      .select("id")
      .eq("id", sessionId)
      .eq("user_id", userId)
      .maybeSingle();

    if (existing?.id) {
      return {
        id: existing.id,
        isNew: false,
      };
    }
  }

  const { data: created, error } = await supabase
    .from("chat_sessions")
    .insert({
      user_id: userId,
      title: deriveSessionTitle(firstMessage),
    })
    .select("id")
    .single();

  if (error) {
    throw error;
  }

  return {
    id: created.id as string,
    isNew: true,
  };
}
