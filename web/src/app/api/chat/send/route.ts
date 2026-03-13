import { NextResponse } from "next/server";
import { z } from "zod";

import { runCompanionFlow } from "@/lib/companion";
import { ensureSessionForUser } from "@/lib/chat";
import { limitChatSend } from "@/lib/rate-limit";
import { createServerSupabase } from "@/lib/supabase/server";

const chatSchema = z.object({
  sessionId: z.string().uuid().optional(),
  message: z.string().min(1).max(3000),
  history: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().min(1).max(2000),
      }),
    )
    .max(8)
    .optional(),
});

export async function POST(request: Request) {
  try {
    const parsed = chatSchema.safeParse(await request.json());

    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    }

    const { message, history = [], sessionId } = parsed.data;

    let activeSessionId = sessionId;
    let supabaseUserId: string | null = null;
    let persistenceEnabled = false;
    let createdNewSession = false;

    const supabase = await createServerSupabase();
    if (supabase) {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      supabaseUserId = user?.id ?? null;

      const rateLimit = limitChatSend(request, supabaseUserId);
      if (!rateLimit.ok) {
        return NextResponse.json(
          {
            error: "Too many messages",
            detail:
              "You're sending messages a little too fast. Give it a minute, then send the next one.",
          },
          {
            status: 429,
            headers: {
              "Retry-After": String(rateLimit.retryAfterSeconds),
            },
          },
        );
      }

      if (supabaseUserId) {
        try {
          const sessionResult = await ensureSessionForUser(
            supabaseUserId,
            activeSessionId,
            message,
          );
          activeSessionId = sessionResult?.id;
          createdNewSession = Boolean(sessionResult?.isNew);

          const { error: userMessageError } = await supabase.from("messages").insert({
            session_id: activeSessionId,
            role: "user",
            content: message,
          });

          if (userMessageError) {
            throw userMessageError;
          }

          const { error: sessionUpdateError } = await supabase
            .from("chat_sessions")
            .update({ updated_at: new Date().toISOString() })
            .eq("id", activeSessionId);

          if (sessionUpdateError) {
            throw sessionUpdateError;
          }

          persistenceEnabled = true;
        } catch {
          activeSessionId = undefined;
          persistenceEnabled = false;
        }
      }
    } else {
      const rateLimit = limitChatSend(request, null);
      if (!rateLimit.ok) {
        return NextResponse.json(
          {
            error: "Too many messages",
            detail:
              "You're sending messages a little too fast. Give it a minute, then send the next one.",
          },
          {
            status: 429,
            headers: {
              "Retry-After": String(rateLimit.retryAfterSeconds),
            },
          },
        );
      }
    }

    const flow = await runCompanionFlow({
      userMessage: message,
      history,
      wantsTitle: createdNewSession || !sessionId,
    });

    if (supabase && supabaseUserId && activeSessionId && persistenceEnabled) {
      try {
        if (flow.title) {
          await supabase
            .from("chat_sessions")
            .update({
              title: flow.title,
              updated_at: new Date().toISOString(),
            })
            .eq("id", activeSessionId);
        }

        const { error: assistantMessageError } = await supabase.from("messages").insert({
          session_id: activeSessionId,
          role: "assistant",
          content: flow.response,
          sentiment: flow.classification.sentiment,
          intent: flow.classification.intent,
          safety_level:
            flow.classification.sentiment === "Crisis" ? "crisis" : "standard",
          resource_payload: flow.resources,
          model_metadata: {
            route: flow.guard.route,
            route_reason: flow.guard.reason,
            conversation_wellness: flow.wellness,
          },
        });

        if (assistantMessageError) {
          throw assistantMessageError;
        }

        const { error: finalSessionUpdateError } = await supabase
          .from("chat_sessions")
          .update({ updated_at: new Date().toISOString() })
          .eq("id", activeSessionId);

        if (finalSessionUpdateError) {
          throw finalSessionUpdateError;
        }
      } catch {
        persistenceEnabled = false;
      }
    }

    return NextResponse.json({
      sessionId: activeSessionId,
      response: flow.response,
      title: flow.title,
      sentiment: flow.classification.sentiment,
      intent: flow.classification.intent,
      wellness: flow.wellness,
      resources: flow.resources,
    });
  } catch (error) {
    const detail =
      error instanceof z.ZodError
        ? "That message payload came through malformed."
        : "The companion hit a temporary server issue. Try again in a second.";

    return NextResponse.json(
      {
        error: "Chat request failed",
        detail,
      },
      { status: 500 },
    );
  }
}
