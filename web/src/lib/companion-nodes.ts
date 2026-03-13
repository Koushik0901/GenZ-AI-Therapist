import { z } from "zod";

import {
  buildResourceQuery,
  callStructuredWithFallback,
  classificationSchema,
  containsPromptAttack,
  ensureClassification,
  ensureGuard,
  fallbackClassification,
  fallbackGuard,
  fallbackTherapistReply,
  fallbackTitle,
  fallbackWellnessSignal,
  graphStateSchema,
  guardSchema,
  normalizeResources,
  resourceSchema,
  sanitizeForPrompt,
  sanitizeSearchResults,
  searchWeb,
  serializeHistory,
  supportFallbackResources,
  therapistSchema,
  titleSchema,
  wellnessSchema,
  CLASSIFICATION_PROMPT,
  GUARD_PROMPT,
  RESOURCE_PROMPT,
  THERAPIST_PROMPT,
  TITLE_PROMPT,
  WELLNESS_PROMPT,
} from "@/lib/companion-foundation";

type GraphState = z.infer<typeof graphStateSchema>;

export function buildPromptInjectionResponse(input: {
  wantsTitle?: boolean;
  promptAttackReason?: string;
}) {
  return {
    guard: {
      route: false,
      reason: input.promptAttackReason || "prompt-extraction-attempt",
    },
    title: input.wantsTitle ? "Boundary check" : undefined,
    classification: {
      sentiment: "Neutral" as const,
      intent: "other" as const,
    },
    wellness: {
      mood: 50,
      energy: 50,
      stress: 40,
    },
    resources: [],
    response:
      "Nah, I can't help with hidden prompts, system rules, or jailbreak stuff. If what's actually going on is stress, spiraling, relationship mess, burnout, or you just need a real place to vent, say that part and I'll meet you there.",
  };
}

export function buildGuardRejectionResponse(input: {
  guard: z.infer<typeof guardSchema> | undefined;
  title?: string;
}) {
  return {
    guard: ensureGuard(input.guard),
    title: input.title,
    classification: {
      sentiment: "Neutral" as const,
      intent: "other" as const,
    },
    wellness: {
      mood: 50,
      energy: 50,
      stress: 45,
    },
    resources: [],
    response:
      "Nah, I can’t help with that one in here. If you want actual emotional support, life stuff, stress, burnout, relationship mess, or just a safe place to yap, I’m here for that.",
  };
}

export async function sanitizeContextNode(state: GraphState) {
  return {
    safeUserMessage: sanitizeForPrompt(state.userMessage, 2200),
    safeHistory: state.history.map((item) => ({
      role: item.role,
      content: sanitizeForPrompt(item.content, 700),
    })),
  };
}

export async function promptInjectionNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const promptAttackDetected = containsPromptAttack(safeUserMessage);

  return {
    promptAttackDetected,
    promptAttackReason: promptAttackDetected
      ? "prompt-extraction-attempt"
      : undefined,
  };
}

export async function guardNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const guard = await callStructuredWithFallback(
    guardSchema,
    {
      prompt: `${GUARD_PROMPT}\n\nLatest untrusted user message:\n<user_message>\n${safeUserMessage}\n</user_message>`,
      temperature: 0.1,
      maxTokens: 120,
    },
    () => fallbackGuard(safeUserMessage),
  );

  return { guard };
}

export async function titleNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const titleOutput = await callStructuredWithFallback(
    titleSchema,
    {
      prompt: `${TITLE_PROMPT}\n\nFirst user message:\n<user_message>\n${safeUserMessage}\n</user_message>`,
      temperature: 0.4,
      maxTokens: 80,
    },
    () => ({
      title: fallbackTitle(safeUserMessage),
    }),
  );

  return {
    title: titleOutput.title,
  };
}

export async function classificationNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const classification = await callStructuredWithFallback(
    classificationSchema,
    {
      prompt: `${CLASSIFICATION_PROMPT}\n\nLatest untrusted user message:\n<user_message>\n${safeUserMessage}\n</user_message>`,
      temperature: 0.1,
      maxTokens: 120,
    },
    () => fallbackClassification(safeUserMessage),
  );

  return { classification };
}

export async function wellnessNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const safeHistory = state.safeHistory ?? [];
  const classification = ensureClassification(state.classification);
  const wellness = await callStructuredWithFallback(
    wellnessSchema,
    {
      prompt: `${WELLNESS_PROMPT}

Latest user message:
<user_message>
${safeUserMessage}
</user_message>

Recent convo:
${serializeHistory(safeHistory)}

Classification:
sentiment=${classification.sentiment}
intent=${classification.intent}
`,
      temperature: 0.2,
      maxTokens: 120,
    },
    () =>
      fallbackWellnessSignal({
        message: safeUserMessage,
        classification,
      }),
  );

  return { wellness };
}

export async function resourceSearchNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const classification = ensureClassification(state.classification);
  const searchResults = sanitizeSearchResults(
    await searchWeb(
      buildResourceQuery(
        safeUserMessage,
        classification.sentiment,
        classification.intent,
      ),
    ),
  );

  return { searchResults };
}

export async function resourceSelectionNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const classification = ensureClassification(state.classification);
  const searchResults = state.searchResults ?? [];
  const fallbackResources = supportFallbackResources(
    safeUserMessage,
    classification,
  );

  if (!searchResults.length) {
    return { resources: fallbackResources };
  }

  const resourceOutput = await callStructuredWithFallback(
    resourceSchema,
    {
      prompt: `${RESOURCE_PROMPT}

User message:
<user_message>
${safeUserMessage}
</user_message>

Classification:
sentiment=${classification.sentiment}
intent=${classification.intent}

Trusted search results data:
${searchResults
  .map(
    (result, index) =>
      `${index + 1}. ${result.title}\nURL: ${result.link}\nSnippet: ${result.snippet}`,
  )
  .join("\n\n")}
`,
      temperature: 0.2,
      maxTokens: 260,
    },
    () => ({
      resources: fallbackResources,
    }),
  );

  return {
    resources: resourceOutput.resources.length
      ? normalizeResources(resourceOutput.resources)
      : fallbackResources,
  };
}

export async function therapistNode(state: GraphState) {
  const safeUserMessage =
    state.safeUserMessage ?? sanitizeForPrompt(state.userMessage, 2200);
  const safeHistory = state.safeHistory ?? [];
  const classification = ensureClassification(state.classification);
  const resources = state.resources ?? [];

  const therapist = await callStructuredWithFallback(
    therapistSchema,
    {
      prompt: `${THERAPIST_PROMPT}

Latest user message:
<user_message>
${safeUserMessage}
</user_message>

Recent convo:
${serializeHistory(safeHistory)}

Classification:
sentiment=${classification.sentiment}
intent=${classification.intent}

Resources:
${resources.length ? JSON.stringify(resources, null, 2) : "[]"}
`,
      temperature: 0.8,
      maxTokens: 420,
    },
    async () => ({
      response: await fallbackTherapistReply({
        userMessage: safeUserMessage,
        history: safeHistory,
        classification,
        resources,
      }),
      sentiment: classification.sentiment,
      intent: classification.intent,
      resources,
    }),
  );

  return {
    response: therapist.response,
    resources: normalizeResources(
      therapist.resources.length ? therapist.resources : resources,
    ),
  };
}

export async function injectionBlockedNode(state: GraphState) {
  return buildPromptInjectionResponse({
    wantsTitle: state.wantsTitle,
    promptAttackReason: state.promptAttackReason,
  });
}

export async function guardBlockedNode(state: GraphState) {
  return buildGuardRejectionResponse({
    guard: state.guard,
    title: state.title,
  });
}

export function routeAfterInjectionCheck(state: GraphState) {
  return state.promptAttackDetected ? "injection_block_node" : "guard_node";
}

export function routeAfterGuard(state: GraphState) {
  if (state.wantsTitle) {
    return "title_node";
  }

  return ensureGuard(state.guard).route
    ? "classification_node"
    : "guard_block_node";
}

export function routeAfterTitle(state: GraphState) {
  return ensureGuard(state.guard).route
    ? "classification_node"
    : "guard_block_node";
}
