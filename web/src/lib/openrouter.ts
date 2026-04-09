/**
 * OpenRouter API Helper
 * Provides a unified interface for calling OpenRouter models (Kimi K2.5, etc.)
 */

import { appEnv, isOpenRouterConfigured } from '@/lib/env';

/**
 * Raw OpenRouter API call
 */
export async function callOpenRouter(input: {
  system: string;
  user: string;
  temperature?: number;
  maxTokens?: number;
}) {
  if (!isOpenRouterConfigured) {
    throw new Error('OpenRouter is not configured (OPENROUTER_API_KEY missing).');
  }

  const model = process.env.OPENROUTER_MODEL || 'moonshotai/kimi-k2.5';

  const response = await fetch(`${appEnv.openRouterBaseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${appEnv.openRouterApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'system',
          content: input.system,
        },
        {
          role: 'user',
          content: input.user,
        },
      ],
      temperature: input.temperature ?? 0.4,
      max_tokens: input.maxTokens ?? 280,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'OpenRouter request failed.');
  }

  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: unknown } }>;
  };

  const content = payload.choices?.[0]?.message?.content;
  return unwrapContent(content).trim();
}

/**
 * Helper to extract text from various content formats
 */
function unwrapContent(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') {
          return part;
        }

        if (
          part &&
          typeof part === 'object' &&
          'type' in part &&
          'text' in part &&
          part.type === 'text'
        ) {
          return String(part.text);
        }

        return '';
      })
      .join('\n');
  }

  return '';
}
