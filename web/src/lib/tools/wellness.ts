import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { Classification } from './classification';

/**
 * Wellness Tool
 * Infers mood, energy, and stress levels from conversation context with confidence scoring
 */

// Schemas
export const WellnessSignalSchema = z.object({
  mood: z.number().int().min(0).max(100).describe('0=darkest, 100=brightest emotional state'),
  energy: z.number().int().min(0).max(100).describe('0=depleted, 100=fully activated'),
  stress: z.number().int().min(0).max(100).describe('0=calm, 100=overwhelmed'),
  confidence: z.number().min(0).max(100).describe('Confidence in this inference'),
  reasoning: z.string().describe('Why these scores were assigned'),
});

export type WellnessSignal = z.infer<typeof WellnessSignalSchema>;

const WELLNESS_PROMPT = `You are a wellness inference expert for emotional support conversations.

Infer the user's likely mood, energy, and stress based on:
- The latest message
- Recent conversation history
- The classification (sentiment/intent)

DO NOT diagnose. This is an inference, not self-reported.

SCORING:
- mood: 0=darkest emotional state (suicidal, hopeless), 50=neutral/mixed, 100=bright/hopeful
- energy: 0=completely depleted/can't move, 50=functional baseline, 100=energized/activated
- stress: 0=calm/regulated, 50=moderate pressure, 100=overwhelmed/panicked

Rate your confidence 0-100. Lower confidence means the conversation doesn't provide enough signal.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "mood": 0-100,
  "energy": 0-100,
  "stress": 0-100,
  "confidence": 0-100,
  "reasoning": "Why you inferred these scores"
}`;

/**
 * Infer wellness signal from conversation context
 */
export async function inferWellness(args: {
  userMessage: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  classification: Classification;
}): Promise<WellnessSignal> {
  const startTime = Date.now();

  try {
    // Build context from history
    const historyContext = args.history
      .slice(-4)
      .map((h) => `${h.role.toUpperCase()}: ${h.content.slice(0, 200)}`)
      .join('\n');

    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `Classification:
sentiment=${args.classification.sentiment}
intent=${args.classification.intent}
confidence=${args.classification.confidence}

Recent context:
${historyContext}

Latest message: "${args.userMessage}"

${WELLNESS_PROMPT}`,
      temperature: 0.3,
      maxTokens: 300,
    });

    // Extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid response format: no JSON found');
    }

    const wellness = WellnessSignalSchema.parse(JSON.parse(jsonMatch[0]));

    // Log successful inference
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'wellness',
      input: {
        message_length: args.userMessage.length,
        history_length: args.history.length,
        classification_sentiment: args.classification.sentiment,
      },
      output: {
        mood: wellness.mood,
        energy: wellness.energy,
        stress: wellness.stress,
        confidence: wellness.confidence,
      },
      duration_ms: duration,
      success: true,
    });

    return wellness;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'wellness_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Wellness inference failed, using fallback'
    );

    // Fallback: Use sentiment-based inference
    const fallback = fallbackWellness(args.userMessage, args.classification);

    logToolCall({
      tool_name: 'wellness',
      input: {
        message_length: args.userMessage.length,
        history_length: args.history.length,
        classification_sentiment: args.classification.sentiment,
      },
      output: fallback,
      duration_ms: duration,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });

    return fallback;
  }
}

/**
 * Fallback wellness inference using sentiment patterns
 */
function fallbackWellness(message: string, classification: Classification): WellnessSignal {
  const lower = message.toLowerCase();

  // Base scores by sentiment
  const presetBySentiment = {
    Crisis: { mood: 12, energy: 20, stress: 92 },
    Negative: { mood: 32, energy: 38, stress: 76 },
    Neutral: { mood: 55, energy: 52, stress: 48 },
    Positive: { mood: 76, energy: 68, stress: 30 },
  } as const;

  let scores = { ...presetBySentiment[classification.sentiment] };

  // Adjust for energy-related keywords
  if (/\b(exhausted|drained|fried|burned out|burnt out|tired|sleeping|bed)\b/i.test(lower)) {
    scores.energy = Math.max(0, scores.energy - 18);
    scores.stress = Math.min(100, scores.stress + 6);
  }

  // Adjust for panic/anxiety
  if (/\b(panic|spiral|overwhelmed|anxious|stressed|shaking|heart racing)\b/i.test(lower)) {
    scores.mood = Math.max(0, scores.mood - 12);
    scores.stress = Math.min(100, scores.stress + 14);
  }

  // Adjust for positive markers
  if (/\b(proud|better|relieved|grateful|calm|okay today|doing better)\b/i.test(lower)) {
    scores.mood = Math.min(100, scores.mood + 12);
    scores.energy = Math.min(100, scores.energy + 8);
    scores.stress = Math.max(0, scores.stress - 12);
  }

  // Adjust for motivation/planning
  if (/\b(plan|goal|start|try|help me|make a change|moving forward)\b/i.test(lower)) {
    scores.energy = Math.min(100, scores.energy + 6);
    scores.mood = Math.min(100, scores.mood + 4);
  }

  // Clamp all values
  scores.mood = Math.max(0, Math.min(100, Math.round(scores.mood)));
  scores.energy = Math.max(0, Math.min(100, Math.round(scores.energy)));
  scores.stress = Math.max(0, Math.min(100, Math.round(scores.stress)));

  // Confidence is lower for fallback (keywords only)
  const baseConfidence = classification.sentiment === 'Crisis' ? 70 : 50;

  return {
    ...scores,
    confidence: baseConfidence,
    reasoning: 'Fallback keyword-based wellness inference from sentiment',
  };
}

/**
 * Get default wellness signal for safety
 */
export function getDefaultWellness(): WellnessSignal {
  return {
    mood: 55,
    energy: 50,
    stress: 50,
    confidence: 30,
    reasoning: 'Default neutral wellness signal',
  };
}
