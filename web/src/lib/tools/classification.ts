import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';

/**
 * Classification Tool
 * Classifies user messages by sentiment and intent with confidence scoring
 */

// Schemas
export const SentimentSchema = z.enum(['Positive', 'Neutral', 'Negative', 'Crisis']);
export const IntentSchema = z.enum([
  'support',
  'information',
  'chitchat',
  'crisis',
  'motivational',
  'venting',
  'other',
]);

export const ClassificationSchema = z.object({
  sentiment: SentimentSchema,
  intent: IntentSchema,
  confidence: z.number().min(0).max(100),
  reasoning: z.string(),
  alternativeInterpretations: z.array(
    z.object({
      interpretation: z.string(),
      probability: z.number().min(0).max(100),
    })
  ),
});

export type Sentiment = z.infer<typeof SentimentSchema>;
export type Intent = z.infer<typeof IntentSchema>;
export type Classification = z.infer<typeof ClassificationSchema>;

/**
 * Classify message with confidence scoring
 */
export async function classifyWithConfidence(
  message: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>
): Promise<Classification> {
  const startTime = Date.now();

  try {
    // Build context from history
    const historyContext = history
      .slice(-3)
      .map((h) => `${h.role.toUpperCase()}: ${h.content}`)
      .join('\n');

    const prompt = `You are a classification expert for emotional support conversations.
    
Classify the user's message on two dimensions:

1. SENTIMENT (how they're feeling):
   - Positive: hopeful, relieved, grateful, proud, good
   - Neutral: mixed, unclear, or factual tone
   - Negative: overwhelmed, anxious, sad, frustrated, burned out (but not immediate danger)
   - Crisis: immediate self-harm risk, suicide ideation, acute safety concern

2. INTENT (what they're seeking):
   - support: emotional validation and understanding
   - information: facts, explanations, resources
   - chitchat: casual conversation
   - crisis: immediate danger or safety concern
   - motivational: encouragement, inspiration, planning
   - venting: expressing emotions without needing advice
   - other: doesn't fit above

Rate your confidence 0-100 for your classification. Lower confidence means the message was ambiguous.

If confidence is low, suggest alternative interpretations with their probabilities.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "sentiment": "...",
  "intent": "...",
  "confidence": 0-100,
  "reasoning": "Why you classified it this way",
  "alternativeInterpretations": [
    {"interpretation": "Alternative sentiment/intent combo", "probability": 0-100}
  ]
}`;

    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `${prompt}

Recent context:
${historyContext}

Current message: "${message}"

Classify this message.`,
      temperature: 0.2, // Low temperature for consistent classification
      maxTokens: 350,
    });

    // Extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid response format: no JSON found');
    }

    const classification = ClassificationSchema.parse(JSON.parse(jsonMatch[0]));

    // Log successful classification
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'classification',
      input: { message, history_length: history.length },
      output: {
        sentiment: classification.sentiment,
        intent: classification.intent,
        confidence: classification.confidence,
      },
      duration_ms: duration,
      success: true,
    });

    return classification;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'classification_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Classification tool failed, using fallback'
    );

    // Fallback: Use keyword-based classification
    const fallback = fallbackClassification(message);

    logToolCall({
      tool_name: 'classification',
      input: { message, history_length: history.length },
      output: fallback,
      duration_ms: duration,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });

    return fallback;
  }
}

/**
 * Fallback classification using keyword patterns
 */
function fallbackClassification(message: string): Classification {
  const lower = message.toLowerCase();

  // Crisis detection
  const crisisPatterns =
    /\b(suicide|kill myself|end my life|self harm|hurt myself|i am not safe|want to die|cutting|harming)\b/i;
  if (crisisPatterns.test(message)) {
    return {
      sentiment: 'Crisis',
      intent: 'crisis',
      confidence: 90,
      reasoning: 'Crisis keywords detected',
      alternativeInterpretations: [],
    };
  }

  // Sentiment patterns
  let sentiment: Sentiment = 'Neutral';
  let confidenceDelta = 0;

  const positivePatterns = /\b(proud|grateful|relieved|happy|better|good|great|excellent|wonderful)\b/i;
  const negativePatterns =
    /\b(overwhelmed|anxious|panic|sad|lonely|burned out|hopeless|stressed|drained|fried|depressed)\b/i;

  if (positivePatterns.test(message)) {
    sentiment = 'Positive';
    confidenceDelta = 20;
  } else if (negativePatterns.test(message)) {
    sentiment = 'Negative';
    confidenceDelta = 15;
  }

  // Intent patterns
  let intent: Intent = 'support';

  const infoPatterns = /\b(how|what|why|can you explain|resource|resources|find|where)\b/i;
  const ventingPatterns = /\b(just|vent|need to get this out|frustrated|ugh|argh)\b/i;
  const motivationalPatterns = /\b(help me|plan|start|make|change|do)\b/i;

  if (infoPatterns.test(message)) {
    intent = 'information';
    confidenceDelta += 10;
  } else if (ventingPatterns.test(message)) {
    intent = 'venting';
    confidenceDelta += 10;
  } else if (motivationalPatterns.test(message)) {
    intent = 'motivational';
    confidenceDelta += 5;
  }

  const confidence = Math.max(45, Math.min(70, 55 + confidenceDelta));

  return {
    sentiment,
    intent,
    confidence,
    reasoning: 'Fallback keyword-based classification',
    alternativeInterpretations: [],
  };
}

/**
 * Get classification with default for safety
 */
export function getDefaultClassification(): Classification {
  return {
    sentiment: 'Neutral',
    intent: 'support',
    confidence: 50,
    reasoning: 'Default safe classification',
    alternativeInterpretations: [],
  };
}
