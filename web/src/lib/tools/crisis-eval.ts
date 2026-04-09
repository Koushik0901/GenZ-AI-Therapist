import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { Classification } from './classification';
import type { WellnessSignal } from './wellness';

/**
 * Crisis Evaluation Tool
 * Multi-factor assessment of crisis severity with confidence scoring
 * Uses: explicit keywords, implicit hopelessness, wellness signals, conversation patterns, LLM evaluation
 */

// Schemas
export const CrisisSeverityLevel = z.enum([
  'safe',
  'at_risk',
  'high_risk',
  'critical',
]);

export const CrisisEvaluationSchema = z.object({
  severity: CrisisSeverityLevel,
  score: z.number().min(0).max(100).describe('Aggregate risk score 0-100'),
  confidence: z.number().min(0).max(100).describe('Confidence in this assessment'),
  explicit_keywords: z.boolean().describe('Explicit crisis keywords detected'),
  implicit_hopelessness: z.boolean().describe('Implicit hopelessness patterns detected'),
  wellness_signal: z.boolean().describe('Wellness signal indicates crisis (very low mood/energy, high stress)'),
  pattern_escalation: z.boolean().describe('Conversation shows escalation pattern'),
  reasoning: z.string().describe('Detailed reasoning for severity assessment'),
  recommended_actions: z.array(z.string()).describe('Recommended immediate actions'),
});

export type CrisisSeverity = z.infer<typeof CrisisSeverityLevel>;
export type CrisisEvaluation = z.infer<typeof CrisisEvaluationSchema>;

// Explicit crisis keywords
const EXPLICIT_CRISIS_PATTERNS = [
  /\b(suicide|suicidal|kill myself|end my life)\b/i,
  /\b(self harm|self-harm|hurt myself|cutting|cutting myself)\b/i,
  /\b(want to die|wanna die|wish i was dead|i am not safe|don'?t feel safe)\b/i,
  /\b(no reason to live|no point|why bother|better off dead)\b/i,
  /\b(overdose|pills|poison|jump|hang|drive off)\b/i,
];

// Implicit hopelessness patterns (without explicit keywords)
const IMPLICIT_HOPELESSNESS_PATTERNS = [
  /\b(everything is pointless|nothing matters|can'?t take it anymore|can'?t go on)\b/i,
  /\b(nobody cares|alone forever|never get better|stuck forever|trapped)\b/i,
  /\b(worthless|burden|everyone would be better off|giving up)\b/i,
  /\b(empty|numb|dead inside|lost all hope|hopeless)\b/i,
];

const CRISIS_PROMPT = `You are an expert in crisis assessment for emotional support conversations.

Assess the crisis severity based on:
- The latest message and recent history
- Explicit risk indicators (suicidal ideation, self-harm intent)
- Implicit hopelessness and despair
- Wellness signals (very low mood, low energy, high stress)
- Conversation escalation patterns

SEVERITY LEVELS:
- safe: No crisis indicators
- at_risk: Distressed but stable; emotional pain present; some hopelessness
- high_risk: Explicit ideation or clear intent; significant hopelessness; acute plan discussion
- critical: Immediate safety concern; active intent; acute access to means; imminent danger

Return ONLY valid JSON (no markdown, no code blocks):
{
  "severity": "safe|at_risk|high_risk|critical",
  "score": 0-100,
  "confidence": 0-100,
  "reasoning": "Detailed assessment reasoning",
  "recommended_actions": ["action1", "action2", ...]
}`;

/**
 * Evaluate crisis severity from conversation context
 */
export async function evaluateCrisis(args: {
  userMessage: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  classification: Classification;
  wellness: WellnessSignal;
}): Promise<CrisisEvaluation> {
  const startTime = Date.now();

  try {
    // Factor 1: Explicit crisis keywords
    const hasExplicitKeywords = EXPLICIT_CRISIS_PATTERNS.some((pattern) =>
      pattern.test(args.userMessage)
    );

    // Factor 2: Implicit hopelessness patterns
    const hasImplicitHopelessness = IMPLICIT_HOPELESSNESS_PATTERNS.some((pattern) =>
      pattern.test(args.userMessage)
    );

    // Factor 3: Wellness signal indicates crisis
    const wellnessIndicatesCrisis =
      args.wellness.mood < 25 && args.wellness.energy < 25 && args.wellness.stress > 85;

    // Factor 4: Pattern escalation (check recent history for worsening)
    const patternEscalation = detectEscalationPattern(args.history);

    // If multiple strong signals, escalate to high risk immediately
    const strongSignalCount = [
      hasExplicitKeywords,
      hasImplicitHopelessness,
      wellnessIndicatesCrisis,
      patternEscalation,
    ].filter(Boolean).length;

    if (hasExplicitKeywords || (strongSignalCount >= 3 && args.wellness.mood < 15)) {
      // Skip LLM call for clear cases
      const evaluation: CrisisEvaluation = {
        severity: hasExplicitKeywords ? 'critical' : 'high_risk',
        score: hasExplicitKeywords ? 95 : 80,
        confidence: 95,
        explicit_keywords: hasExplicitKeywords,
        implicit_hopelessness: hasImplicitHopelessness,
        wellness_signal: wellnessIndicatesCrisis,
        pattern_escalation: patternEscalation,
        reasoning: hasExplicitKeywords
          ? 'Explicit suicide/self-harm keywords detected - CRITICAL'
          : 'Multiple strong crisis indicators detected',
        recommended_actions: [
          'Encourage immediate contact with crisis services',
          'Suggest calling emergency services or crisis hotline',
          'Ask about immediate safety and access to means',
        ],
      };

      logToolCall({
        tool_name: 'crisis_eval',
        input: {
          message_length: args.userMessage.length,
          has_explicit: hasExplicitKeywords,
          has_implicit: hasImplicitHopelessness,
        },
        output: { severity: evaluation.severity, score: evaluation.score },
        duration_ms: Date.now() - startTime,
        success: true,
      });

      return evaluation;
    }

    // For ambiguous cases, use LLM evaluation
    const historyContext = args.history
      .slice(-4)
      .map((h) => `${h.role.toUpperCase()}: ${h.content.slice(0, 200)}`)
      .join('\n');

    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `Classification: sentiment=${args.classification.sentiment}, intent=${args.classification.intent}
Wellness: mood=${args.wellness.mood}, energy=${args.wellness.energy}, stress=${args.wellness.stress}

Recent context:
${historyContext}

Latest message: "${args.userMessage}"

Explicit keywords detected: ${hasExplicitKeywords}
Implicit hopelessness detected: ${hasImplicitHopelessness}
Conversation escalation: ${patternEscalation}

${CRISIS_PROMPT}`,
      temperature: 0.1,
      maxTokens: 400,
    });

    // Extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid response format: no JSON found');
    }

    const parsed = JSON.parse(jsonMatch[0]);

    const evaluation: CrisisEvaluation = {
      severity: parsed.severity as CrisisSeverity,
      score: Math.round(parsed.score),
      confidence: Math.round(parsed.confidence),
      explicit_keywords: hasExplicitKeywords,
      implicit_hopelessness: hasImplicitHopelessness,
      wellness_signal: wellnessIndicatesCrisis,
      pattern_escalation: patternEscalation,
      reasoning: parsed.reasoning || 'LLM-based assessment',
      recommended_actions: parsed.recommended_actions || [],
    };

    // Validate result
    CrisisEvaluationSchema.parse(evaluation);

    // Log successful evaluation
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'crisis_eval',
      input: {
        message_length: args.userMessage.length,
        has_explicit: hasExplicitKeywords,
      },
      output: {
        severity: evaluation.severity,
        score: evaluation.score,
        confidence: evaluation.confidence,
      },
      duration_ms: duration,
      success: true,
    });

    return evaluation;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'crisis_eval_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Crisis evaluation failed, using fallback'
    );

    // Fallback: Use pattern-based assessment
    const fallback = fallbackCrisisEval(args.userMessage, args.classification, args.wellness);

    logToolCall({
      tool_name: 'crisis_eval',
      input: { message_length: args.userMessage.length },
      output: fallback,
      duration_ms: duration,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });

    return fallback;
  }
}

/**
 * Detect escalation patterns in conversation history
 */
function detectEscalationPattern(
  history: Array<{ role: 'user' | 'assistant'; content: string }>
): boolean {
  if (history.length < 4) return false;

  // Check if recent user messages show increasing hopelessness
  const recentUserMessages = history
    .filter((h) => h.role === 'user')
    .slice(-3)
    .map((h) => h.content.toLowerCase());

  if (recentUserMessages.length < 2) return false;

  const hopelessnessKeywords = [
    'hopeless',
    'pointless',
    'nothing matters',
    'give up',
    'tired',
    'done',
  ];

  const hopelessnessCountByMessage = recentUserMessages.map((msg) =>
    hopelessnessKeywords.filter((kw) => msg.includes(kw)).length
  );

  // Escalation if hopelessness indicators increase across messages
  return (
    hopelessnessCountByMessage[hopelessnessCountByMessage.length - 1] >
    hopelessnessCountByMessage[0]
  );
}

/**
 * Fallback crisis evaluation using pattern matching
 */
function fallbackCrisisEval(
  message: string,
  classification: Classification,
  wellness: WellnessSignal
): CrisisEvaluation {
  const hasExplicitKeywords = EXPLICIT_CRISIS_PATTERNS.some((pattern) => pattern.test(message));
  const hasImplicitHopelessness = IMPLICIT_HOPELESSNESS_PATTERNS.some((pattern) =>
    pattern.test(message)
  );
  const wellnessIndicatesCrisis =
    wellness.mood < 25 && wellness.energy < 25 && wellness.stress > 85;

  let severity: CrisisSeverity = 'safe';
  let score = 15;

  if (hasExplicitKeywords) {
    severity = 'critical';
    score = 95;
  } else if (hasImplicitHopelessness && wellnessIndicatesCrisis) {
    severity = 'high_risk';
    score = 75;
  } else if (classification.sentiment === 'Crisis' || hasImplicitHopelessness) {
    severity = 'at_risk';
    score = 45;
  } else if (classification.sentiment === 'Negative') {
    severity = 'at_risk';
    score = 35;
  }

  return {
    severity,
    score,
    confidence: 65,
    explicit_keywords: hasExplicitKeywords,
    implicit_hopelessness: hasImplicitHopelessness,
    wellness_signal: wellnessIndicatesCrisis,
    pattern_escalation: false,
    reasoning: `Pattern-based assessment: ${severity}`,
    recommended_actions:
      severity === 'critical' || severity === 'high_risk'
        ? [
            'Encourage immediate contact with crisis services',
            'Suggest calling emergency services or crisis hotline',
          ]
        : [],
  };
}

/**
 * Get default safe evaluation
 */
export function getDefaultCrisisEval(): CrisisEvaluation {
  return {
    severity: 'safe',
    score: 5,
    confidence: 100,
    explicit_keywords: false,
    implicit_hopelessness: false,
    wellness_signal: false,
    pattern_escalation: false,
    reasoning: 'Default safe assessment',
    recommended_actions: [],
  };
}
