import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { Classification } from './classification';
import type { SessionTypeDetection } from './session-type';
import type { ResponseQuality } from './response-eval';
import type { ResourceItem } from '@/lib/companion-foundation';

/**
 * Response Regeneration Tool
 * Automatically retries response generation with different strategies when quality is low
 * Uses feedback from response evaluation to guide regeneration approach
 */

export const RegenerationStrategyEnum = z.enum([
  'more_validation',
  'more_warmth',
  'more_clarity',
  'shorter_response',
  'concrete_steps',
  'resources_focus',
  'empathy_first',
  'reframe_positive',
]);

export const RegenerationAttemptSchema = z.object({
  attempt_number: z.number().min(1).max(3),
  strategy: RegenerationStrategyEnum,
  generated_response: z.string(),
  estimated_quality: z.number().min(0).max(100),
  reason_for_strategy: z.string(),
});

export type RegenerationStrategy = z.infer<typeof RegenerationStrategyEnum>;
export type RegenerationAttempt = z.infer<typeof RegenerationAttemptSchema>;

/**
 * Determine regeneration strategy based on evaluation feedback
 */
export function selectRegenerationStrategy(
  evaluation: ResponseQuality,
  sessionType: SessionTypeDetection
): RegenerationStrategy {
  const weaknesses = evaluation.weaknesses;

  // If validation is lowest and significantly below target, focus on validation
  if (
    evaluation.validation_score < 60 &&
    evaluation.validation_score < evaluation.warmth_score - 10
  ) {
    return 'more_validation';
  }

  // If warmth is low, focus on human tone
  if (evaluation.warmth_score < 60) {
    return 'more_warmth';
  }

  // If clarity is low
  if (evaluation.clarity_score < 60) {
    return 'shorter_response';
  }

  // If relevance is low
  if (evaluation.relevance_score < 60) {
    // Choose strategy based on session type
    if (sessionType.primary_type === 'problem_solving') {
      return 'concrete_steps';
    } else if (sessionType.primary_type === 'validation_seeking') {
      return 'more_validation';
    } else if (sessionType.primary_type === 'information_seeking') {
      return 'resources_focus';
    } else if (sessionType.primary_type === 'venting') {
      return 'empathy_first';
    }
  }

  // Default: focus on warmth and validation
  return 'empathy_first';
}

/**
 * Regenerate response using specified strategy
 */
export async function regenerateResponse(args: {
  userMessage: string;
  failedResponse: string;
  evaluation: ResponseQuality;
  classification: Classification;
  sessionType: SessionTypeDetection;
  attemptNumber: number;
  resources?: ResourceItem[];
}): Promise<RegenerationAttempt> {
  const startTime = Date.now();

  try {
    // Select strategy
    const strategy = selectRegenerationStrategy(args.evaluation, args.sessionType);

    // Build regeneration prompt
    const prompt = buildRegenerationPrompt(
      args.userMessage,
      args.failedResponse,
      args.evaluation,
      strategy,
      args.sessionType,
      args.resources
    );

    // Generate new response
    const newResponse = await callOpenRouter({
      system: `You are GenZ AI Therapist generating a NEW response after the first attempt was not quite right.
      
This is attempt ${args.attemptNumber}/3. Be creative and try a different angle.
Focus on: ${strategyDescription(strategy)}`,
      user: prompt,
      temperature: 0.7, // Higher temp for variation
      maxTokens: 400,
    });

    const response = newResponse.trim();

    // Log regeneration attempt
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'response_regeneration',
      input: {
        attempt_number: args.attemptNumber,
        strategy,
        session_type: args.sessionType.primary_type,
      },
      output: {
        response_length: response.length,
        strategy_used: strategy,
      },
      duration_ms: duration,
      success: true,
    });

    return {
      attempt_number: args.attemptNumber,
      strategy,
      generated_response: response,
      estimated_quality: 70, // Optimistic estimate
      reason_for_strategy: `Regenerating with focus on ${strategy.replace(/_/g, ' ')} to improve response quality`,
    };
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'response_regeneration_error',
        error: error instanceof Error ? error.message : String(error),
      },
      'Response regeneration failed'
    );

    logToolCall({
      tool_name: 'response_regeneration',
      input: { attempt_number: args.attemptNumber },
      output: { error: true },
      duration_ms: duration,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });

    // Fallback: return slightly modified original
    return {
      attempt_number: args.attemptNumber,
      strategy: 'empathy_first',
      generated_response: fallbackRegeneratedResponse(
        args.userMessage,
        args.sessionType
      ),
      estimated_quality: 65,
      reason_for_strategy: 'Fallback response due to API error',
    };
  }
}

/**
 * Build regeneration prompt
 */
function buildRegenerationPrompt(
  userMessage: string,
  failedResponse: string,
  evaluation: ResponseQuality,
  strategy: RegenerationStrategy,
  sessionType: SessionTypeDetection,
  resources?: ResourceItem[]
): string {
  let prompt = `User message: "${userMessage}"

Your previous response (which didn't quite land):
"""
${failedResponse}
"""

That response scored:
- Warmth: ${evaluation.warmth_score}/100
- Validation: ${evaluation.validation_score}/100
- Clarity: ${evaluation.clarity_score}/100
- Relevance: ${evaluation.relevance_score}/100

${evaluation.weaknesses.length > 0 ? `Issues: ${evaluation.weaknesses.join(', ')}` : ''}

Session type: ${sessionType.primary_type}
User needs: ${sessionType.user_needs.join(', ')}

Regenerate the response focusing on: ${strategyDescription(strategy)}

Keep it 1-3 short paragraphs. Use Gen Z voice (lowkey, honest, human).`;

  if (resources && resources.length > 0) {
    prompt += `\n\nAvailable resources:\n${resources.map((r) => `- ${r.description} (${r.url})`).join('\n')}`;
  }

  prompt += '\n\nReturn ONLY the new response, no explanation.';

  return prompt;
}

/**
 * Describe strategy in human terms
 */
function strategyDescription(strategy: RegenerationStrategy): string {
  const descriptions = {
    more_validation: 'making the user feel heard and validated',
    more_warmth: 'sounding more human and genuine',
    more_clarity: 'being clearer and more concise',
    shorter_response: 'shorter and punchier',
    concrete_steps: 'offering concrete next steps',
    resources_focus: 'offering helpful resources and information',
    empathy_first: 'leading with empathy and understanding',
    reframe_positive: 'finding a more hopeful angle',
  };

  return descriptions[strategy];
}

/**
 * Fallback regenerated response
 */
function fallbackRegeneratedResponse(
  userMessage: string,
  sessionType: SessionTypeDetection
): string {
  if (sessionType.primary_type === 'venting') {
    return "I hear you. That sounds really frustrating, and you have every right to feel the way you do. Sometimes you just need to let it out, and that's okay.";
  } else if (sessionType.primary_type === 'problem_solving') {
    return "That's tough. Let's break this down together. What feels like the most pressing part right now? We can tackle it step by step.";
  } else if (sessionType.primary_type === 'validation_seeking') {
    return "You are not overreacting. Your feelings make complete sense. You are definitely not alone in this.";
  } else if (sessionType.primary_type === 'information_seeking') {
    return 'That is a great question. I would recommend checking out some trusted resources for accurate information on this.';
  } else if (sessionType.primary_type === 'crisis') {
    return 'I am really glad you reached out. This is bigger than what I can handle, so please call 988 or text HOME to 741741 right now. You deserve real support.';
  } else {
    return 'I get it. Tell me more about what you are feeling, and we can figure this out together.';
  }
}

/**
 * Should we regenerate?
 */
export function shouldRegenerate(
  evaluation: ResponseQuality,
  attemptNumber: number
): boolean {
  // Already exceeded max attempts
  if (attemptNumber >= 3) return false;

  // Yes, if quality is low
  return evaluation.overall_quality < 65;
}

/**
 * Did regeneration succeed?
 */
export function didRegenerationSucceed(newEvaluation: ResponseQuality): boolean {
  // Success if we reach minimum quality threshold
  return newEvaluation.overall_quality >= 70;
}

/**
 * Get default regeneration attempt
 */
export function getDefaultRegenerationAttempt(attemptNumber: number): RegenerationAttempt {
  return {
    attempt_number: attemptNumber,
    strategy: 'empathy_first',
    generated_response:
      "I am here to help. Tell me more about what is going on, and we can work through this together.",
    estimated_quality: 68,
    reason_for_strategy: 'Default safe regeneration',
  };
}
