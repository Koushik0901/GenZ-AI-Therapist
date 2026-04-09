import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { SessionTypeDetection } from './session-type';
import type { CrisisEvaluation } from './crisis-eval';

/**
 * Response Evaluation Tool
 * Self-evaluates generated responses for warmth, validation, clarity, relevance before returning to user
 * Enables response regeneration loop if quality is too low
 */

export const ResponseQualitySchema = z.object({
  warmth_score: z.number().min(0).max(100).describe('Does it sound human and caring?'),
  validation_score: z.number().min(0).max(100).describe('Does it validate the user?'),
  clarity_score: z.number().min(0).max(100).describe('Is it clear and easy to follow?'),
  relevance_score: z.number().min(0).max(100).describe('Does it address what they need?'),
  overall_quality: z.number().min(0).max(100).describe('Combined quality score'),
  strengths: z.array(z.string()).max(3).describe('What works well'),
  weaknesses: z.array(z.string()).max(3).describe('What could improve'),
  should_regenerate: z.boolean().describe('Should we try a different response?'),
  regeneration_guidance: z.string().optional().describe('How to improve if regenerating'),
});

export type ResponseQuality = z.infer<typeof ResponseQualitySchema>;

const RESPONSE_EVAL_PROMPT = `You are an expert evaluator of emotional support responses.

Score the provided response on four dimensions:

1. WARMTH (0-100): Does it sound human, caring, genuine?
   - Avoid: clinical tone, "I understand" repeated, overly formal
   - Aim for: real words (lowkey, yeah, honestly), conversational, emotionally present

2. VALIDATION (0-100): Does it acknowledge and affirm the user's feelings?
   - Include: "that makes sense", "your feelings are valid", reflection of what they said
   - Avoid: jumping to solutions, dismissing feelings, toxic positivity

3. CLARITY (0-100): Is it easy to follow and understand?
   - Short paragraphs, clear structure, no jargon
   - Avoid: rambling, confusing advice, unclear pronouns

4. RELEVANCE (0-100): Does it actually address what they need?
   - For VENTING: let them vent, don't offer solutions
   - For PROBLEM-SOLVING: offer concrete steps or ideas
   - For VALIDATION: affirm and reflect, don't pivot to advice
   - For INFORMATION: clear facts and resources
   - For CRISIS: safety first, resources, emergency numbers

Calculate OVERALL QUALITY as weighted average: warmth 25%, validation 35%, clarity 20%, relevance 20%

If overall < 65, mark should_regenerate=true with guidance.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "warmth_score": 0-100,
  "validation_score": 0-100,
  "clarity_score": 0-100,
  "relevance_score": 0-100,
  "overall_quality": 0-100,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "should_regenerate": true/false,
  "regeneration_guidance": "How to improve"
}`;

/**
 * Evaluate a response for quality
 */
export async function evaluateResponse(args: {
  userMessage: string;
  responseText: string;
  sessionType: SessionTypeDetection;
  crisis: CrisisEvaluation;
}): Promise<ResponseQuality> {
  const startTime = Date.now();

  try {
    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `User message: "${args.userMessage}"
Session type: ${args.sessionType.primary_type}
Crisis severity: ${args.crisis.severity}
User needs: ${args.sessionType.user_needs.join(', ')}

Generated response to evaluate:
"""
${args.responseText}
"""

${RESPONSE_EVAL_PROMPT}`,
      temperature: 0.3,
      maxTokens: 400,
    });

    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid response format: no JSON found');
    }

    const evaluation = ResponseQualitySchema.parse(JSON.parse(jsonMatch[0]));

    // Log evaluation
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'response_eval',
      input: {
        response_length: args.responseText.length,
        session_type: args.sessionType.primary_type,
        crisis_severity: args.crisis.severity,
      },
      output: {
        overall_quality: evaluation.overall_quality,
        should_regenerate: evaluation.should_regenerate,
      },
      duration_ms: duration,
      success: true,
    });

    return evaluation;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'response_eval_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Response evaluation failed, using fallback'
    );

    // Fallback: use keyword-based evaluation
    const fallback = fallbackResponseEval(
      args.responseText,
      args.sessionType.primary_type,
      args.crisis.severity
    );

    logToolCall({
      tool_name: 'response_eval',
      input: {
        response_length: args.responseText.length,
        session_type: args.sessionType.primary_type,
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
 * Fallback response evaluation using heuristics
 */
function fallbackResponseEval(
  responseText: string,
  sessionType: string,
  crisisSeverity: string
): ResponseQuality {
  const lower = responseText.toLowerCase();

  // Warmth indicators
  let warmth = 60;
  if (/\b(lowkey|honestly|yeah|not gonna lie|that makes sense|i hear you)\b/i.test(lower)) {
    warmth += 15;
  }
  if (/\b(i understand|you are right|your feelings|valid)\b/i.test(lower)) {
    warmth += 10;
  }
  if (/\b(here'?s|now|you need to|you should)\b/i.test(lower)) {
    warmth -= 10; // Clinical tone
  }

  // Validation indicators
  let validation = 55;
  if (
    /\b(that makes sense|you are not|your feelings|valid|right to|understandable|makes total sense)\b/i.test(
      lower
    )
  ) {
    validation += 20;
  }
  if (
    /\b(but you should|however|actually|the real issue|you need to fix)\b/i.test(lower)
  ) {
    validation -= 15; // Negates validation
  }

  // Clarity indicators
  let clarity = 65;
  const paragraphs = responseText.split('\n\n').length;
  if (paragraphs > 5) clarity -= 10; // Too long
  if (paragraphs < 2) clarity -= 5; // Too short
  const sentences = responseText.split(/[.!?]/).length;
  if (sentences > 8) clarity -= 5; // Too many sentences per paragraph (average)

  // Relevance by session type
  let relevance = 60;
  if (sessionType === 'venting') {
    if (/\b(let'?s solve|here'?s what you should|action plan)\b/i.test(lower)) {
      relevance -= 20; // Shouldn't offer solutions during venting
    }
    if (/\b(sounds frustrating|that'?s valid|i hear|you have every right)\b/i.test(lower)) {
      relevance += 15; // Good venting response
    }
  } else if (sessionType === 'problem_solving') {
    if (/\b(here'?s|step|try|what if|option)\b/i.test(lower)) {
      relevance += 20; // Good problem-solving response
    }
    if (/\b(just let it out|vent away|feel your feelings)\b/i.test(lower)) {
      relevance -= 15; // Shouldn't avoid solutions
    }
  } else if (sessionType === 'validation_seeking') {
    if (/\b(you are not alone|that'?s not crazy|normal to feel)\b/i.test(lower)) {
      relevance += 20; // Good validation
    }
    if (/\b(here'?s how to fix|here'?s the solution)\b/i.test(lower)) {
      relevance -= 15; // Shouldn't jump to solutions
    }
  } else if (sessionType === 'information_seeking') {
    if (/\b(research|studies|data|according to|experts|found that)\b/i.test(lower)) {
      relevance += 20; // Good information response
    }
  } else if (sessionType === 'crisis') {
    if (
      /\b(crisis|emergency|hotline|call|text|988|1-800|immediate help)\b/i.test(
        lower
      )
    ) {
      relevance += 25; // Good crisis response
    }
  }

  // Clamp scores
  warmth = Math.max(0, Math.min(100, Math.round(warmth)));
  validation = Math.max(0, Math.min(100, Math.round(validation)));
  clarity = Math.max(0, Math.min(100, Math.round(clarity)));
  relevance = Math.max(0, Math.min(100, Math.round(relevance)));

  // Calculate overall quality: warmth 25%, validation 35%, clarity 20%, relevance 20%
  const overall = Math.round(
    warmth * 0.25 + validation * 0.35 + clarity * 0.2 + relevance * 0.2
  );

  // Build strengths and weaknesses
  const strengths: string[] = [];
  const weaknesses: string[] = [];

  if (warmth > 70) strengths.push('Warm and genuine tone');
  else if (warmth < 50) weaknesses.push('Tone feels too clinical or detached');

  if (validation > 70) strengths.push('Validates user feelings');
  else if (validation < 50) weaknesses.push('Doesn\'t sufficiently validate feelings');

  if (clarity > 70) strengths.push('Clear and easy to follow');
  else if (clarity < 50) weaknesses.push('Could be clearer or more concise');

  if (relevance > 70) strengths.push('Directly addresses what they need');
  else if (relevance < 50) weaknesses.push('Doesn\'t quite hit the mark for their needs');

  const should_regenerate = overall < 65;

  return {
    warmth_score: warmth,
    validation_score: validation,
    clarity_score: clarity,
    relevance_score: relevance,
    overall_quality: overall,
    strengths: strengths.slice(0, 3),
    weaknesses: weaknesses.slice(0, 3),
    should_regenerate,
    regeneration_guidance: should_regenerate
      ? `Try a response that focuses more on ${
          validation < warmth ? 'warmth and authenticity' : 'validating their feelings'
        }`
      : undefined,
  };
}

/**
 * Get default passing evaluation
 */
export function getDefaultResponseEval(): ResponseQuality {
  return {
    warmth_score: 70,
    validation_score: 75,
    clarity_score: 80,
    relevance_score: 75,
    overall_quality: 75,
    strengths: ['Genuine tone', 'Validates feelings', 'Clear structure'],
    weaknesses: [],
    should_regenerate: false,
  };
}
