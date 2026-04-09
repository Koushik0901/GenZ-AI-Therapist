import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { Classification } from './classification';
import type { SessionTypeDetection } from './session-type';

/**
 * Clarification Questions Tool
 * Generates relevant questions when classification confidence is low (<50%)
 * Helps user clarify what they actually need before generating response
 */

export const ClarificationQuestionsSchema = z.object({
  needs_clarification: z.boolean().describe('Should we ask clarifying questions?'),
  confidence_issue: z.string().describe('What about the message is ambiguous'),
  questions: z
    .array(z.string())
    .max(3)
    .describe('Up to 3 clarifying questions'),
  suggested_focus: z.string().optional().describe('What to focus on once clarified'),
});

export type ClarificationQuestions = z.infer<typeof ClarificationQuestionsSchema>;

const CLARIFICATION_PROMPT = `You are skilled at clarifying what users actually need in emotional support conversations.

When a user's message is ambiguous or unclear (low confidence classification), generate 1-3 clarifying questions.

These questions should:
- Be warm and conversational (not clinical)
- Help narrow down what the user actually needs
- Explore intent (do they want validation? advice? to vent? resources?)
- Keep it brief and non-invasive
- Use Gen Z voice (lowkey, genuine, conversational)

Questions should be open-ended when possible but can be yes/no for quick clarification.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "needs_clarification": true,
  "confidence_issue": "What is ambiguous",
  "questions": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "suggested_focus": "What to focus on once clarified"
}`;

/**
 * Generate clarification questions for ambiguous messages
 */
export async function generateClarificationQuestions(args: {
  userMessage: string;
  classification: Classification;
  sessionType: SessionTypeDetection;
}): Promise<ClarificationQuestions> {
  const startTime = Date.now();

  try {
    // Skip if confidence is already high
    if (args.classification.confidence > 65) {
      return {
        needs_clarification: false,
        confidence_issue: 'Confidence already high',
        questions: [],
      };
    }

    // Use LLM to generate clarifying questions
    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `User message: "${args.userMessage}"

Current classification:
- Sentiment: ${args.classification.sentiment} (confidence ${args.classification.confidence}%)
- Intent: ${args.classification.intent}
- Reasoning: ${args.classification.reasoning}

Current session type guess: ${args.sessionType.primary_type} (confidence ${args.sessionType.confidence}%)

${CLARIFICATION_PROMPT}`,
      temperature: 0.4,
      maxTokens: 300,
    });

    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid JSON response');
    }

    const clarification = ClarificationQuestionsSchema.parse(
      JSON.parse(jsonMatch[0])
    );

    // Log successful generation
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'clarification_questions',
      input: {
        message_length: args.userMessage.length,
        classification_confidence: args.classification.confidence,
      },
      output: {
        needs_clarification: clarification.needs_clarification,
        question_count: clarification.questions.length,
      },
      duration_ms: duration,
      success: true,
    });

    return clarification;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'clarification_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Clarification generation failed, using fallback'
    );

    // Fallback: generate generic clarifying questions
    const fallback = fallbackClarificationQuestions(args.userMessage, args.classification);

    logToolCall({
      tool_name: 'clarification_questions',
      input: {
        message_length: args.userMessage.length,
        classification_confidence: args.classification.confidence,
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
 * Fallback clarification question generation
 */
function fallbackClarificationQuestions(
  message: string,
  classification: Classification
): ClarificationQuestions {
  const lower = message.toLowerCase();

  // Determine what's unclear
  let confidenceIssue = 'Message is ambiguous';
  let questions: string[] = [];
  let suggestedFocus: string | undefined;

  // Case 1: Low sentiment confidence
  if (
    (classification.sentiment === 'Neutral' || classification.sentiment === 'Negative') &&
    classification.confidence < 50
  ) {
    confidenceIssue = 'Not sure if you are expressing feelings or asking for something';
    questions = [
      'Are you looking for someone to listen and validate, or do you want advice?',
      'Are you venting about this situation or trying to solve it?',
    ];
    suggestedFocus = 'Whether you need to be heard or need solutions';
  }

  // Case 2: Unclear intent
  if (classification.intent === 'other' || classification.confidence < 45) {
    confidenceIssue = 'Not clear what you are asking for';
    questions = [
      'What would help most right now - just listening, resources, or brainstorming ideas?',
      'Are you trying to understand something better, or do you need support?',
    ];
    suggestedFocus = 'Your actual need (support/info/solutions/validation)';
  }

  // Case 3: Formal or unclear language
  if (/\b(possibly|maybe|somewhat|perhaps|unclear)\b/i.test(lower)) {
    confidenceIssue = 'Message seems indirect or uncertain';
    questions = [
      'What is really on your mind right now?',
      'What would make this conversation most helpful for you?',
    ];
    suggestedFocus = 'Your actual feelings and needs';
  }

  // Default questions if nothing matches
  if (questions.length === 0) {
    confidenceIssue = 'Message interpretation is unclear';
    questions = [
      'Tell me more about what is going on.',
      'What do you need most right now?',
      'How can I help?',
    ];
    suggestedFocus = 'Understanding your core need';
  }

  return {
    needs_clarification: true,
    confidence_issue: confidenceIssue,
    questions: questions.slice(0, 3),
    suggested_focus: suggestedFocus,
  };
}

/**
 * Should we ask clarifying questions?
 */
export function shouldAskClarification(classification: Classification): boolean {
  return classification.confidence < 50;
}

/**
 * Format questions for user display
 */
export function formatQuestionsForDisplay(questions: string[]): string {
  if (questions.length === 0) return '';

  if (questions.length === 1) {
    return questions[0];
  }

  return (
    'A few quick questions:\n\n' +
    questions.map((q, i) => `${i + 1}. ${q}`).join('\n\n')
  );
}
