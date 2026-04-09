import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { Classification } from './classification';

/**
 * Session Type Detection Tool
 * Identifies the user's primary conversational mode to enable different response strategies
 * Venting != Problem-Solving != Validation != Information != Crisis
 */

export const SessionTypeEnum = z.enum([
  'venting',
  'problem_solving',
  'validation_seeking',
  'information_seeking',
  'crisis',
  'chitchat',
]);

export const SessionTypeSchema = z.object({
  primary_type: SessionTypeEnum,
  secondary_types: z.array(SessionTypeEnum).max(2),
  confidence: z.number().min(0).max(100),
  reasoning: z.string(),
  user_needs: z.array(z.string()).max(3),
  recommended_strategy: z.string(),
});

export type SessionType = z.infer<typeof SessionTypeEnum>;
export type SessionTypeDetection = z.infer<typeof SessionTypeSchema>;

// Venting patterns: expressing without seeking solution
const VENTING_PATTERNS = [
  /\b(just need to vent|need to get this out|let me vent|rant|frustrated|ugh|argh|this is so annoying)\b/i,
  /\b(can'?t believe|seriously\?|like what|omg)\b/i,
  /\b(i hate|i'?m so fed up|fed up with)\b/i,
];

// Problem-solving patterns: seeking concrete solutions
const PROBLEM_SOLVING_PATTERNS = [
  /\b(how can i|how do i|what should i|what can i do|help me solve|help me fix|need a solution)\b/i,
  /\b(plan|strategy|approach|step by step|action|make a change|try|start)\b/i,
  /\b(what if|if i|can i|would it help if|would that work)\b/i,
];

// Validation-seeking patterns: need affirmation, not advice
const VALIDATION_PATTERNS = [
  /\b(am i right|am i crazy|is that normal|is it okay|does that make sense)\b/i,
  /\b(you understand|right\?|don'?t you think|do you agree|is it wrong)\b/i,
  /\b(i feel like|i think|shouldn'?t|i shouldn'?t)\b/i,
];

// Information-seeking patterns: asking for facts, resources, explanations
const INFORMATION_PATTERNS = [
  /\b(what is|how does|why do|tell me about|explain|where can i|resources|research|studies)\b/i,
  /\b(how does.*work|what does.*mean|what are|who|when)\b/i,
];

const SESSION_TYPE_PROMPT = `You are an expert at identifying conversational patterns in emotional support.

Determine the user's PRIMARY conversational mode:

VENTING: User needs to EXPRESS frustration/feelings without needing advice. They want to be heard, not fixed.
- Phrases: "just need to vent", "let me rant", "I am so frustrated", "can't believe this"
- Emotions: high energy complaint, need for release
- What they DON'T want: solutions, advice, lecture

PROBLEM-SOLVING: User wants CONCRETE SOLUTIONS or a path forward.
- Phrases: "how can I", "what should I do", "help me fix", "make a plan"
- Emotions: frustration mixed with agency
- What they want: strategies, steps, actionable ideas

VALIDATION-SEEKING: User needs AFFIRMATION that their feelings/thoughts are okay.
- Phrases: "am I right", "is this normal", "does that make sense", "do you agree"
- Emotions: uncertainty mixed with feeling
- What they want: reassurance, acknowledgment, "you're not crazy"

INFORMATION-SEEKING: User wants FACTS, EXPLANATIONS, or RESOURCES.
- Phrases: "what is", "how does", "where can I find", "tell me about"
- Emotions: curiosity, need for knowledge
- What they want: clear info, resources, research

CRISIS: User in immediate emotional danger (handled separately in crisis eval).

CHITCHAT: Casual, low-stakes conversation.

Identify secondary modes if present. Suggest what the user actually needs.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "primary_type": "venting|problem_solving|validation_seeking|information_seeking|crisis|chitchat",
  "secondary_types": [],
  "confidence": 0-100,
  "reasoning": "Why this is the primary type",
  "user_needs": ["need1", "need2"],
  "recommended_strategy": "How to respond effectively"
}`;

/**
 * Detect the user's primary conversational session type
 */
export async function detectSessionType(args: {
  userMessage: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  classification: Classification;
}): Promise<SessionTypeDetection> {
  const startTime = Date.now();

  try {
    // Quick keyword-based detection first
    const quickDetection = quickDetectSessionType(args.userMessage);

    // If high confidence from keywords, return early
    if (quickDetection.confidence > 80) {
      logToolCall({
        tool_name: 'session_type',
        input: { message_length: args.userMessage.length },
        output: {
          primary_type: quickDetection.primary_type,
          confidence: quickDetection.confidence,
        },
        duration_ms: Date.now() - startTime,
        success: true,
      });

      return quickDetection;
    }

    // For ambiguous cases, use LLM for nuance
    const historyContext = args.history
      .slice(-3)
      .map((h) => `${h.role.toUpperCase()}: ${h.content.slice(0, 180)}`)
      .join('\n');

    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `Classification: sentiment=${args.classification.sentiment}, intent=${args.classification.intent}

Recent context:
${historyContext}

Latest message: "${args.userMessage}"

${SESSION_TYPE_PROMPT}`,
      temperature: 0.2,
      maxTokens: 350,
    });

    // Extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid response format: no JSON found');
    }

    const detection = SessionTypeSchema.parse(JSON.parse(jsonMatch[0]));

    // Log successful detection
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'session_type',
      input: { message_length: args.userMessage.length },
      output: {
        primary_type: detection.primary_type,
        confidence: detection.confidence,
      },
      duration_ms: duration,
      success: true,
    });

    return detection;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'session_type_fallback',
        error: error instanceof Error ? error.message : String(error),
      },
      'Session type detection failed, using fallback'
    );

    // Fallback: Use keyword-based detection
    const fallback = quickDetectSessionType(args.userMessage);

    logToolCall({
      tool_name: 'session_type',
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
 * Quick keyword-based session type detection
 */
function quickDetectSessionType(message: string): SessionTypeDetection {
  const lower = message.toLowerCase();

  // Count pattern matches
  const ventingCount = VENTING_PATTERNS.filter((p) => p.test(lower)).length;
  const problemSolvingCount = PROBLEM_SOLVING_PATTERNS.filter((p) => p.test(lower)).length;
  const validationCount = VALIDATION_PATTERNS.filter((p) => p.test(lower)).length;
  const informationCount = INFORMATION_PATTERNS.filter((p) => p.test(lower)).length;

  // Crisis overrides all
  if (/\b(suicide|kill myself|self harm|want to die)\b/i.test(lower)) {
    return {
      primary_type: 'crisis',
      secondary_types: [],
      confidence: 95,
      reasoning: 'Crisis keywords detected',
      user_needs: ['immediate safety', 'crisis support'],
      recommended_strategy: 'Prioritize safety and crisis resources',
    };
  }

  // Determine primary type by highest count
  const counts = [
    { type: 'venting' as const, count: ventingCount },
    { type: 'problem_solving' as const, count: problemSolvingCount },
    { type: 'validation_seeking' as const, count: validationCount },
    { type: 'information_seeking' as const, count: informationCount },
  ];

  counts.sort((a, b) => b.count - a.count);
  const primary = counts[0];
  const secondary = counts.slice(1, 3).filter((c) => c.count > 0);

  // Calculate confidence based on pattern strength
  let confidence = 50;
  if (primary.count >= 2) confidence = 85;
  else if (primary.count === 1) confidence = 65;

  // If no patterns match strongly, default to validation/support seeking
  if (primary.count === 0) {
    return {
      primary_type: 'validation_seeking',
      secondary_types: [],
      confidence: 40,
      reasoning: 'No strong pattern match, defaulting to support seeking',
      user_needs: ['understanding', 'support'],
      recommended_strategy: 'Validate feelings and explore what they need',
    };
  }

  const strategies = {
    venting: 'Listen actively, validate frustration, avoid jumping to solutions. Let them express fully.',
    problem_solving:
      'Ask clarifying questions, offer concrete steps, explore options together.',
    validation_seeking: 'Affirm their feelings and perspective. Mirror back what you hear.',
    information_seeking: 'Provide clear facts, explain thoughtfully, offer resources.',
    crisis:
      'Prioritize immediate safety. Encourage crisis hotline or emergency services.',
    chitchat: 'Keep it light, genuine, and human. Build rapport.',
  };

  return {
    primary_type: primary.type,
    secondary_types: secondary.map((s) => s.type),
    confidence,
    reasoning: `Pattern matching found ${primary.count} ${primary.type} indicators`,
    user_needs: getUserNeeds(primary.type),
    recommended_strategy: strategies[primary.type],
  };
}

/**
 * Infer user needs based on session type
 */
function getUserNeeds(type: SessionType): string[] {
  const needs = {
    venting: ['to be heard', 'to express frustration', 'emotional release'],
    problem_solving: ['concrete solutions', 'actionable steps', 'clarity on next moves'],
    validation_seeking: ['affirmation', 'reassurance', 'to know they are not alone'],
    information_seeking: ['facts', 'explanations', 'resources or guidance'],
    crisis: ['immediate safety', 'crisis support', 'emergency help'],
    chitchat: ['connection', 'casual conversation', 'human interaction'],
  };

  return needs[type].slice(0, 3);
}

/**
 * Get default session type
 */
export function getDefaultSessionType(): SessionTypeDetection {
  return {
    primary_type: 'validation_seeking',
    secondary_types: [],
    confidence: 40,
    reasoning: 'Default safe session type assumption',
    user_needs: ['understanding', 'support'],
    recommended_strategy: 'Validate and explore what they need',
  };
}
