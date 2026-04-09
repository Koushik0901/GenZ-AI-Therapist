import { z } from 'zod';
import { callOpenRouter } from '@/lib/openrouter';
import { logger, logToolCall } from '@/lib/logging';
import type { WellnessSignal } from './wellness';
import type { SessionTypeDetection } from './session-type';

/**
 * Pattern Detection Tool
 * Identifies important patterns in conversation history
 * Tracks: wellness decline, repeated topics, progress markers, mood shifts
 */

export const PatternTypeEnum = z.enum([
  'wellness_decline',
  'wellness_improvement',
  'repeated_topic',
  'crisis_escalation',
  'progress_unnoticed',
  'avoidance_pattern',
  'coping_strategy_working',
  'cognitive_distortion',
]);

export const DetectedPatternSchema = z.object({
  type: PatternTypeEnum,
  severity: z.enum(['minor', 'moderate', 'major']),
  evidence: z.string().describe('Specific evidence from conversation'),
  recommendation: z.string().describe('How to address this pattern'),
  confidence: z.number().min(0).max(100),
});

export const PatternDetectionResultSchema = z.object({
  patterns: z.array(DetectedPatternSchema),
  overall_trajectory: z.enum(['improving', 'stable', 'declining']),
  key_themes: z.array(z.string()).max(3),
  alerts: z.array(z.string()).describe('Important alerts for therapist'),
  reasoning: z.string(),
});

export type PatternType = z.infer<typeof PatternTypeEnum>;
export type DetectedPattern = z.infer<typeof DetectedPatternSchema>;
export type PatternDetectionResult = z.infer<typeof PatternDetectionResultSchema>;

/**
 * Detect important patterns in conversation history
 */
export async function detectPatterns(args: {
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  recentWellness: WellnessSignal[];
  currentSessionType: SessionTypeDetection;
}): Promise<PatternDetectionResult> {
  const startTime = Date.now();

  try {
    // Need at least 4 exchanges to detect patterns
    if (args.history.length < 4) {
      return getDefaultPatternDetection();
    }

    // Quick pattern detection first
    const quickPatterns = quickDetectPatterns(args.history, args.recentWellness);

    // If we found strong patterns or have limited history, use quick detection
    if (quickPatterns.patterns.length > 0 || args.history.length < 8) {
      logToolCall({
        tool_name: 'pattern_detection',
        input: { history_length: args.history.length },
        output: {
          patterns_found: quickPatterns.patterns.length,
          trajectory: quickPatterns.overall_trajectory,
        },
        duration_ms: Date.now() - startTime,
        success: true,
      });

      return quickPatterns;
    }

    // For longer histories, use LLM for nuanced pattern analysis
    const historyText = args.history
      .slice(-16) // Last 8 exchanges
      .map((h) => `${h.role.toUpperCase()}: ${h.content.slice(0, 150)}`)
      .join('\n\n');

    const wellnessText =
      args.recentWellness.length > 0
        ? `Recent wellness scores: ${args.recentWellness
            .map((w) => `mood=${w.mood}, energy=${w.energy}, stress=${w.stress}`)
            .join(' → ')}`
        : 'No wellness data available';

    const response = await callOpenRouter({
      system: 'Return valid JSON only. No markdown, no code blocks.',
      user: `Analyze this conversation for important patterns:

${wellnessText}

Recent conversation:
${historyText}

Current session type: ${args.currentSessionType.primary_type}

Identify patterns like:
- Wellness declining or improving over conversation
- Same topic coming up repeatedly without resolution
- Crisis escalation patterns (worsening language/ideas)
- User noticing their own progress (self-awareness)
- Avoidance of certain topics
- What coping strategies seem to help
- Thinking patterns (e.g. catastrophizing, black-and-white thinking)

Return JSON:
{
  "patterns": [
    {"type": "...", "severity": "minor|moderate|major", "evidence": "...", "recommendation": "...", "confidence": 0-100}
  ],
  "overall_trajectory": "improving|stable|declining",
  "key_themes": ["theme1", "theme2"],
  "alerts": ["alert1"],
  "reasoning": "Summary of pattern analysis"
}`,
      temperature: 0.3,
      maxTokens: 500,
    });

    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid JSON response');
    }

    const result = PatternDetectionResultSchema.parse(JSON.parse(jsonMatch[0]));

    // Log successful detection
    const duration = Date.now() - startTime;
    logToolCall({
      tool_name: 'pattern_detection',
      input: { history_length: args.history.length },
      output: {
        patterns_found: result.patterns.length,
        trajectory: result.overall_trajectory,
        alerts: result.alerts.length,
      },
      duration_ms: duration,
      success: true,
    });

    return result;
  } catch (error) {
    const duration = Date.now() - startTime;

    logger.warn(
      {
        type: 'pattern_detection_error',
        error: error instanceof Error ? error.message : String(error),
      },
      'Pattern detection failed'
    );

    logToolCall({
      tool_name: 'pattern_detection',
      input: { history_length: args.history.length },
      output: { error: true },
      duration_ms: duration,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    });

    return getDefaultPatternDetection();
  }
}

/**
 * Quick pattern detection using heuristics
 */
function quickDetectPatterns(
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  recentWellness: WellnessSignal[]
): PatternDetectionResult {
  const patterns: DetectedPattern[] = [];
  const alerts: string[] = [];
  let trajectory: 'improving' | 'stable' | 'declining' = 'stable';

  // Analyze wellness trend
  if (recentWellness.length >= 2) {
    const first = recentWellness[0];
    const last = recentWellness[recentWellness.length - 1];

    const moodChange = last.mood - first.mood;
    const stressChange = last.stress - first.stress;

    if (moodChange < -15 || stressChange > 15) {
      trajectory = 'declining';
      patterns.push({
        type: 'wellness_decline',
        severity: moodChange < -25 ? 'major' : 'moderate',
        evidence: `Mood dropped from ${first.mood} to ${last.mood}, stress rose from ${first.stress} to ${last.stress}`,
        recommendation: 'Check in about what changed. May need more support or resources.',
        confidence: 85,
      });
      alerts.push('Wellness decline detected - may need intervention');
    } else if (moodChange > 15 || stressChange < -15) {
      trajectory = 'improving';
      patterns.push({
        type: 'wellness_improvement',
        severity: 'major',
        evidence: `Mood improved from ${first.mood} to ${last.mood}`,
        recommendation: 'Acknowledge progress and reinforce what is working.',
        confidence: 85,
      });
    }
  }

  // Detect repeated topics
  const userMessages = history
    .filter((h) => h.role === 'user')
    .map((h) => h.content.toLowerCase());

  const keywords = extractKeywords(userMessages);
  const topicCounts = countKeywords(keywords);

  for (const [topic, count] of Object.entries(topicCounts)) {
    if (count >= 3) {
      patterns.push({
        type: 'repeated_topic',
        severity: count >= 4 ? 'major' : 'moderate',
        evidence: `Topic "${topic}" mentioned ${count} times without clear resolution`,
        recommendation: `Explore if there are unmet needs around "${topic}". Consider deeper problem-solving.`,
        confidence: 75,
      });
    }
  }

  // Detect crisis escalation
  const recentUserMessages = userMessages.slice(-3);
  const crisisWords = ['hopeless', 'pointless', 'cannot go on', 'want to die', 'kill myself'];
  const escalationCount = recentUserMessages.filter((msg) =>
    crisisWords.some((w) => msg.includes(w))
  ).length;

  if (escalationCount >= 2) {
    patterns.push({
      type: 'crisis_escalation',
      severity: 'major',
      evidence: 'Crisis-level language appearing repeatedly in recent messages',
      recommendation:
        'URGENT: Assess immediate safety. Offer crisis resources. Consider escalation.',
      confidence: 90,
    });
    alerts.push('CRISIS ESCALATION - Immediate attention required');
  }

  // Detect progress awareness
  const progressWords = ['better', 'improving', 'helped', 'working', 'proud', 'relieved'];
  const progressCount = userMessages.filter((msg) =>
    progressWords.some((w) => msg.includes(w))
  ).length;

  if (progressCount >= 2 && trajectory === 'improving') {
    patterns.push({
      type: 'coping_strategy_working',
      severity: 'major',
      evidence: 'User noting improvements and positive shifts',
      recommendation: 'Reinforce these wins. Help them recognize their own resilience.',
      confidence: 80,
    });
  }

  // Detect avoidance patterns
  if (/\b(but|however|anyway|moving on)\b/i.test(userMessages.join(' '))) {
    const avoidanceCount = userMessages.filter(
      (msg) => /\b(but|however|anyway|moving on)\b/i.test(msg)
    ).length;

    if (avoidanceCount >= 2) {
      patterns.push({
        type: 'avoidance_pattern',
        severity: 'minor',
        evidence: 'Switching topics frequently, deflecting from deeper issues',
        recommendation:
          'Gently notice the pattern. Ask if there are things they are hesitant to explore.',
        confidence: 65,
      });
    }
  }

  // Extract key themes
  const keyThemes = Object.entries(topicCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map((entry) => entry[0]);

  return {
    patterns,
    overall_trajectory: trajectory,
    key_themes: keyThemes,
    alerts,
    reasoning:
      patterns.length > 0
        ? `Detected ${patterns.length} pattern(s) in conversation. Trajectory: ${trajectory}`
        : 'No strong patterns detected yet',
  };
}

/**
 * Extract significant keywords from messages
 */
function extractKeywords(messages: string[]): string[] {
  const stopWords = new Set([
    'the',
    'a',
    'an',
    'and',
    'or',
    'but',
    'in',
    'on',
    'at',
    'to',
    'for',
    'is',
    'are',
    'be',
    'have',
    'has',
    'do',
    'does',
    'did',
    'will',
    'would',
    'could',
    'should',
    'that',
    'this',
    'it',
    'i',
    'you',
    'me',
    'my',
    'your',
  ]);

  const keywords: string[] = [];

  for (const msg of messages) {
    const words = msg
      .split(/\s+/)
      .filter((w) => !stopWords.has(w) && w.length > 4)
      .map((w) => w.replace(/[^\w]/g, '').toLowerCase());

    keywords.push(...words);
  }

  return keywords;
}

/**
 * Count keyword frequencies
 */
function countKeywords(keywords: string[]): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const kw of keywords) {
    counts[kw] = (counts[kw] ?? 0) + 1;
  }

  // Filter out rare words
  return Object.fromEntries(
    Object.entries(counts).filter(([, count]) => count >= 2)
  );
}

/**
 * Get default pattern detection result
 */
export function getDefaultPatternDetection(): PatternDetectionResult {
  return {
    patterns: [],
    overall_trajectory: 'stable',
    key_themes: [],
    alerts: [],
    reasoning: 'Insufficient conversation history to detect patterns',
  };
}
