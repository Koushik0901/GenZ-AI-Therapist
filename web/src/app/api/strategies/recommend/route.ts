import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { getUserPreferenceLearner } from '@/lib/user-preferences';
import { logger } from '@/lib/logging';

/**
 * Strategy Recommendation API Endpoint
 * GET /api/strategies/recommend?session_type=venting&user_id=user_123
 * Returns ranked strategies based on user preferences and session type
 */

const QuerySchema = z.object({
  session_type: z.enum([
    'venting',
    'problem_solving',
    'validation_seeking',
    'information_seeking',
    'crisis',
    'chitchat',
  ]),
  user_id: z.string().optional(),
});

type SessionType = z.infer<typeof QuerySchema>['session_type'];

const STRATEGY_SESSION_MAPPING: Record<SessionType, string[]> = {
  venting: ['empathy_first', 'more_validation', 'more_warmth'],
  problem_solving: ['concrete_steps', 'reframe_positive', 'empathy_first'],
  validation_seeking: ['more_validation', 'empathy_first', 'more_warmth'],
  information_seeking: ['resources_focus', 'more_clarity', 'concrete_steps'],
  crisis: ['empathy_first', 'resources_focus', 'more_warmth'],
  chitchat: ['empathy_first', 'more_warmth', 'reframe_positive'],
};

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const sessionType = searchParams.get('session_type');
    const userId = searchParams.get('user_id');

    // Validate query parameters
    const parsed = QuerySchema.parse({
      session_type: sessionType,
      user_id: userId || undefined,
    });

    // Get relevant strategies for this session type
    const relevantStrategies = STRATEGY_SESSION_MAPPING[parsed.session_type];

    // Get user preferences if user_id provided
    let rankedStrategies = relevantStrategies;

    if (parsed.user_id) {
      try {
        const preferenceLearner = getUserPreferenceLearner(parsed.user_id);
        const preferences = await preferenceLearner.getPreferences();

        // Rank strategies by user preference scores
        rankedStrategies = relevantStrategies.sort((a, b) => {
          const scoreA = preferences.preferred_strategies[a] || 50;
          const scoreB = preferences.preferred_strategies[b] || 50;
          return scoreB - scoreA;
        });

        logger.debug(
          {
            user_id: parsed.user_id,
            session_type: parsed.session_type,
            ranked_strategies: rankedStrategies,
          },
          'Strategies ranked by user preferences'
        );
      } catch (error) {
        logger.warn(
          {
            error: error instanceof Error ? error.message : String(error),
            user_id: parsed.user_id,
          },
          'Failed to get user preferences, using default ranking'
        );
      }
    }

    // Get preference scores for response
    let preferencesData = null;
    if (parsed.user_id) {
      try {
        const preferenceLearner = getUserPreferenceLearner(parsed.user_id);
        const preferences = await preferenceLearner.getPreferences();

        preferencesData = {
          verbosity_preference: preferences.verbosity_preference,
          resource_preference: preferences.resource_preference,
          tone_preference: preferences.tone_preference,
          satisfaction_score: Math.round(preferences.avg_response_satisfaction),
        };
      } catch (error) {
        logger.debug({}, 'Could not retrieve user preferences');
      }
    }

    return NextResponse.json(
      {
        success: true,
        session_type: parsed.session_type,
        user_id: parsed.user_id || null,
        primary_strategy: rankedStrategies[0],
        ranked_strategies: rankedStrategies,
        strategy_details: {
          [rankedStrategies[0]]: {
            name: rankedStrategies[0],
            description: getStrategyDescription(rankedStrategies[0]),
            rank: 1,
          },
          ...(rankedStrategies[1] && {
            [rankedStrategies[1]]: {
              name: rankedStrategies[1],
              description: getStrategyDescription(rankedStrategies[1]),
              rank: 2,
            },
          }),
          ...(rankedStrategies[2] && {
            [rankedStrategies[2]]: {
              name: rankedStrategies[2],
              description: getStrategyDescription(rankedStrategies[2]),
              rank: 3,
            },
          }),
        },
        user_preferences: preferencesData,
      },
      { status: 200 }
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      logger.warn(
        { errors: error.errors },
        'Invalid strategy recommendation query'
      );

      return NextResponse.json(
        {
          success: false,
          message: 'Invalid query parameters',
          errors: error.errors,
        },
        { status: 400 }
      );
    }

    logger.error(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      'Strategy recommendation API error'
    );

    return NextResponse.json(
      {
        success: false,
        message: 'Internal server error',
      },
      { status: 500 }
    );
  }
}

/**
 * Get human-readable description of strategy
 */
function getStrategyDescription(strategy: string): string {
  const descriptions: Record<string, string> = {
    empathy_first: 'Lead with empathy and understanding before offering solutions',
    more_validation: 'Focus on affirming and validating the user\'s feelings',
    more_warmth: 'Prioritize sounding genuine, human, and emotionally present',
    more_clarity: 'Keep response concise and easy to understand',
    shorter_response: 'Provide brief, punchy responses without lengthy explanations',
    concrete_steps: 'Offer specific, actionable steps or ideas to try',
    resources_focus: 'Prioritize providing helpful resources and information',
    reframe_positive: 'Help find a more hopeful or constructive angle',
  };

  return descriptions[strategy] || 'Response strategy';
}
