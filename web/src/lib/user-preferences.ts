import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';

/**
 * User Preference Learning System
 * Adapts strategies and responses based on historical user feedback
 */

export interface UserPreference {
  user_id: string;
  preferred_session_types: string[];
  preferred_strategies: Record<string, number>; // strategy -> effectiveness score
  avg_response_satisfaction: number;
  crisis_support_preference?: string;
  verbosity_preference: 'short' | 'medium' | 'long'; // how much explanation they like
  resource_preference: 'minimal' | 'moderate' | 'comprehensive'; // how many resources they want
  tone_preference: 'clinical' | 'gen_z' | 'balanced'; // Gen Z slang preference
  last_updated: string;
}

export class UserPreferenceLearner {
  private userId: string;
  private supabase = isSupabaseConfigured
    ? createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey)
    : null;

  constructor(userId: string) {
    this.userId = userId;
  }

  /**
   * Get or create user preferences
   */
  async getPreferences(): Promise<UserPreference> {
    if (!this.supabase) {
      return this.getDefaultPreferences();
    }

    try {
      // In a real system, you'd query a user_preferences table
      // For now, return defaults
      return this.getDefaultPreferences();
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error fetching preferences'
      );
      return this.getDefaultPreferences();
    }
  }

  /**
   * Record feedback and update preferences
   */
  async recordFeedback(
    sessionId: string,
    sentiment: 'positive' | 'negative',
    strategy: string,
    quality: number,
    comment?: string
  ): Promise<void> {
    try {
      const prefs = await this.getPreferences();

      // Update strategy effectiveness
      const currentScore = prefs.preferred_strategies[strategy] ?? 50;
      const adjustment = sentiment === 'positive' ? 5 : -10;
      const newScore = Math.max(0, Math.min(100, currentScore + adjustment));

      prefs.preferred_strategies[strategy] = newScore;

      // Update satisfaction score
      const satisfactionAdjustment = sentiment === 'positive' ? quality / 100 : -(100 - quality) / 100;
      prefs.avg_response_satisfaction = Math.max(
        0,
        Math.min(
          100,
          prefs.avg_response_satisfaction + satisfactionAdjustment * 5
        )
      );

      prefs.last_updated = new Date().toISOString();

      // Infer preferences from comment
      if (comment) {
        this.inferPreferencesFromComment(comment, prefs);
      }

      // Save to Supabase (if table exists)
      if (this.supabase) {
        await this.savePreferences(prefs);
      }

      logger.debug(
        {
          user_id: this.userId,
          strategy,
          new_score: newScore,
          satisfaction: prefs.avg_response_satisfaction,
        },
        'User preferences updated'
      );
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error recording feedback'
      );
    }
  }

  /**
   * Get best strategy for session type
   */
  async getBestStrategy(sessionType: string): Promise<string> {
    const prefs = await this.getPreferences();
    const strategyScores = Object.entries(prefs.preferred_strategies);

    // Filter for relevant session type
    let bestStrategy = 'empathy_first';
    let bestScore = 0;

    for (const [strategy, score] of strategyScores) {
      if (score > bestScore && this.isStrategyRelevantForSessionType(strategy, sessionType)) {
        bestScore = score;
        bestStrategy = strategy;
      }
    }

    return bestStrategy;
  }

  /**
   * Infer preferences from user comment
   */
  private inferPreferencesFromComment(comment: string, prefs: UserPreference): void {
    const lower = comment.toLowerCase();

    // Verbosity detection
    if (/\b(too long|tl;dr|too much|wall of text)\b/i.test(lower)) {
      prefs.verbosity_preference = 'short';
    } else if (/\b(more detail|explain more|deep dive|comprehensive)\b/i.test(lower)) {
      prefs.verbosity_preference = 'long';
    }

    // Resource preference
    if (/\b(too many resources|overwhelming|simplify)\b/i.test(lower)) {
      prefs.resource_preference = 'minimal';
    } else if (/\b(more resources|links|references)\b/i.test(lower)) {
      prefs.resource_preference = 'comprehensive';
    }

    // Tone preference
    if (/\b(too casual|not serious|gen.*z|slang)\b/i.test(lower)) {
      prefs.tone_preference = 'gen_z';
    } else if (/\b(professional|clinical|formal)\b/i.test(lower)) {
      prefs.tone_preference = 'clinical';
    }

    logger.debug(
      { inferred_prefs: { verbosity: prefs.verbosity_preference, resources: prefs.resource_preference } },
      'Preferences inferred from comment'
    );
  }

  /**
   * Check if strategy is relevant for session type
   */
  private isStrategyRelevantForSessionType(strategy: string, sessionType: string): boolean {
    const relevantStrategies: Record<string, string[]> = {
      venting: ['empathy_first', 'more_validation'],
      problem_solving: ['concrete_steps', 'reframe_positive'],
      validation_seeking: ['more_validation', 'empathy_first'],
      information_seeking: ['resources_focus', 'clarity'],
      crisis: ['empathy_first', 'resources_focus'],
    };

    const relevant = relevantStrategies[sessionType] || ['empathy_first'];
    return relevant.includes(strategy);
  }

  /**
   * Save preferences to Supabase
   */
  private async savePreferences(prefs: UserPreference): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      // You'd create a user_preferences table for this
      // For now, just log that we would save
      logger.debug(
        { user_id: this.userId, satisfaction: prefs.avg_response_satisfaction },
        'Preferences would be saved to Supabase'
      );
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error saving preferences'
      );
    }
  }

  /**
   * Get default preferences
   */
  private getDefaultPreferences(): UserPreference {
    return {
      user_id: this.userId,
      preferred_session_types: [],
      preferred_strategies: {
        empathy_first: 50,
        more_validation: 50,
        more_warmth: 50,
        concrete_steps: 50,
        resources_focus: 50,
        more_clarity: 50,
        shorter_response: 50,
        reframe_positive: 50,
      },
      avg_response_satisfaction: 70,
      verbosity_preference: 'medium',
      resource_preference: 'moderate',
      tone_preference: 'gen_z',
      last_updated: new Date().toISOString(),
    };
  }
}

/**
 * Get user preference learner instance
 */
export function getUserPreferenceLearner(userId: string): UserPreferenceLearner {
  return new UserPreferenceLearner(userId);
}
