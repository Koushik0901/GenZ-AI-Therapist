import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';

/**
 * A/B Testing Framework
 * Compare different strategies and configurations to optimize responses
 */

export interface ABTestVariant {
  id: string;
  name: string;
  type: 'strategy_selection' | 'resource_search' | 'response_eval';
  config: Record<string, unknown>;
  active: boolean;
  created_at: string;
}

export interface ABTestResult {
  test_id: string;
  variant_id: string;
  session_id: string;
  response_quality: number;
  user_feedback: 'positive' | 'negative' | null;
  metrics: {
    response_time_ms?: number;
    regeneration_attempts?: number;
    resource_count?: number;
    clarity_score?: number;
    validation_score?: number;
    warmth_score?: number;
  };
}

export class ABTestManager {
  private supabase = isSupabaseConfigured
    ? createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey)
    : null;

  /**
   * Get active test variants
   */
  async getActiveVariants(type?: string): Promise<ABTestVariant[]> {
    if (!this.supabase) {
      return [];
    }

    try {
      let query = this.supabase.from('ab_test_variants').select('*').eq('active', true);

      if (type) {
        query = query.eq('variant_type', type);
      }

      const { data, error } = await query;

      if (error) {
        logger.warn({ error: error.message }, 'Failed to fetch AB test variants');
        return [];
      }

      return data || [];
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error fetching AB test variants'
      );
      return [];
    }
  }

  /**
   * Select variant for user (round-robin or random)
   */
  async selectVariantForUser(sessionId: string, type: string): Promise<ABTestVariant | null> {
    const variants = await this.getActiveVariants(type);

    if (variants.length === 0) {
      return null;
    }

    // Simple round-robin: assign variant based on session hash
    const variantIndex = this.hashSessionId(sessionId) % variants.length;
    return variants[variantIndex];
  }

  /**
   * Record A/B test result
   */
  async recordResult(result: ABTestResult): Promise<void> {
    if (!this.supabase) {
      logger.debug({ test_id: result.test_id }, 'A/B test result (not persisted)');
      return;
    }

    try {
      const { error } = await this.supabase.from('ab_test_results').insert([
        {
          test_id: result.test_id,
          variant_id: result.variant_id,
          session_id: result.session_id,
          response_quality: result.response_quality,
          user_feedback: result.user_feedback,
          metrics: result.metrics,
          created_at: new Date().toISOString(),
        },
      ]);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to record A/B test result');
      } else {
        logger.debug(
          { test_id: result.test_id, variant_id: result.variant_id },
          'A/B test result recorded'
        );
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error recording A/B test result'
      );
    }
  }

  /**
   * Get results for variant (aggregate metrics)
   */
  async getVariantResults(variantId: string): Promise<{
    test_count: number;
    avg_quality: number;
    positive_feedback_rate: number;
    avg_response_time: number;
  }> {
    if (!this.supabase) {
      return {
        test_count: 0,
        avg_quality: 0,
        positive_feedback_rate: 0,
        avg_response_time: 0,
      };
    }

    try {
      const { data, error } = await this.supabase
        .from('ab_test_results')
        .select('response_quality, user_feedback, metrics')
        .eq('variant_id', variantId);

      if (error || !data) {
        return {
          test_count: 0,
          avg_quality: 0,
          positive_feedback_rate: 0,
          avg_response_time: 0,
        };
      }

      const testCount = data.length;
      const avgQuality = data.reduce((sum, r) => sum + (r.response_quality || 0), 0) / testCount || 0;
      const positiveFeedback = data.filter((r) => r.user_feedback === 'positive').length;
      const avgResponseTime =
        data.reduce((sum, r) => sum + (r.metrics?.response_time_ms || 0), 0) / testCount || 0;

      return {
        test_count: testCount,
        avg_quality: Math.round(avgQuality),
        positive_feedback_rate: (positiveFeedback / testCount) * 100,
        avg_response_time: Math.round(avgResponseTime),
      };
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? error.message : String(err) },
        'Error fetching variant results'
      );
      return {
        test_count: 0,
        avg_quality: 0,
        positive_feedback_rate: 0,
        avg_response_time: 0,
      };
    }
  }

  /**
   * Compare variants
   */
  async compareVariants(variantIds: string[]): Promise<
    Record<string, { test_count: number; avg_quality: number; positive_feedback_rate: number }>
  > {
    const results: Record<string, any> = {};

    for (const variantId of variantIds) {
      results[variantId] = await this.getVariantResults(variantId);
    }

    return results;
  }

  /**
   * Simple hash function for consistent variant assignment
   */
  private hashSessionId(sessionId: string): number {
    let hash = 0;
    for (let i = 0; i < sessionId.length; i++) {
      const char = sessionId.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Create new test variant
   */
  async createVariant(
    name: string,
    type: 'strategy_selection' | 'resource_search' | 'response_eval',
    config: Record<string, unknown>,
    description?: string
  ): Promise<ABTestVariant | null> {
    if (!this.supabase) {
      return null;
    }

    try {
      const { data, error } = await this.supabase
        .from('ab_test_variants')
        .insert([
          {
            variant_name: name,
            variant_type: type,
            description: description || null,
            config,
            active: true,
            created_at: new Date().toISOString(),
          },
        ])
        .select()
        .single();

      if (error) {
        logger.warn({ error: error.message }, 'Failed to create variant');
        return null;
      }

      logger.info(
        { variant_name: name, variant_type: type },
        'A/B test variant created'
      );

      return {
        id: data.id,
        name: data.variant_name,
        type: data.variant_type as any,
        config: data.config,
        active: data.active,
        created_at: data.created_at,
      };
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error creating variant'
      );
      return null;
    }
  }

  /**
   * Deactivate variant
   */
  async deactivateVariant(variantId: string): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      const { error } = await this.supabase
        .from('ab_test_variants')
        .update({ active: false })
        .eq('id', variantId);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to deactivate variant');
      } else {
        logger.info({ variant_id: variantId }, 'Variant deactivated');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error deactivating variant'
      );
    }
  }
}

/**
 * Get AB test manager instance
 */
export function getABTestManager(): ABTestManager {
  return new ABTestManager();
}
