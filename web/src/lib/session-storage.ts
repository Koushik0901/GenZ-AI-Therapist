import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';
import type { Classification } from './tools/classification';
import type { WellnessSignal } from './tools/wellness';
import type { CrisisEvaluation } from './tools/crisis-eval';
import type { SessionTypeDetection } from './tools/session-type';

/**
 * Session Storage System
 * Manages multi-turn conversation state, metadata, and user preferences
 */

export interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: {
    classification?: Classification;
    wellness?: WellnessSignal;
    crisis?: CrisisEvaluation;
    sessionType?: SessionTypeDetection;
    responseQuality?: number;
  };
}

export interface SessionState {
  session_id: string;
  user_id?: string;
  messages: SessionMessage[];
  start_time: string;
  last_activity: string;
  session_type?: string;
  avg_quality?: number;
  crisis_detected: boolean;
  total_messages: number;
  metadata?: {
    avg_classification_confidence?: number;
    crisis_escalations?: number;
    regenerations?: number;
  };
}

/**
 * Session Manager
 */
export class SessionManager {
  private sessionId: string;
  private userId?: string;
  private supabase = isSupabaseConfigured
    ? createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey)
    : null;

  constructor(sessionId: string, userId?: string) {
    this.sessionId = sessionId;
    this.userId = userId;
  }

  /**
   * Initialize a new session
   */
  async initialize(): Promise<void> {
    if (!this.supabase) {
      logger.debug({}, 'Supabase not configured, using in-memory session storage');
      return;
    }

    try {
      const { error } = await this.supabase.from('session_metadata').insert([
        {
          session_id: this.sessionId,
          user_id: this.userId || null,
          start_time: new Date().toISOString(),
          message_count: 0,
          crisis_detected: false,
        },
      ]);

      if (error) {
        logger.warn(
          { error: error.message },
          'Failed to initialize session in Supabase'
        );
      } else {
        logger.debug({ session_id: this.sessionId }, 'Session initialized');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error initializing session'
      );
    }
  }

  /**
   * Add message to session
   */
  async addMessage(message: SessionMessage): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      // Store message in Supabase (you'd create a messages table for this)
      // For now, just update session metadata
      const { error } = await this.supabase
        .from('session_metadata')
        .update({
          message_count: (await this.getMessageCount()) + 1,
          last_activity: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('session_id', this.sessionId);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to update session');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error adding message to session'
      );
    }
  }

  /**
   * Get message count for session
   */
  private async getMessageCount(): Promise<number> {
    if (!this.supabase) {
      return 0;
    }

    try {
      const { data, error } = await this.supabase
        .from('session_metadata')
        .select('message_count')
        .eq('session_id', this.sessionId)
        .single();

      if (error) return 0;
      return data?.message_count || 0;
    } catch {
      return 0;
    }
  }

  /**
   * Update session quality metrics
   */
  async updateQuality(quality: number, confidence?: number): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      const { error } = await this.supabase
        .from('session_metadata')
        .update({
          avg_response_quality: quality,
          avg_classification_confidence: confidence,
          updated_at: new Date().toISOString(),
        })
        .eq('session_id', this.sessionId);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to update quality metrics');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error updating quality metrics'
      );
    }
  }

  /**
   * Mark crisis detected in session
   */
  async markCrisisDetected(severity: string): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      const { error } = await this.supabase
        .from('session_metadata')
        .update({
          crisis_detected: severity === 'critical' || severity === 'high_risk',
          updated_at: new Date().toISOString(),
        })
        .eq('session_id', this.sessionId);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to mark crisis');
      }

      // Also create an alert
      if (severity === 'critical') {
        await this.createAlert('crisis_escalation', 'critical', `Crisis detected: ${severity}`);
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error marking crisis'
      );
    }
  }

  /**
   * Record strategy performance
   */
  async recordStrategyPerformance(
    strategyName: string,
    sessionType: string,
    qualityBefore: number,
    qualityAfter: number,
    success: boolean
  ): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      // Upsert strategy performance record
      const { error } = await this.supabase.from('strategy_performance').upsert(
        [
          {
            strategy_name: strategyName,
            session_type: sessionType,
            quality_before: qualityBefore,
            quality_after: qualityAfter,
            success_count: success ? 1 : 0,
            used_count: 1,
            avg_quality_improvement: qualityAfter - qualityBefore,
          },
        ],
        {
          onConflict: 'strategy_name,session_type,attempt_number',
        }
      );

      if (error) {
        logger.warn({ error: error.message }, 'Failed to record strategy performance');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error recording strategy performance'
      );
    }
  }

  /**
   * Create system alert
   */
  async createAlert(
    type: string,
    severity: string,
    message: string,
    details?: Record<string, unknown>
  ): Promise<void> {
    if (!this.supabase) {
      logger.warn(
        { type, severity, message },
        'Alert (Supabase not configured)'
      );
      return;
    }

    try {
      const { error } = await this.supabase.from('system_alerts').insert([
        {
          alert_type: type,
          severity,
          session_id: this.sessionId,
          message,
          details,
          created_at: new Date().toISOString(),
        },
      ]);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to create alert');
      } else {
        logger.info({ type, severity, message }, 'System alert created');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error creating alert'
      );
    }
  }

  /**
   * Finalize session
   */
  async finalize(): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      const { error } = await this.supabase
        .from('session_metadata')
        .update({
          end_time: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('session_id', this.sessionId);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to finalize session');
      } else {
        logger.debug({ session_id: this.sessionId }, 'Session finalized');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error finalizing session'
      );
    }
  }
}

/**
 * Helper to create a new session
 */
export function createSession(userId?: string): SessionManager {
  const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const manager = new SessionManager(sessionId, userId);
  manager.initialize();
  return manager;
}
