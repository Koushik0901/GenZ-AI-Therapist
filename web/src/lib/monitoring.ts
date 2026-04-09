import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';

/**
 * Real-Time Monitoring & Alerts System
 * Tracks system health, performance, and critical patterns
 */

export type AlertType =
  | 'crisis_escalation'
  | 'quality_decline'
  | 'api_error'
  | 'pattern_detected'
  | 'user_support_needed';

export type AlertSeverity = 'info' | 'warning' | 'critical';

export interface SystemAlert {
  id?: string;
  alert_type: AlertType;
  severity: AlertSeverity;
  session_id?: string;
  message: string;
  details?: Record<string, unknown>;
  created_at: string;
  acknowledged: boolean;
}

export interface SystemMetrics {
  timestamp: string;
  avg_response_quality: number;
  avg_response_time_ms: number;
  total_requests: number;
  error_count: number;
  error_rate: number; // percentage
  crisis_detections: number;
  regenerations: number;
  user_satisfaction_rate: number;
}

export class MonitoringService {
  private supabase = isSupabaseConfigured
    ? createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey)
    : null;

  private metrics: SystemMetrics = {
    timestamp: new Date().toISOString(),
    avg_response_quality: 0,
    avg_response_time_ms: 0,
    total_requests: 0,
    error_count: 0,
    error_rate: 0,
    crisis_detections: 0,
    regenerations: 0,
    user_satisfaction_rate: 0,
  };

  /**
   * Create alert
   */
  async createAlert(
    type: AlertType,
    severity: AlertSeverity,
    message: string,
    sessionId?: string,
    details?: Record<string, unknown>
  ): Promise<void> {
    const alert: SystemAlert = {
      alert_type: type,
      severity,
      session_id: sessionId,
      message,
      details,
      created_at: new Date().toISOString(),
      acknowledged: false,
    };

    // Log locally
    if (severity === 'critical') {
      logger.error(alert, `ALERT [${type}]: ${message}`);
    } else if (severity === 'warning') {
      logger.warn(alert, `ALERT [${type}]: ${message}`);
    } else {
      logger.info(alert, `ALERT [${type}]: ${message}`);
    }

    // Store in Supabase
    if (this.supabase) {
      try {
        const { error } = await this.supabase.from('system_alerts').insert([alert]);

        if (error) {
          logger.warn({ error: error.message }, 'Failed to store alert');
        }
      } catch (err) {
        logger.error(
          { error: err instanceof Error ? err.message : String(err) },
          'Error creating alert'
        );
      }
    }
  }

  /**
   * Record response metrics
   */
  recordResponseMetrics(
    quality: number,
    responseTimeMs: number,
    success: boolean
  ): void {
    this.metrics.total_requests++;
    this.metrics.avg_response_quality =
      (this.metrics.avg_response_quality * (this.metrics.total_requests - 1) + quality) /
      this.metrics.total_requests;
    this.metrics.avg_response_time_ms =
      (this.metrics.avg_response_time_ms * (this.metrics.total_requests - 1) + responseTimeMs) /
      this.metrics.total_requests;

    if (!success) {
      this.metrics.error_count++;
    }

    this.metrics.error_rate =
      (this.metrics.error_count / this.metrics.total_requests) * 100;

    // Log metrics every 10 requests
    if (this.metrics.total_requests % 10 === 0) {
      logger.debug(
        {
          avg_quality: Math.round(this.metrics.avg_response_quality),
          avg_time_ms: Math.round(this.metrics.avg_response_time_ms),
          error_rate: this.metrics.error_rate.toFixed(2) + '%',
        },
        'System metrics'
      );
    }

    // Alert if quality drops below 60
    if (quality < 60) {
      this.createAlert(
        'quality_decline',
        'warning',
        `Response quality dropped to ${quality}`,
        undefined,
        { response_quality: quality }
      );
    }
  }

  /**
   * Record crisis detection
   */
  recordCrisisDetection(severity: string, sessionId: string): void {
    this.metrics.crisis_detections++;

    const alertSeverity = severity === 'critical' ? 'critical' : severity === 'high_risk' ? 'warning' : 'info';

    this.createAlert(
      'crisis_escalation',
      alertSeverity,
      `Crisis detected: ${severity}`,
      sessionId,
      { severity }
    );
  }

  /**
   * Record regeneration
   */
  recordRegeneration(sessionId: string, reason: string): void {
    this.metrics.regenerations++;

    this.createAlert(
      'pattern_detected',
      'info',
      `Response regeneration: ${reason}`,
      sessionId,
      { reason }
    );
  }

  /**
   * Record API error
   */
  recordAPIError(errorType: string, errorMessage: string, sessionId?: string): void {
    this.createAlert(
      'api_error',
      'warning',
      `API Error [${errorType}]: ${errorMessage}`,
      sessionId,
      { error_type: errorType, error_message: errorMessage }
    );
  }

  /**
   * Get active alerts
   */
  async getActiveAlerts(): Promise<SystemAlert[]> {
    if (!this.supabase) {
      return [];
    }

    try {
      const { data, error } = await this.supabase
        .from('system_alerts')
        .select('*')
        .eq('acknowledged', false)
        .order('created_at', { ascending: false })
        .limit(50);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to fetch alerts');
        return [];
      }

      return data || [];
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error fetching alerts'
      );
      return [];
    }
  }

  /**
   * Acknowledge alert
   */
  async acknowledgeAlert(alertId: string, acknowledgedBy?: string): Promise<void> {
    if (!this.supabase) {
      return;
    }

    try {
      const { error } = await this.supabase
        .from('system_alerts')
        .update({
          acknowledged: true,
          acknowledged_at: new Date().toISOString(),
          acknowledged_by: acknowledgedBy || null,
        })
        .eq('id', alertId);

      if (error) {
        logger.warn({ error: error.message }, 'Failed to acknowledge alert');
      } else {
        logger.debug({ alert_id: alertId }, 'Alert acknowledged');
      }
    } catch (err) {
      logger.error(
        { error: err instanceof Error ? err.message : String(err) },
        'Error acknowledging alert'
      );
    }
  }

  /**
   * Get system health status
   */
  getHealthStatus(): {
    status: 'healthy' | 'degraded' | 'critical';
    avg_quality: number;
    error_rate: number;
    crisis_count: number;
  } {
    let status: 'healthy' | 'degraded' | 'critical' = 'healthy';

    if (this.metrics.error_rate > 10 || this.metrics.avg_response_quality < 60) {
      status = 'degraded';
    }

    if (this.metrics.error_rate > 20 || this.metrics.avg_response_quality < 50) {
      status = 'critical';
    }

    return {
      status,
      avg_quality: Math.round(this.metrics.avg_response_quality),
      error_rate: Math.round(this.metrics.error_rate),
      crisis_count: this.metrics.crisis_detections,
    };
  }

  /**
   * Get performance report
   */
  getPerformanceReport(): SystemMetrics {
    return { ...this.metrics };
  }

  /**
   * Record user satisfaction
   */
  recordUserSatisfaction(positive: boolean): void {
    const totalFeedback = this.metrics.total_requests; // Simplified
    const newRate = positive ? 100 : 0;

    this.metrics.user_satisfaction_rate =
      (this.metrics.user_satisfaction_rate * (totalFeedback - 1) + newRate) / totalFeedback;

    if (this.metrics.user_satisfaction_rate < 50) {
      this.createAlert(
        'user_support_needed',
        'warning',
        `User satisfaction rate dropping: ${this.metrics.user_satisfaction_rate.toFixed(1)}%`
      );
    }
  }
}

/**
 * Global monitoring service instance
 */
let monitoringService: MonitoringService | null = null;

export function getMonitoringService(): MonitoringService {
  if (!monitoringService) {
    monitoringService = new MonitoringService();
  }
  return monitoringService;
}
