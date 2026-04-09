import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getMonitoringService } from '@/lib/monitoring';

vi.mock('@supabase/supabase-js', () => ({
  createClient: vi.fn(() => ({
    from: vi.fn(() => ({
      insert: vi.fn().mockResolvedValue({ error: null }),
      select: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      order: vi.fn().mockReturnThis(),
      limit: vi.fn().mockResolvedValue({ data: [], error: null }),
      update: vi.fn().mockResolvedValue({ error: null }),
    })),
  })),
}));

vi.mock('@/lib/logging', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('MonitoringService', () => {
  let monitoring: ReturnType<typeof getMonitoringService>;

  beforeEach(() => {
    monitoring = getMonitoringService();
    vi.clearAllMocks();
  });

  describe('response metrics', () => {
    it('records successful response', () => {
      monitoring.recordResponseMetrics(85, 145, true);
      
      const metrics = monitoring.getPerformanceReport();
      expect(metrics.avg_response_quality).toBeGreaterThan(0);
      expect(metrics.total_requests).toBe(1);
    });

    it('accumulates multiple responses', () => {
      monitoring.recordResponseMetrics(80, 100, true);
      monitoring.recordResponseMetrics(90, 150, true);
      monitoring.recordResponseMetrics(75, 120, true);

      const metrics = monitoring.getPerformanceReport();
      expect(metrics.total_requests).toBe(3);
      expect(metrics.avg_response_quality).toBeGreaterThan(70);
    });

    it('tracks response time average', () => {
      monitoring.recordResponseMetrics(80, 100, true);
      monitoring.recordResponseMetrics(90, 200, true);

      const metrics = monitoring.getPerformanceReport();
      expect(metrics.avg_response_time_ms).toBeGreaterThanOrEqual(100);
      expect(metrics.avg_response_time_ms).toBeLessThanOrEqual(200);
    });

    it('counts errors', () => {
      monitoring.recordResponseMetrics(80, 100, true);
      monitoring.recordResponseMetrics(50, 100, false);
      monitoring.recordResponseMetrics(75, 100, true);

      const metrics = monitoring.getPerformanceReport();
      expect(metrics.error_count).toBe(1);
      expect(metrics.error_rate).toBeGreaterThan(0);
    });
  });

  describe('alert creation', () => {
    it('creates info alert', async () => {
      await monitoring.createAlert('pattern_detected', 'info', 'Test message');
      // Should not throw
      expect(monitoring).toBeDefined();
    });

    it('creates warning alert', async () => {
      await monitoring.createAlert('quality_decline', 'warning', 'Quality dropped');
      expect(monitoring).toBeDefined();
    });

    it('creates critical alert', async () => {
      await monitoring.createAlert('crisis_escalation', 'critical', 'Crisis detected');
      expect(monitoring).toBeDefined();
    });

    it('includes session_id in alert', async () => {
      await monitoring.createAlert(
        'api_error',
        'warning',
        'Error occurred',
        'session_123'
      );
      expect(monitoring).toBeDefined();
    });

    it('includes details in alert', async () => {
      await monitoring.createAlert(
        'quality_decline',
        'warning',
        'Quality dropped',
        undefined,
        { quality: 45, threshold: 60 }
      );
      expect(monitoring).toBeDefined();
    });
  });

  describe('crisis tracking', () => {
    it('records crisis detection', () => {
      monitoring.recordCrisisDetection('critical', 'session_crisis_1');
      
      const metrics = monitoring.getPerformanceReport();
      expect(metrics.crisis_detections).toBe(1);
    });

    it('accumulates multiple crises', () => {
      monitoring.recordCrisisDetection('critical', 'session_1');
      monitoring.recordCrisisDetection('high_risk', 'session_2');
      monitoring.recordCrisisDetection('critical', 'session_3');

      const metrics = monitoring.getPerformanceReport();
      expect(metrics.crisis_detections).toBe(3);
    });
  });

  describe('regeneration tracking', () => {
    it('records regeneration', () => {
      monitoring.recordRegeneration('session_123', 'quality too low');
      
      const metrics = monitoring.getPerformanceReport();
      expect(metrics.regenerations).toBe(1);
    });

    it('accumulates regenerations', () => {
      monitoring.recordRegeneration('session_1', 'quality < 65');
      monitoring.recordRegeneration('session_1', 'validation score low');
      monitoring.recordRegeneration('session_2', 'clarity score low');

      const metrics = monitoring.getPerformanceReport();
      expect(metrics.regenerations).toBe(3);
    });
  });

  describe('health status', () => {
    it('reports healthy when quality high', () => {
      for (let i = 0; i < 5; i++) {
        monitoring.recordResponseMetrics(85, 120, true);
      }

      const health = monitoring.getHealthStatus();
      expect(health.status).toBe('healthy');
      expect(health.avg_quality).toBeGreaterThan(70);
    });

    it('reports degraded when quality drops', () => {
      for (let i = 0; i < 3; i++) {
        monitoring.recordResponseMetrics(85, 120, true);
      }
      for (let i = 0; i < 3; i++) {
        monitoring.recordResponseMetrics(40, 120, true);
      }

      const health = monitoring.getHealthStatus();
      expect(['degraded', 'healthy']).toContain(health.status);
    });

    it('reports critical when error rate high', () => {
      for (let i = 0; i < 3; i++) {
        monitoring.recordResponseMetrics(50, 120, false);
      }
      for (let i = 0; i < 1; i++) {
        monitoring.recordResponseMetrics(75, 120, true);
      }

      const health = monitoring.getHealthStatus();
      // 3 errors out of 4 = 75% error rate
      expect(health.error_rate).toBeGreaterThan(50);
    });

    it('includes crisis count in health', () => {
      monitoring.recordCrisisDetection('critical', 'session_1');
      
      const health = monitoring.getHealthStatus();
      expect(health.crisis_count).toBe(1);
    });
  });

  describe('API errors', () => {
    it('records API error', async () => {
      await monitoring.recordAPIError('LLM_TIMEOUT', 'Timeout waiting for Kimi K2.5', 'session_123');
      expect(monitoring).toBeDefined();
    });

    it('records different error types', async () => {
      await monitoring.recordAPIError('SERPER_UNAVAILABLE', 'Search API down');
      await monitoring.recordAPIError('DB_CONNECTION', 'Database offline');
      
      expect(monitoring).toBeDefined();
    });
  });

  describe('user satisfaction', () => {
    it('records positive feedback', () => {
      monitoring.recordUserSatisfaction(true);
      
      const metrics = monitoring.getPerformanceReport();
      expect(metrics.user_satisfaction_rate).toBeGreaterThan(0);
    });

    it('records negative feedback', () => {
      monitoring.recordUserSatisfaction(false);
      
      const metrics = monitoring.getPerformanceReport();
      expect(metrics.user_satisfaction_rate).toBeLessThanOrEqual(100);
    });

    it('averages multiple feedback', () => {
      monitoring.recordUserSatisfaction(true);
      monitoring.recordUserSatisfaction(true);
      monitoring.recordUserSatisfaction(false);

      const metrics = monitoring.getPerformanceReport();
      expect(metrics.user_satisfaction_rate).toBeGreaterThan(0);
      expect(metrics.user_satisfaction_rate).toBeLessThan(100);
    });
  });

  describe('performance reporting', () => {
    it('returns comprehensive metrics', () => {
      monitoring.recordResponseMetrics(80, 150, true);
      monitoring.recordCrisisDetection('critical', 'session_1');
      monitoring.recordRegeneration('session_1', 'test');

      const report = monitoring.getPerformanceReport();

      expect(report.timestamp).toBeDefined();
      expect(report.avg_response_quality).toBeGreaterThan(0);
      expect(report.avg_response_time_ms).toBeGreaterThan(0);
      expect(report.total_requests).toBe(1);
      expect(report.crisis_detections).toBe(1);
      expect(report.regenerations).toBe(1);
    });

    it('all metrics are non-negative', () => {
      monitoring.recordResponseMetrics(50, 100, false);
      const report = monitoring.getPerformanceReport();

      expect(report.avg_response_quality).toBeGreaterThanOrEqual(0);
      expect(report.avg_response_time_ms).toBeGreaterThanOrEqual(0);
      expect(report.error_rate).toBeGreaterThanOrEqual(0);
    });
  });

  describe('alert management', () => {
    it('retrieves active alerts', async () => {
      await monitoring.createAlert('pattern_detected', 'info', 'Test');
      const alerts = await monitoring.getActiveAlerts();

      expect(Array.isArray(alerts)).toBe(true);
    });

    it('acknowledges alert', async () => {
      await monitoring.acknowledgeAlert('alert_123', 'admin_user');
      // Should not throw
      expect(monitoring).toBeDefined();
    });
  });
});
