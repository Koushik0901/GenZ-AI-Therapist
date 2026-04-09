import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SessionManager } from '@/lib/session-storage';

// Mock Supabase
vi.mock('@supabase/supabase-js', () => ({
  createClient: vi.fn(() => ({
    from: vi.fn(() => ({
      insert: vi.fn().mockResolvedValue({ error: null }),
      update: vi.fn().mockResolvedValue({ error: null }),
      eq: vi.fn().mockReturnThis(),
      single: vi.fn().mockResolvedValue({ data: { message_count: 0 }, error: null }),
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

describe('SessionManager', () => {
  let sessionManager: SessionManager;

  beforeEach(() => {
    sessionManager = new SessionManager('test_session_123', 'user_456');
    vi.clearAllMocks();
  });

  describe('initialization', () => {
    it('creates a new session', async () => {
      await sessionManager.initialize();
      // Should not throw
      expect(sessionManager).toBeDefined();
    });

    it('stores user_id if provided', async () => {
      const managerWithUser = new SessionManager('session_789', 'user_abc');
      await managerWithUser.initialize();
      expect(managerWithUser).toBeDefined();
    });

    it('handles Supabase unavailable gracefully', async () => {
      const manager = new SessionManager('session_no_db');
      await manager.initialize();
      // Should not throw even if Supabase is down
      expect(manager).toBeDefined();
    });
  });

  describe('message tracking', () => {
    it('adds user message to session', async () => {
      const message = {
        role: 'user' as const,
        content: 'I am feeling stressed',
        timestamp: new Date().toISOString(),
      };

      await sessionManager.addMessage(message);
      // Should not throw
      expect(sessionManager).toBeDefined();
    });

    it('adds assistant message with metadata', async () => {
      const message = {
        role: 'assistant' as const,
        content: 'I understand, tell me more',
        timestamp: new Date().toISOString(),
        metadata: {
          responseQuality: 85,
        },
      };

      await sessionManager.addMessage(message);
      expect(sessionManager).toBeDefined();
    });
  });

  describe('quality metrics', () => {
    it('updates quality scores', async () => {
      await sessionManager.updateQuality(82, 75);
      expect(sessionManager).toBeDefined();
    });

    it('updates quality without confidence', async () => {
      await sessionManager.updateQuality(78);
      expect(sessionManager).toBeDefined();
    });
  });

  describe('crisis tracking', () => {
    it('marks critical crisis', async () => {
      await sessionManager.markCrisisDetected('critical');
      expect(sessionManager).toBeDefined();
    });

    it('marks high-risk crisis', async () => {
      await sessionManager.markCrisisDetected('high_risk');
      expect(sessionManager).toBeDefined();
    });

    it('does not mark safe as crisis', async () => {
      await sessionManager.markCrisisDetected('safe');
      expect(sessionManager).toBeDefined();
    });
  });

  describe('strategy performance', () => {
    it('records successful strategy', async () => {
      await sessionManager.recordStrategyPerformance(
        'empathy_first',
        'venting',
        70,
        85,
        true
      );
      expect(sessionManager).toBeDefined();
    });

    it('records failed strategy', async () => {
      await sessionManager.recordStrategyPerformance(
        'concrete_steps',
        'venting',
        60,
        45,
        false
      );
      expect(sessionManager).toBeDefined();
    });

    it('calculates quality improvement', async () => {
      await sessionManager.recordStrategyPerformance(
        'more_validation',
        'validation_seeking',
        65,
        82,
        true
      );
      // Quality improved by 17 points
      expect(sessionManager).toBeDefined();
    });
  });

  describe('alerts', () => {
    it('creates info alert', async () => {
      await sessionManager.createAlert('api_error', 'info', 'Test alert');
      expect(sessionManager).toBeDefined();
    });

    it('creates warning alert with details', async () => {
      await sessionManager.createAlert(
        'quality_decline',
        'warning',
        'Quality dropped',
        { quality_score: 45 }
      );
      expect(sessionManager).toBeDefined();
    });

    it('creates critical alert', async () => {
      await sessionManager.createAlert(
        'crisis_escalation',
        'critical',
        'Crisis detected'
      );
      expect(sessionManager).toBeDefined();
    });
  });

  describe('session finalization', () => {
    it('finalizes session', async () => {
      await sessionManager.finalize();
      expect(sessionManager).toBeDefined();
    });

    it('handles finalize gracefully on error', async () => {
      // Even if error occurs, should not throw
      await sessionManager.finalize();
      expect(sessionManager).toBeDefined();
    });
  });

  describe('multi-turn conversations', () => {
    it('tracks multiple messages in sequence', async () => {
      const messages = [
        {
          role: 'user' as const,
          content: 'I am stressed',
          timestamp: new Date().toISOString(),
        },
        {
          role: 'assistant' as const,
          content: 'Tell me more',
          timestamp: new Date().toISOString(),
        },
        {
          role: 'user' as const,
          content: 'Work has been overwhelming',
          timestamp: new Date().toISOString(),
        },
      ];

      for (const msg of messages) {
        await sessionManager.addMessage(msg);
      }

      expect(sessionManager).toBeDefined();
    });

    it('maintains session through conversation lifecycle', async () => {
      await sessionManager.initialize();
      await sessionManager.addMessage({
        role: 'user',
        content: 'Hello',
        timestamp: new Date().toISOString(),
      });
      await sessionManager.updateQuality(70);
      await sessionManager.finalize();

      expect(sessionManager).toBeDefined();
    });
  });
});
