import { describe, it, expect, vi, beforeEach } from 'vitest';
import { evaluateCrisis, getDefaultCrisisEval } from '@/lib/tools/crisis-eval';
import * as openrouter from '@/lib/openrouter';
import type { Classification } from '@/lib/tools/classification';
import type { WellnessSignal } from '@/lib/tools/wellness';

// Mock OpenRouter
vi.mock('@/lib/openrouter');
vi.mock('@/lib/logging', () => ({
  logger: { warn: vi.fn() },
  logToolCall: vi.fn(),
}));

const mockClassification: Classification = {
  sentiment: 'Negative',
  intent: 'support',
  confidence: 75,
  reasoning: 'Test',
  alternativeInterpretations: [],
};

const mockWellness: WellnessSignal = {
  mood: 50,
  energy: 50,
  stress: 50,
  confidence: 75,
  reasoning: 'Test',
};

describe('Crisis Evaluation Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('evaluateCrisis', () => {
    it('detects explicit suicide keywords as critical', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I want to kill myself',
        history: [],
        classification: mockClassification,
        wellness: mockWellness,
      });

      expect(result.severity).toBe('critical');
      expect(result.score).toBeGreaterThan(90);
      expect(result.explicit_keywords).toBe(true);
      expect(result.confidence).toBeGreaterThan(90);
    });

    it('detects self-harm keywords as critical', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I am going to hurt myself',
        history: [],
        classification: mockClassification,
        wellness: mockWellness,
      });

      expect(result.severity).toBe('critical');
      expect(result.explicit_keywords).toBe(true);
    });

    it('detects "want to die" phrases as critical', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I just want to die, I cannot do this',
        history: [],
        classification: mockClassification,
        wellness: mockWellness,
      });

      expect(result.severity).toBe('critical');
      expect(result.explicit_keywords).toBe(true);
    });

    it('detects implicit hopelessness without explicit keywords', async () => {
      const result = await evaluateCrisis({
        userMessage: 'Everything is pointless, nothing matters anymore',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Crisis',
        },
        wellness: {
          ...mockWellness,
          mood: 10,
          energy: 15,
          stress: 95,
        },
      });

      expect(result.implicit_hopelessness).toBe(true);
      expect(result.severity).toBe('high_risk');
      expect(result.score).toBeGreaterThan(70);
    });

    it('identifies crisis from multiple signal combination', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I cannot take it anymore',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Crisis',
        },
        wellness: {
          ...mockWellness,
          mood: 8,
          energy: 10,
          stress: 96,
        },
      });

      expect(result.severity).toBe('high_risk');
      expect(result.wellness_signal).toBe(true);
    });

    it('marks safe when no crisis indicators present', async () => {
      const result = await evaluateCrisis({
        userMessage: 'Just wanted to say hi',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Positive',
        },
        wellness: {
          ...mockWellness,
          mood: 75,
          energy: 70,
          stress: 30,
        },
      });

      expect(result.severity).toBe('safe');
      expect(result.score).toBeLessThan(20);
      expect(result.explicit_keywords).toBe(false);
      expect(result.implicit_hopelessness).toBe(false);
    });

    it('marks at_risk for negative sentiment without explicit crisis', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I feel really sad and hopeless today',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Negative',
        },
        wellness: {
          ...mockWellness,
          mood: 25,
          energy: 30,
          stress: 80,
        },
      });

      expect(result.severity).toBe('at_risk');
      expect(result.score).toBeGreaterThan(30);
      expect(result.score).toBeLessThan(70);
    });

    it('uses LLM evaluation for ambiguous cases', async () => {
      const mockResponse = JSON.stringify({
        severity: 'high_risk',
        score: 72,
        confidence: 81,
        reasoning: 'LLM assessment of ambiguous message',
        recommended_actions: [
          'Encourage crisis hotline contact',
          'Assess immediate safety',
        ],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateCrisis({
        userMessage: 'Sometimes I wonder if it would be better if I was not here',
        history: [],
        classification: mockClassification,
        wellness: {
          ...mockWellness,
          mood: 28,
          stress: 77,
        },
      });

      expect(result.severity).toBe('high_risk');
      expect(result.score).toBeGreaterThan(70);
    });

    it('detects conversation escalation pattern', async () => {
      const history = [
        { role: 'user' as const, content: 'I am stressed' },
        { role: 'assistant' as const, content: 'I hear you' },
        { role: 'user' as const, content: 'Things feel hopeless' },
        { role: 'assistant' as const, content: 'Tell me more' },
        {
          role: 'user' as const,
          content: 'Everything is pointless and hopeless',
        },
      ];

      const result = await evaluateCrisis({
        userMessage: 'I cannot take this anymore',
        history,
        classification: mockClassification,
        wellness: {
          ...mockWellness,
          mood: 15,
          stress: 90,
        },
      });

      expect(result.pattern_escalation).toBe(true);
    });

    it('uses fallback on LLM failure', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await evaluateCrisis({
        userMessage: 'I am not sure I can continue',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Negative',
        },
        wellness: {
          ...mockWellness,
          mood: 20,
          stress: 85,
        },
      });

      // Fallback should at least identify as at_risk
      expect(result.severity).not.toBe('safe');
      expect(result.confidence).toBeGreaterThan(50);
    });

    it('includes recommended actions for crisis', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I want to end my life',
        history: [],
        classification: mockClassification,
        wellness: mockWellness,
      });

      expect(result.recommended_actions.length).toBeGreaterThan(0);
      expect(
        result.recommended_actions.some((action) =>
          action.toLowerCase().includes('crisis')
        )
      ).toBe(true);
    });

    it('no recommended actions for safe assessment', async () => {
      const result = await evaluateCrisis({
        userMessage: 'Having a good day today',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Positive',
        },
        wellness: {
          ...mockWellness,
          mood: 80,
          energy: 75,
          stress: 20,
        },
      });

      expect(result.recommended_actions.length).toBe(0);
    });

    it('considers case insensitivity for keywords', async () => {
      const result = await evaluateCrisis({
        userMessage: 'I WANT TO KILL MYSELF',
        history: [],
        classification: mockClassification,
        wellness: mockWellness,
      });

      expect(result.explicit_keywords).toBe(true);
      expect(result.severity).toBe('critical');
    });
  });

  describe('getDefaultCrisisEval', () => {
    it('returns safe default evaluation', () => {
      const result = getDefaultCrisisEval();

      expect(result.severity).toBe('safe');
      expect(result.score).toBeLessThan(10);
      expect(result.confidence).toBe(100);
    });

    it('has no recommended actions for default', () => {
      const result = getDefaultCrisisEval();
      expect(result.recommended_actions).toEqual([]);
    });

    it('marks all factors as false for default', () => {
      const result = getDefaultCrisisEval();
      expect(result.explicit_keywords).toBe(false);
      expect(result.implicit_hopelessness).toBe(false);
      expect(result.wellness_signal).toBe(false);
      expect(result.pattern_escalation).toBe(false);
    });
  });

  describe('Multi-factor severity assessment', () => {
    it('escalates when multiple strong factors present', async () => {
      const result = await evaluateCrisis({
        userMessage: 'Everything is hopeless, there is no point anymore',
        history: [],
        classification: {
          ...mockClassification,
          sentiment: 'Crisis',
        },
        wellness: {
          ...mockWellness,
          mood: 12,
          energy: 18,
          stress: 94,
        },
      });

      expect(result.severity).toBe('high_risk');
      expect(result.implicit_hopelessness).toBe(true);
      expect(result.wellness_signal).toBe(true);
    });

    it('identifies safety when crisis keywords present', async () => {
      // Explicit keywords should bypass LLM and immediately escalate
      const result = await evaluateCrisis({
        userMessage: 'I am going to kill myself right now',
        history: [],
        classification: mockClassification,
        wellness: mockWellness,
      });

      expect(result.severity).toBe('critical');
      expect(result.score).toBeGreaterThan(90);
      // Should not call LLM for clear cases
      expect(vi.mocked(openrouter.callOpenRouter).mock.calls.length).toBe(0);
    });
  });
});
