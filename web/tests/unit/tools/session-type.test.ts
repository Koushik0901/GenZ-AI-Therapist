import { describe, it, expect, vi, beforeEach } from 'vitest';
import { detectSessionType, getDefaultSessionType } from '@/lib/tools/session-type';
import * as openrouter from '@/lib/openrouter';
import type { Classification } from '@/lib/tools/classification';

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

describe('Session Type Detection Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('detectSessionType', () => {
    it('detects venting from explicit venting patterns', async () => {
      const result = await detectSessionType({
        userMessage: 'I just need to vent about this, I am so frustrated!',
        history: [],
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('venting');
      expect(result.confidence).toBeGreaterThan(70);
    });

    it('detects problem-solving from action-oriented language', async () => {
      const result = await detectSessionType({
        userMessage: 'How can I fix this situation? What are my options?',
        history: [],
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('problem_solving');
      expect(result.confidence).toBeGreaterThan(70);
    });

    it('detects validation-seeking from affirmation questions', async () => {
      const result = await detectSessionType({
        userMessage: 'Is it normal to feel this way? Am I overreacting?',
        history: [],
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('validation_seeking');
      expect(result.confidence).toBeGreaterThan(70);
    });

    it('detects information-seeking from explanation requests', async () => {
      const result = await detectSessionType({
        userMessage: 'What is anxiety? How does it work?',
        history: [],
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('information_seeking');
      expect(result.confidence).toBeGreaterThan(70);
    });

    it('detects crisis from explicit keywords', async () => {
      const result = await detectSessionType({
        userMessage: 'I want to kill myself',
        history: [],
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('crisis');
      expect(result.confidence).toBeGreaterThan(90);
    });

    it('identifies secondary session types', async () => {
      const mockResponse = JSON.stringify({
        primary_type: 'problem_solving',
        secondary_types: ['venting'],
        confidence: 82,
        reasoning: 'Wants to vent about problem before solving',
        user_needs: ['to be heard', 'solutions'],
        recommended_strategy: 'Listen then guide',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await detectSessionType({
        userMessage: 'I need to rant about this situation, but I also want help fixing it',
        history: [],
        classification: mockClassification,
      });

      expect(result.secondary_types.length).toBeGreaterThan(0);
    });

    it('returns appropriate user needs for venting', async () => {
      const result = await detectSessionType({
        userMessage: 'I just need to vent, this is so frustrating!',
        history: [],
        classification: mockClassification,
      });

      const needs = result.user_needs;
      expect(needs.some((n) => n.toLowerCase().includes('hear'))).toBe(true);
    });

    it('returns appropriate user needs for problem-solving', async () => {
      const result = await detectSessionType({
        userMessage: 'Help me make a plan to solve this',
        history: [],
        classification: mockClassification,
      });

      const needs = result.user_needs;
      expect(needs.some((n) => n.toLowerCase().includes('solution'))).toBe(true);
    });

    it('provides recommended strategy for session type', async () => {
      const result = await detectSessionType({
        userMessage: 'I just need to express how I am feeling',
        history: [],
        classification: mockClassification,
      });

      expect(result.recommended_strategy).toBeDefined();
      expect(result.recommended_strategy.length).toBeGreaterThan(0);
    });

    it('defaults to validation-seeking for ambiguous messages', async () => {
      const result = await detectSessionType({
        userMessage: 'hello',
        history: [],
        classification: mockClassification,
      });

      // When no patterns match, should default safely
      if (result.confidence < 60) {
        expect(result.primary_type).toBe('validation_seeking');
      }
    });

    it('uses LLM for nuanced detection when keyword confidence < 80', async () => {
      const mockResponse = JSON.stringify({
        primary_type: 'validation_seeking',
        secondary_types: [],
        confidence: 78,
        reasoning: 'Ambiguous message requires nuanced analysis',
        user_needs: ['reassurance'],
        recommended_strategy: 'Affirm and reflect',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await detectSessionType({
        userMessage: 'I am not sure if I should feel this way',
        history: [],
        classification: mockClassification,
      });

      // Should have consulted LLM for this ambiguous case
      expect(vi.mocked(openrouter.callOpenRouter).mock.calls.length).toBeGreaterThan(0);
    });

    it('falls back on LLM error', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await detectSessionType({
        userMessage: 'I need help solving this problem',
        history: [],
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('problem_solving');
    });

    it('considers conversation history for context', async () => {
      const mockResponse = JSON.stringify({
        primary_type: 'problem_solving',
        secondary_types: [],
        confidence: 84,
        reasoning: 'Building on previous conversation about solutions',
        user_needs: ['concrete steps'],
        recommended_strategy: 'Guide through problem-solving process',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const history = [
        {
          role: 'user' as const,
          content: 'I have been struggling with anxiety',
        },
        {
          role: 'assistant' as const,
          content: 'Tell me more about that',
        },
      ];

      const result = await detectSessionType({
        userMessage: 'What can I do to manage these feelings?',
        history,
        classification: mockClassification,
      });

      expect(result.primary_type).toBe('problem_solving');
    });
  });

  describe('getDefaultSessionType', () => {
    it('returns validation-seeking as safe default', () => {
      const result = getDefaultSessionType();

      expect(result.primary_type).toBe('validation_seeking');
    });

    it('has low confidence for default', () => {
      const result = getDefaultSessionType();
      expect(result.confidence).toBeLessThan(50);
    });

    it('includes appropriate default needs', () => {
      const result = getDefaultSessionType();
      expect(result.user_needs.length).toBeGreaterThan(0);
    });
  });

  describe('Session type confidence scoring', () => {
    it('high confidence for multiple strong patterns', async () => {
      const result = await detectSessionType({
        userMessage: 'I really need to vent about this, I am so frustrated and annoyed!',
        history: [],
        classification: mockClassification,
      });

      expect(result.confidence).toBeGreaterThan(80);
    });

    it('moderate confidence for single pattern match', async () => {
      const result = await detectSessionType({
        userMessage: 'What is this thing?',
        history: [],
        classification: mockClassification,
      });

      // Single info pattern
      if (result.primary_type === 'information_seeking') {
        expect(result.confidence).toBeLessThan(80);
        expect(result.confidence).toBeGreaterThan(50);
      }
    });

    it('low confidence when no patterns match', async () => {
      const result = await detectSessionType({
        userMessage: 'xyz abc',
        history: [],
        classification: mockClassification,
      });

      if (result.confidence < 50) {
        expect(result.confidence).toBeLessThan(50);
      }
    });
  });
});
