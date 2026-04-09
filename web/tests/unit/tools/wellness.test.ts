import { describe, it, expect, vi, beforeEach } from 'vitest';
import { inferWellness, getDefaultWellness } from '@/lib/tools/wellness';
import * as openrouter from '@/lib/openrouter';
import type { Classification } from '@/lib/tools/classification';

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
  reasoning: 'Test classification',
  alternativeInterpretations: [],
};

describe('Wellness Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('inferWellness', () => {
    it('infers low mood and high stress for crisis sentiment', async () => {
      const mockResponse = JSON.stringify({
        mood: 15,
        energy: 22,
        stress: 88,
        confidence: 92,
        reasoning: 'Crisis sentiment with hopelessness indicators',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const crisisClassification: Classification = {
        ...mockClassification,
        sentiment: 'Crisis',
      };

      const result = await inferWellness({
        userMessage: 'I want to end this',
        history: [],
        classification: crisisClassification,
      });

      expect(result.mood).toBeLessThan(30);
      expect(result.stress).toBeGreaterThan(80);
      expect(result.confidence).toBeGreaterThan(85);
    });

    it('infers positive scores for positive sentiment', async () => {
      const mockResponse = JSON.stringify({
        mood: 78,
        energy: 72,
        stress: 28,
        confidence: 86,
        reasoning: 'Positive sentiment with relief indicators',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const positiveClassification: Classification = {
        ...mockClassification,
        sentiment: 'Positive',
      };

      const result = await inferWellness({
        userMessage: 'Things are actually looking up!',
        history: [],
        classification: positiveClassification,
      });

      expect(result.mood).toBeGreaterThan(70);
      expect(result.energy).toBeGreaterThan(60);
      expect(result.stress).toBeLessThan(35);
    });

    it('infers moderate scores for neutral sentiment', async () => {
      const mockResponse = JSON.stringify({
        mood: 55,
        energy: 52,
        stress: 48,
        confidence: 72,
        reasoning: 'Neutral sentiment, balanced emotional state',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const neutralClassification: Classification = {
        ...mockClassification,
        sentiment: 'Neutral',
      };

      const result = await inferWellness({
        userMessage: 'Just another day',
        history: [],
        classification: neutralClassification,
      });

      expect(result.mood).toBeGreaterThan(40);
      expect(result.mood).toBeLessThan(70);
      expect(result.stress).toBeGreaterThan(35);
      expect(result.stress).toBeLessThan(65);
    });

    it('uses fallback on LLM failure', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await inferWellness({
        userMessage: 'I am exhausted and burned out',
        history: [],
        classification: mockClassification,
      });

      // Fallback should detect low energy
      expect(result.energy).toBeLessThan(50);
      expect(result.confidence).toBeLessThan(75);
    });

    it('adjusts scores based on energy keywords', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const result = await inferWellness({
        userMessage: 'I am completely exhausted and drained',
        history: [],
        classification: mockClassification,
      });

      expect(result.energy).toBeLessThan(40);
    });

    it('adjusts scores based on panic keywords', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const result = await inferWellness({
        userMessage: 'I am panicking and overwhelmed',
        history: [],
        classification: mockClassification,
      });

      expect(result.stress).toBeGreaterThan(70);
      expect(result.mood).toBeLessThan(50);
    });

    it('considers conversation history in inference', async () => {
      const mockResponse = JSON.stringify({
        mood: 48,
        energy: 45,
        stress: 62,
        confidence: 81,
        reasoning: 'Building on conversation showing escalation',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const history = [
        { role: 'user' as const, content: 'Things have been really hard lately' },
        {
          role: 'assistant' as const,
          content: 'I hear you. Tell me what's going on.',
        },
        {
          role: 'user' as const,
          content: 'It just keeps getting worse',
        },
      ];

      const result = await inferWellness({
        userMessage: 'I do not know how much longer I can handle this',
        history,
        classification: mockClassification,
      });

      expect(result.confidence).toBeGreaterThan(75);
    });

    it('bounds scores between 0-100', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const result = await inferWellness({
        userMessage: 'Test message',
        history: [],
        classification: mockClassification,
      });

      expect(result.mood).toBeGreaterThanOrEqual(0);
      expect(result.mood).toBeLessThanOrEqual(100);
      expect(result.energy).toBeGreaterThanOrEqual(0);
      expect(result.energy).toBeLessThanOrEqual(100);
      expect(result.stress).toBeGreaterThanOrEqual(0);
      expect(result.stress).toBeLessThanOrEqual(100);
    });
  });

  describe('getDefaultWellness', () => {
    it('returns neutral balanced default', () => {
      const result = getDefaultWellness();

      expect(result.mood).toBe(55);
      expect(result.energy).toBe(50);
      expect(result.stress).toBe(50);
      expect(result.confidence).toBe(30);
    });

    it('indicates low confidence for default', () => {
      const result = getDefaultWellness();
      expect(result.confidence).toBeLessThan(40);
    });
  });

  describe('Fallback score adjustments', () => {
    it('increases stress and decreases energy for positive markers', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const positiveClassification: Classification = {
        ...mockClassification,
        sentiment: 'Positive',
      };

      const result = await inferWellness({
        userMessage: 'I am proud and feeling better today, really grateful',
        history: [],
        classification: positiveClassification,
      });

      expect(result.mood).toBeGreaterThan(65);
      expect(result.stress).toBeLessThan(50);
    });

    it('boosts energy slightly for planning/motivation keywords', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const motivationalClassification: Classification = {
        ...mockClassification,
        intent: 'motivational',
      };

      const result = await inferWellness({
        userMessage: 'Can we make a plan to help me move forward?',
        history: [],
        classification: motivationalClassification,
      });

      expect(result.energy).toBeGreaterThan(45);
    });
  });
});
