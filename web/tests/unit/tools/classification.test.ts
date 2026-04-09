import { describe, it, expect, vi, beforeEach } from 'vitest';
import { classifyWithConfidence, getDefaultClassification } from '@/lib/tools/classification';
import * as openrouter from '@/lib/openrouter';

// Mock OpenRouter
vi.mock('@/lib/openrouter');
vi.mock('@/lib/logging', () => ({
  logger: { warn: vi.fn() },
  logToolCall: vi.fn(),
}));

describe('Classification Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('classifyWithConfidence', () => {
    it('classifies crisis sentiment correctly', async () => {
      const mockResponse = JSON.stringify({
        sentiment: 'Crisis',
        intent: 'crisis',
        confidence: 98,
        reasoning: 'Explicit suicide ideation detected',
        alternativeInterpretations: [],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await classifyWithConfidence('I want to kill myself', []);

      expect(result.sentiment).toBe('Crisis');
      expect(result.intent).toBe('crisis');
      expect(result.confidence).toBeGreaterThanOrEqual(90);
    });

    it('classifies positive sentiment correctly', async () => {
      const mockResponse = JSON.stringify({
        sentiment: 'Positive',
        intent: 'support',
        confidence: 85,
        reasoning: 'User expressing gratitude and relief',
        alternativeInterpretations: [],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await classifyWithConfidence('Feeling so much better today, really grateful!', []);

      expect(result.sentiment).toBe('Positive');
      expect(result.confidence).toBeGreaterThan(70);
    });

    it('classifies negative sentiment with venting intent', async () => {
      const mockResponse = JSON.stringify({
        sentiment: 'Negative',
        intent: 'venting',
        confidence: 92,
        reasoning: 'User expressing frustration and needs to vent',
        alternativeInterpretations: [
          { interpretation: 'support intent', probability: 8 },
        ],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await classifyWithConfidence(
        'I am so frustrated with this situation, I just need to vent',
        []
      );

      expect(result.sentiment).toBe('Negative');
      expect(result.intent).toBe('venting');
    });

    it('classifies information-seeking intent', async () => {
      const mockResponse = JSON.stringify({
        sentiment: 'Neutral',
        intent: 'information',
        confidence: 88,
        reasoning: 'User asking for resources and information',
        alternativeInterpretations: [],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await classifyWithConfidence('Where can I find therapy resources in my area?', []);

      expect(result.intent).toBe('information');
    });

    it('uses fallback on LLM failure', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await classifyWithConfidence('I am overwhelmed and anxious', []);

      // Fallback should detect negative sentiment
      expect(result.sentiment).toBe('Negative');
      expect(result.confidence).toBeLessThan(75);
    });

    it('handles ambiguous messages with low confidence', async () => {
      const mockResponse = JSON.stringify({
        sentiment: 'Neutral',
        intent: 'other',
        confidence: 42,
        reasoning: 'Message is vague and could be interpreted multiple ways',
        alternativeInterpretations: [
          { interpretation: 'support seeking', probability: 35 },
          { interpretation: 'chitchat', probability: 23 },
        ],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await classifyWithConfidence('what do you think?', []);

      expect(result.confidence).toBeLessThan(60);
      expect(result.alternativeInterpretations.length).toBeGreaterThan(0);
    });

    it('considers conversation history in classification', async () => {
      const mockResponse = JSON.stringify({
        sentiment: 'Negative',
        intent: 'support',
        confidence: 78,
        reasoning: 'Building on previous conversation about burnout',
        alternativeInterpretations: [],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const history = [
        { role: 'user' as const, content: 'I have been burned out for weeks' },
        {
          role: 'assistant' as const,
          content: 'That sounds really heavy. Tell me more.',
        },
      ];

      const result = await classifyWithConfidence("Today is even worse", history);

      expect(result.sentiment).toBe('Negative');
    });
  });

  describe('getDefaultClassification', () => {
    it('returns safe neutral default', () => {
      const result = getDefaultClassification();

      expect(result.sentiment).toBe('Neutral');
      expect(result.intent).toBe('support');
      expect(result.confidence).toBe(50);
    });

    it('has empty alternative interpretations', () => {
      const result = getDefaultClassification();
      expect(result.alternativeInterpretations).toEqual([]);
    });
  });

  describe('Crisis detection in fallback', () => {
    it('detects suicide keywords', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const result = await classifyWithConfidence(
        'I want to kill myself, I cannot do this anymore',
        []
      );

      expect(result.sentiment).toBe('Crisis');
      expect(result.intent).toBe('crisis');
      expect(result.confidence).toBeGreaterThan(85);
    });

    it('detects self-harm keywords', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API unavailable')
      );

      const result = await classifyWithConfidence('I am thinking about cutting myself', []);

      expect(result.sentiment).toBe('Crisis');
      expect(result.confidence).toBeGreaterThan(85);
    });
  });
});
