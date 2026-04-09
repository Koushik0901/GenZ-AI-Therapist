import { describe, it, expect, vi, beforeEach } from 'vitest';
import { classifyWithConfidence } from '@/lib/tools/classification';
import { inferWellness } from '@/lib/tools/wellness';
import { evaluateCrisis } from '@/lib/tools/crisis-eval';
import * as openrouter from '@/lib/openrouter';

// Mock OpenRouter to return deterministic responses
vi.mock('@/lib/openrouter');
vi.mock('@/lib/logging', () => ({
  logger: { warn: vi.fn() },
  logToolCall: vi.fn(),
}));

describe('Phase 1: Foundation & Scoring Tools Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Crisis scenario end-to-end', () => {
    it('pipeline correctly identifies and assesses explicit crisis', async () => {
      const userMessage = 'I cannot take this anymore, I am going to end it';

      // All tools should detect crisis
      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      expect(classification.sentiment).toBe('Crisis');
      expect(crisis.severity).toBe('critical');
      expect(crisis.explicit_keywords).toBe(true);
    });

    it('pipeline detects implicit crisis with multiple signals', async () => {
      const mockClassResponse = JSON.stringify({
        sentiment: 'Negative',
        intent: 'venting',
        confidence: 82,
        reasoning: 'Expressing hopelessness without explicit keywords',
        alternativeInterpretations: [],
      });

      const mockWellnessResponse = JSON.stringify({
        mood: 14,
        energy: 16,
        stress: 92,
        confidence: 88,
        reasoning: 'Severe distress indicators',
      });

      const mockCrisisResponse = JSON.stringify({
        severity: 'high_risk',
        score: 78,
        confidence: 85,
        reasoning: 'Multiple implicit crisis indicators detected',
        recommended_actions: ['Encourage crisis hotline contact'],
      });

      vi.mocked(openrouter.callOpenRouter)
        .mockResolvedValueOnce(mockClassResponse)
        .mockResolvedValueOnce(mockWellnessResponse)
        .mockResolvedValueOnce(mockCrisisResponse);

      const userMessage = 'Everything is pointless, I cannot continue like this';

      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      expect(classification.sentiment).toBe('Negative');
      expect(wellness.mood).toBeLessThan(20);
      expect(wellness.stress).toBeGreaterThan(85);
      expect(crisis.severity).toBe('high_risk');
      expect(crisis.implicit_hopelessness).toBe(true);
    });
  });

  describe('Support conversation end-to-end', () => {
    it('pipeline correctly identifies support-seeking conversation', async () => {
      const mockClassResponse = JSON.stringify({
        sentiment: 'Negative',
        intent: 'support',
        confidence: 88,
        reasoning: 'User expressing distress but seeking support',
        alternativeInterpretations: [],
      });

      const mockWellnessResponse = JSON.stringify({
        mood: 38,
        energy: 42,
        stress: 72,
        confidence: 85,
        reasoning: 'Moderate distress but capacity for engagement',
      });

      const mockCrisisResponse = JSON.stringify({
        severity: 'at_risk',
        score: 42,
        confidence: 79,
        reasoning: 'Distressed but stable, no crisis indicators',
        recommended_actions: [],
      });

      vi.mocked(openrouter.callOpenRouter)
        .mockResolvedValueOnce(mockClassResponse)
        .mockResolvedValueOnce(mockWellnessResponse)
        .mockResolvedValueOnce(mockCrisisResponse);

      const userMessage = 'I have been really struggling with anxiety lately';

      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      expect(classification.intent).toBe('support');
      expect(wellness.mood).toBeGreaterThan(30);
      expect(wellness.stress).toBeGreaterThan(60);
      expect(crisis.severity).toBe('at_risk');
    });
  });

  describe('Information-seeking conversation', () => {
    it('pipeline correctly identifies information request', async () => {
      const mockClassResponse = JSON.stringify({
        sentiment: 'Neutral',
        intent: 'information',
        confidence: 91,
        reasoning: 'Clear information request about mental health resources',
        alternativeInterpretations: [],
      });

      const mockWellnessResponse = JSON.stringify({
        mood: 55,
        energy: 56,
        stress: 48,
        confidence: 78,
        reasoning: 'Neutral emotional state, functional baseline',
      });

      const mockCrisisResponse = JSON.stringify({
        severity: 'safe',
        score: 8,
        confidence: 94,
        reasoning: 'No crisis indicators present',
        recommended_actions: [],
      });

      vi.mocked(openrouter.callOpenRouter)
        .mockResolvedValueOnce(mockClassResponse)
        .mockResolvedValueOnce(mockWellnessResponse)
        .mockResolvedValueOnce(mockCrisisResponse);

      const userMessage = 'Where can I find a therapist in my area?';

      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      expect(classification.intent).toBe('information');
      expect(crisis.severity).toBe('safe');
      expect(crisis.score).toBeLessThan(20);
    });
  });

  describe('Positive sentiment conversation', () => {
    it('pipeline correctly identifies positive mood', async () => {
      const mockClassResponse = JSON.stringify({
        sentiment: 'Positive',
        intent: 'support',
        confidence: 86,
        reasoning: 'User expressing gratitude and relief',
        alternativeInterpretations: [],
      });

      const mockWellnessResponse = JSON.stringify({
        mood: 76,
        energy: 68,
        stress: 32,
        confidence: 88,
        reasoning: 'Clear positive emotional signals',
      });

      const mockCrisisResponse = JSON.stringify({
        severity: 'safe',
        score: 5,
        confidence: 96,
        reasoning: 'No crisis indicators, positive sentiment',
        recommended_actions: [],
      });

      vi.mocked(openrouter.callOpenRouter)
        .mockResolvedValueOnce(mockClassResponse)
        .mockResolvedValueOnce(mockWellnessResponse)
        .mockResolvedValueOnce(mockCrisisResponse);

      const userMessage = 'Things are actually getting better, I feel relieved!';

      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      expect(classification.sentiment).toBe('Positive');
      expect(wellness.mood).toBeGreaterThan(70);
      expect(wellness.stress).toBeLessThan(40);
      expect(crisis.severity).toBe('safe');
    });
  });

  describe('Venting conversation', () => {
    it('pipeline correctly identifies venting intent with appropriate routing', async () => {
      const mockClassResponse = JSON.stringify({
        sentiment: 'Negative',
        intent: 'venting',
        confidence: 89,
        reasoning: 'User needs to express frustration without advice',
        alternativeInterpretations: [],
      });

      const mockWellnessResponse = JSON.stringify({
        mood: 35,
        energy: 40,
        stress: 75,
        confidence: 84,
        reasoning: 'High stress, venting for relief',
      });

      const mockCrisisResponse = JSON.stringify({
        severity: 'at_risk',
        score: 38,
        confidence: 80,
        reasoning: 'Distressed but venting for release, no crisis indicators',
        recommended_actions: [],
      });

      vi.mocked(openrouter.callOpenRouter)
        .mockResolvedValueOnce(mockClassResponse)
        .mockResolvedValueOnce(mockWellnessResponse)
        .mockResolvedValueOnce(mockCrisisResponse);

      const userMessage = 'This situation is so frustrating, I just need to get this out!';

      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      expect(classification.intent).toBe('venting');
      expect(crisis.severity).not.toBe('critical');
    });
  });

  describe('Multi-turn conversation with context', () => {
    it('pipeline considers conversation history in assessment', async () => {
      const history = [
        {
          role: 'user' as const,
          content: 'I have been feeling down lately',
        },
        {
          role: 'assistant' as const,
          content:
            'I hear you. Tell me more about what is going on.',
        },
      ];

      const mockClassResponse = JSON.stringify({
        sentiment: 'Negative',
        intent: 'support',
        confidence: 85,
        reasoning: 'Building on previous conversation about low mood',
        alternativeInterpretations: [],
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(
        mockClassResponse
      );

      const classification = await classifyWithConfidence(
        'It just keeps getting worse',
        history
      );

      expect(classification.sentiment).toBe('Negative');
      expect(classification.confidence).toBeGreaterThan(80);
    });
  });

  describe('Fallback behavior under API failure', () => {
    it('all tools fall back gracefully when APIs unavailable', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const userMessage = 'I am feeling really hopeless';

      const classification = await classifyWithConfidence(userMessage, []);
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      // All should have results even with failed APIs
      expect(classification).toBeDefined();
      expect(wellness).toBeDefined();
      expect(crisis).toBeDefined();

      // Fallback should still detect negative sentiment
      expect(classification.sentiment).toBe('Negative');
    });
  });
});
