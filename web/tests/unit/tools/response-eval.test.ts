import { describe, it, expect, vi, beforeEach } from 'vitest';
import { evaluateResponse, getDefaultResponseEval } from '@/lib/tools/response-eval';
import * as openrouter from '@/lib/openrouter';
import type { SessionTypeDetection } from '@/lib/tools/session-type';
import type { CrisisEvaluation } from '@/lib/tools/crisis-eval';

vi.mock('@/lib/openrouter');
vi.mock('@/lib/logging', () => ({
  logger: { warn: vi.fn() },
  logToolCall: vi.fn(),
}));

const mockSessionType: SessionTypeDetection = {
  primary_type: 'support',
  secondary_types: [],
  confidence: 75,
  reasoning: 'Test',
  user_needs: ['understanding'],
  recommended_strategy: 'Be supportive',
};

const mockCrisis: CrisisEvaluation = {
  severity: 'safe',
  score: 10,
  confidence: 95,
  explicit_keywords: false,
  implicit_hopelessness: false,
  wellness_signal: false,
  pattern_escalation: false,
  reasoning: 'Test',
  recommended_actions: [],
};

describe('Response Evaluation Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('evaluateResponse', () => {
    it('scores high-quality supportive response', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 85,
        validation_score: 88,
        clarity_score: 82,
        relevance_score: 80,
        overall_quality: 84,
        strengths: ['Warm tone', 'Validates feelings', 'Clear advice'],
        weaknesses: [],
        should_regenerate: false,
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateResponse({
        userMessage: 'I am feeling overwhelmed',
        responseText:
          'That sounds really hard. Your feelings make total sense. Here are some things that might help: take a break, breathe, reach out to someone.',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.overall_quality).toBeGreaterThan(80);
      expect(result.should_regenerate).toBe(false);
    });

    it('identifies cold/clinical response needing warmth', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 45,
        validation_score: 50,
        clarity_score: 78,
        relevance_score: 70,
        overall_quality: 58,
        strengths: ['Clear structure'],
        weaknesses: ['Too clinical', 'Lacks validation'],
        should_regenerate: true,
        regeneration_guidance: 'Use more human, conversational language',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateResponse({
        userMessage: 'I am struggling',
        responseText: 'Emotional distress is a recognized psychological state. Consider cognitive behavioral interventions.',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.warmth_score).toBeLessThan(60);
      expect(result.should_regenerate).toBe(true);
    });

    it('detects validation failures for venting response', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 72,
        validation_score: 42,
        clarity_score: 80,
        relevance_score: 35,
        overall_quality: 58,
        strengths: ['Clear'],
        weaknesses: ['Ignores validation need', 'Offers solutions during venting'],
        should_regenerate: true,
        regeneration_guidance: 'Let them vent without jumping to solutions',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateResponse({
        userMessage: 'I am so frustrated with this situation!',
        responseText: 'Here is what you should do to fix this: Step 1... Step 2...',
        sessionType: {
          ...mockSessionType,
          primary_type: 'venting',
        },
        crisis: mockCrisis,
      });

      expect(result.relevance_score).toBeLessThan(50);
      expect(result.should_regenerate).toBe(true);
    });

    it('evaluates problem-solving response quality', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 75,
        validation_score: 70,
        clarity_score: 85,
        relevance_score: 88,
        overall_quality: 79,
        strengths: ['Concrete steps', 'Warm tone', 'Actionable'],
        weaknesses: [],
        should_regenerate: false,
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateResponse({
        userMessage: 'How can I manage my anxiety better?',
        responseText: 'Great question. Try these approaches: 1) breathing exercises, 2) regular exercise, 3) talk to someone you trust. See what feels right.',
        sessionType: {
          ...mockSessionType,
          primary_type: 'problem_solving',
        },
        crisis: mockCrisis,
      });

      expect(result.relevance_score).toBeGreaterThan(80);
      expect(result.should_regenerate).toBe(false);
    });

    it('flags crisis response missing emergency resources', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 68,
        validation_score: 75,
        clarity_score: 70,
        relevance_score: 35,
        overall_quality: 56,
        strengths: ['Validates'],
        weaknesses: ['Missing crisis resources', 'No emergency hotline info'],
        should_regenerate: true,
        regeneration_guidance: 'Include crisis hotline and emergency services info',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateResponse({
        userMessage: 'I want to hurt myself',
        responseText: 'That sounds really painful. You deserve support and care.',
        sessionType: {
          ...mockSessionType,
          primary_type: 'crisis',
        },
        crisis: {
          ...mockCrisis,
          severity: 'high_risk',
          score: 78,
        },
      });

      expect(result.relevance_score).toBeLessThan(50);
      expect(result.should_regenerate).toBe(true);
    });

    it('uses fallback on LLM error', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await evaluateResponse({
        userMessage: 'I am struggling',
        responseText: 'That sounds hard. I hear you. Here is what might help...',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.overall_quality).toBeDefined();
      expect(result.warmth_score).toBeDefined();
      expect(result.validation_score).toBeDefined();
    });

    it('detects too-long responses', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const longResponse =
        'This is a very long response. '.repeat(50) + 'It goes on and on.';

      const result = await evaluateResponse({
        userMessage: 'I need help',
        responseText: longResponse,
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.clarity_score).toBeLessThan(70);
    });

    it('scores validation quality highly when present', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await evaluateResponse({
        userMessage: 'I feel sad',
        responseText: 'Your feelings are completely valid. It makes total sense that you feel this way. You are not alone in this.',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.validation_score).toBeGreaterThan(70);
    });

    it('scores warmth high for conversational language', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await evaluateResponse({
        userMessage: 'I am exhausted',
        responseText: 'Yeah, that sounds honestly exhausting. Not gonna lie, that is a lot. But you are handling it, and that matters.',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.warmth_score).toBeGreaterThan(65);
    });

    it('recommends regeneration when quality < 65', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 50,
        validation_score: 45,
        clarity_score: 55,
        relevance_score: 48,
        overall_quality: 49,
        strengths: [],
        weaknesses: ['Too clinical', 'No validation', 'Unclear'],
        should_regenerate: true,
        regeneration_guidance: 'Try a warmer, more validating tone with clearer structure',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await evaluateResponse({
        userMessage: 'I am sad',
        responseText: 'Sadness is an emotion.',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.overall_quality).toBeLessThan(65);
      expect(result.should_regenerate).toBe(true);
      expect(result.regeneration_guidance).toBeDefined();
    });

    it('detects validation-negating language', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await evaluateResponse({
        userMessage: 'I feel bad',
        responseText: 'But actually, you should not feel that way. Here is the real issue...',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(result.validation_score).toBeLessThan(55);
    });
  });

  describe('getDefaultResponseEval', () => {
    it('returns passing evaluation', () => {
      const result = getDefaultResponseEval();

      expect(result.overall_quality).toBeGreaterThan(65);
      expect(result.should_regenerate).toBe(false);
    });

    it('has all scores > 65', () => {
      const result = getDefaultResponseEval();

      expect(result.warmth_score).toBeGreaterThan(65);
      expect(result.validation_score).toBeGreaterThan(65);
      expect(result.clarity_score).toBeGreaterThan(65);
      expect(result.relevance_score).toBeGreaterThan(65);
    });

    it('includes strengths in default', () => {
      const result = getDefaultResponseEval();
      expect(result.strengths.length).toBeGreaterThan(0);
    });

    it('has no weaknesses in default', () => {
      const result = getDefaultResponseEval();
      expect(result.weaknesses).toEqual([]);
    });
  });

  describe('Dimension weighting', () => {
    it('weights validation heavily (35%)', async () => {
      // High validation should boost overall score more than other dimensions
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const validatingResponse = await evaluateResponse({
        userMessage: 'I feel bad',
        responseText: 'Your feelings are valid, you are not alone, I understand',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      const nonValidatingResponse = await evaluateResponse({
        userMessage: 'I feel bad',
        responseText: 'Here is what you should do instead',
        sessionType: mockSessionType,
        crisis: mockCrisis,
      });

      expect(validatingResponse.overall_quality).toBeGreaterThan(
        nonValidatingResponse.overall_quality
      );
    });
  });
});
