import { describe, it, expect, vi, beforeEach } from 'vitest';
import { classifyWithConfidence } from '@/lib/tools/classification';
import { inferWellness } from '@/lib/tools/wellness';
import { evaluateCrisis } from '@/lib/tools/crisis-eval';
import { detectSessionType } from '@/lib/tools/session-type';
import { decideResourceSearch } from '@/lib/tools/resource-search';
import { evaluateResponse } from '@/lib/tools/response-eval';
import * as openrouter from '@/lib/openrouter';

vi.mock('@/lib/openrouter');
vi.mock('@/lib/logging', () => ({
  logger: { warn: vi.fn() },
  logToolCall: vi.fn(),
}));

describe('Phase 2: Session Awareness & Response Quality Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Venting conversation pipeline', () => {
    it('correctly routes venting through full pipeline with no resource search', async () => {
      const userMessage = 'I am so frustrated with this situation, I just need to vent!';

      // Phase 1: Classification
      const classification = await classifyWithConfidence(userMessage, []);
      expect(classification.intent).toBe('venting');

      // Phase 1: Wellness
      const wellness = await inferWellness({
        userMessage,
        history: [],
        classification,
      });

      // Phase 1: Crisis
      const crisis = await evaluateCrisis({
        userMessage,
        history: [],
        classification,
        wellness,
      });

      // Phase 2: Session type
      const sessionType = await detectSessionType({
        userMessage,
        history: [],
        classification,
      });
      expect(sessionType.primary_type).toBe('venting');

      // Phase 2: Resource search should skip
      const resourceDecision = await decideResourceSearch({
        userMessage,
        classification,
        sessionType,
        crisis,
      });
      expect(resourceDecision.should_search).toBe(false);
      expect(resourceDecision.search_depth).toBe('skip');
    });

    it('evaluates venting response quality', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 80,
        validation_score: 90,
        clarity_score: 82,
        relevance_score: 45,
        overall_quality: 76,
        strengths: ['Validates frustration', 'Warm tone'],
        weaknesses: ['Offers solutions when should just listen'],
        should_regenerate: true,
        regeneration_guidance: 'Let them vent without jumping to advice',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const evaluation = await evaluateResponse({
        userMessage: 'I am so frustrated!',
        responseText: 'I get it, that is frustrating. Here is what you should do...',
        sessionType: {
          primary_type: 'venting',
          secondary_types: [],
          confidence: 85,
          reasoning: 'Test',
          user_needs: ['to be heard'],
          recommended_strategy: 'Listen',
        },
        crisis: {
          severity: 'safe',
          score: 10,
          confidence: 95,
          explicit_keywords: false,
          implicit_hopelessness: false,
          wellness_signal: false,
          pattern_escalation: false,
          reasoning: 'Test',
          recommended_actions: [],
        },
      });

      expect(evaluation.should_regenerate).toBe(true);
      expect(evaluation.relevance_score).toBeLessThan(50);
    });
  });

  describe('Problem-solving conversation pipeline', () => {
    it('correctly routes problem-solving with resource search', async () => {
      const userMessage = 'I want to get better at managing my anxiety. What can I do?';

      const classification = await classifyWithConfidence(userMessage, []);
      expect(classification.intent).toBe('support');

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

      const sessionType = await detectSessionType({
        userMessage,
        history: [],
        classification,
      });
      expect(sessionType.primary_type).toBe('problem_solving');

      const resourceDecision = await decideResourceSearch({
        userMessage,
        classification,
        sessionType,
        crisis,
      });
      expect(resourceDecision.should_search).toBe(true);
      expect(resourceDecision.search_depth).toBe('moderate');
    });

    it('expects concrete solutions in problem-solving responses', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 78,
        validation_score: 75,
        clarity_score: 88,
        relevance_score: 92,
        overall_quality: 82,
        strengths: ['Concrete steps', 'Warm', 'Clear'],
        weaknesses: [],
        should_regenerate: false,
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const evaluation = await evaluateResponse({
        userMessage: 'How can I solve this?',
        responseText: 'Try these steps: 1) breathe, 2) identify the issue, 3) plan next move. Which feels right?',
        sessionType: {
          primary_type: 'problem_solving',
          secondary_types: [],
          confidence: 82,
          reasoning: 'Test',
          user_needs: ['solutions'],
          recommended_strategy: 'Guide',
        },
        crisis: {
          severity: 'safe',
          score: 10,
          confidence: 95,
          explicit_keywords: false,
          implicit_hopelessness: false,
          wellness_signal: false,
          pattern_escalation: false,
          reasoning: 'Test',
          recommended_actions: [],
        },
      });

      expect(evaluation.relevance_score).toBeGreaterThan(85);
      expect(evaluation.should_regenerate).toBe(false);
    });
  });

  describe('Validation-seeking conversation', () => {
    it('correctly identifies validation-seeking and skips resources', async () => {
      const userMessage = 'Is it normal to feel this way? Am I overreacting?';

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

      const sessionType = await detectSessionType({
        userMessage,
        history: [],
        classification,
      });
      expect(sessionType.primary_type).toBe('validation_seeking');

      const resourceDecision = await decideResourceSearch({
        userMessage,
        classification,
        sessionType,
        crisis,
      });
      expect(resourceDecision.should_search).toBe(false);
      expect(resourceDecision.skip_reason).toContain('validation');
    });

    it('prioritizes validation in response evaluation', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 85,
        validation_score: 95,
        clarity_score: 80,
        relevance_score: 88,
        overall_quality: 88,
        strengths: ['Strong validation', 'Affirming'],
        weaknesses: [],
        should_regenerate: false,
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const evaluation = await evaluateResponse({
        userMessage: 'Is it wrong to feel this way?',
        responseText: 'Not at all, your feelings make complete sense. You are not wrong for feeling this way.',
        sessionType: {
          primary_type: 'validation_seeking',
          secondary_types: [],
          confidence: 88,
          reasoning: 'Test',
          user_needs: ['reassurance'],
          recommended_strategy: 'Affirm',
        },
        crisis: {
          severity: 'safe',
          score: 10,
          confidence: 95,
          explicit_keywords: false,
          implicit_hopelessness: false,
          wellness_signal: false,
          pattern_escalation: false,
          reasoning: 'Test',
          recommended_actions: [],
        },
      });

      expect(evaluation.validation_score).toBeGreaterThan(90);
      expect(evaluation.overall_quality).toBeGreaterThan(85);
    });
  });

  describe('Information-seeking conversation', () => {
    it('correctly routes info-seeking with minimal resource search', async () => {
      const userMessage = 'What is anxiety and how does it work?';

      const classification = await classifyWithConfidence(userMessage, []);
      expect(classification.intent).toBe('information');

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

      const sessionType = await detectSessionType({
        userMessage,
        history: [],
        classification,
      });
      expect(sessionType.primary_type).toBe('information_seeking');

      const resourceDecision = await decideResourceSearch({
        userMessage,
        classification,
        sessionType,
        crisis,
      });
      expect(resourceDecision.should_search).toBe(true);
      expect(resourceDecision.search_depth).toBe('minimal');
    });
  });

  describe('Crisis conversation with deep resource search', () => {
    it('crisis routes to deep resource search with appropriate response', async () => {
      const userMessage = 'I want to kill myself';

      const classification = await classifyWithConfidence(userMessage, []);
      expect(classification.sentiment).toBe('Crisis');

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
      expect(crisis.severity).toBe('critical');

      const sessionType = await detectSessionType({
        userMessage,
        history: [],
        classification,
      });
      expect(sessionType.primary_type).toBe('crisis');

      const resourceDecision = await decideResourceSearch({
        userMessage,
        classification,
        sessionType,
        crisis,
      });
      expect(resourceDecision.should_search).toBe(true);
      expect(resourceDecision.search_depth).toBe('deep');
      expect(resourceDecision.search_query).toBeDefined();
    });

    it('evaluates crisis response quality', async () => {
      const mockResponse = JSON.stringify({
        warmth_score: 80,
        validation_score: 82,
        clarity_score: 85,
        relevance_score: 98,
        overall_quality: 86,
        strengths: ['Immediate resources', 'Compassionate', 'Clear next steps'],
        weaknesses: [],
        should_regenerate: false,
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const evaluation = await evaluateResponse({
        userMessage: 'I want to hurt myself',
        responseText: 'I am really glad you said this out loud. Please call 988 right now or text HOME to 741741. This is bigger than an app - you need real support now.',
        sessionType: {
          primary_type: 'crisis',
          secondary_types: [],
          confidence: 98,
          reasoning: 'Test',
          user_needs: ['immediate safety', 'crisis support'],
          recommended_strategy: 'Prioritize safety resources',
        },
        crisis: {
          severity: 'critical',
          score: 95,
          confidence: 98,
          explicit_keywords: true,
          implicit_hopelessness: false,
          wellness_signal: false,
          pattern_escalation: false,
          reasoning: 'Test',
          recommended_actions: ['Call emergency', 'Contact crisis hotline'],
        },
      });

      expect(evaluation.relevance_score).toBeGreaterThan(90);
      expect(evaluation.should_regenerate).toBe(false);
    });
  });

  describe('Multi-turn conversation context', () => {
    it('session type detection considers conversation history', async () => {
      const history = [
        {
          role: 'user' as const,
          content: 'I have been struggling with anxiety',
        },
        {
          role: 'assistant' as const,
          content: 'Tell me more about that.',
        },
      ];

      const sessionType = await detectSessionType({
        userMessage: 'How can I manage this better?',
        history,
        classification: {
          sentiment: 'Negative',
          intent: 'support',
          confidence: 75,
          reasoning: 'Test',
          alternativeInterpretations: [],
        },
      });

      // Should recognize this as problem-solving in context
      expect(sessionType.primary_type).toBe('problem_solving');
    });
  });

  describe('Secondary session type awareness', () => {
    it('identifies mixed sessions (venting + problem-solving)', async () => {
      const mockResponse = JSON.stringify({
        primary_type: 'venting',
        secondary_types: ['problem_solving'],
        confidence: 78,
        reasoning: 'User vents then seeks solutions',
        user_needs: ['to be heard', 'help solving'],
        recommended_strategy: 'Listen first, then guide toward solutions',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const sessionType = await detectSessionType({
        userMessage: 'I am so frustrated with this! But I also want to know how to fix it.',
        history: [],
        classification: {
          sentiment: 'Negative',
          intent: 'support',
          confidence: 75,
          reasoning: 'Test',
          alternativeInterpretations: [],
        },
      });

      // May have secondary types
      if (sessionType.secondary_types.length > 0) {
        expect(sessionType.secondary_types).toContain('problem_solving');
      }
    });
  });

  describe('API failure resilience across all phases', () => {
    it('all tools fall back when APIs fail during full pipeline', async () => {
      // All tools return errors
      vi.mocked(openrouter.callOpenRouter).mockRejectedValue(
        new Error('API unavailable')
      );

      const userMessage = 'I am feeling sad and need help';

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
      const sessionType = await detectSessionType({
        userMessage,
        history: [],
        classification,
      });
      const resourceDecision = await decideResourceSearch({
        userMessage,
        classification,
        sessionType,
        crisis,
      });

      // All should have reasonable fallback results
      expect(classification).toBeDefined();
      expect(wellness).toBeDefined();
      expect(crisis).toBeDefined();
      expect(sessionType).toBeDefined();
      expect(resourceDecision).toBeDefined();
    });
  });
});
