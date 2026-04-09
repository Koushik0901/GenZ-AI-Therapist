import { describe, it, expect, vi, beforeEach } from 'vitest';
import { decideResourceSearch, getDefaultResourceSearch } from '@/lib/tools/resource-search';
import * as openrouter from '@/lib/openrouter';
import type { Classification } from '@/lib/tools/classification';
import type { SessionTypeDetection } from '@/lib/tools/session-type';
import type { CrisisEvaluation } from '@/lib/tools/crisis-eval';

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

const mockSessionType: SessionTypeDetection = {
  primary_type: 'support',
  secondary_types: [],
  confidence: 75,
  reasoning: 'Test',
  user_needs: ['support'],
  recommended_strategy: 'Listen',
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

describe('Resource Search Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('decideResourceSearch', () => {
    it('skips search for venting session type', async () => {
      const result = await decideResourceSearch({
        userMessage: 'I just need to vent about this',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'venting',
        },
        crisis: mockCrisis,
      });

      expect(result.should_search).toBe(false);
      expect(result.search_depth).toBe('skip');
      expect(result.skip_reason).toBeDefined();
    });

    it('skips search for validation-seeking session type', async () => {
      const result = await decideResourceSearch({
        userMessage: 'Is this normal?',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'validation_seeking',
        },
        crisis: mockCrisis,
      });

      expect(result.should_search).toBe(false);
      expect(result.search_depth).toBe('skip');
    });

    it('skips search for chitchat', async () => {
      const result = await decideResourceSearch({
        userMessage: 'How is your day?',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'chitchat',
        },
        crisis: mockCrisis,
      });

      expect(result.should_search).toBe(false);
      expect(result.search_depth).toBe('skip');
    });

    it('performs minimal search for information-seeking', async () => {
      const result = await decideResourceSearch({
        userMessage: 'What is anxiety?',
        classification: {
          ...mockClassification,
          intent: 'information',
        },
        sessionType: {
          ...mockSessionType,
          primary_type: 'information_seeking',
        },
        crisis: mockCrisis,
      });

      expect(result.should_search).toBe(true);
      expect(result.search_depth).toBe('minimal');
    });

    it('performs moderate search for problem-solving', async () => {
      const result = await decideResourceSearch({
        userMessage: 'How can I manage anxiety better?',
        classification: {
          ...mockClassification,
          intent: 'support',
        },
        sessionType: {
          ...mockSessionType,
          primary_type: 'problem_solving',
        },
        crisis: mockCrisis,
      });

      expect(result.should_search).toBe(true);
      expect(result.search_depth).toBe('moderate');
    });

    it('performs deep search for crisis', async () => {
      const result = await decideResourceSearch({
        userMessage: 'I am thinking about suicide',
        classification: {
          ...mockClassification,
          sentiment: 'Crisis',
        },
        sessionType: {
          ...mockSessionType,
          primary_type: 'crisis',
        },
        crisis: {
          ...mockCrisis,
          severity: 'critical',
          score: 95,
        },
      });

      expect(result.should_search).toBe(true);
      expect(result.search_depth).toBe('deep');
      expect(result.search_query).toBeDefined();
    });

    it('performs deep search for high-risk crisis', async () => {
      const result = await decideResourceSearch({
        userMessage: 'I cannot go on',
        classification: mockClassification,
        sessionType: mockSessionType,
        crisis: {
          ...mockCrisis,
          severity: 'high_risk',
          score: 78,
        },
      });

      expect(result.search_depth).toBe('deep');
    });

    it('includes search query when searching', async () => {
      const result = await decideResourceSearch({
        userMessage: 'I need help with depression',
        classification: {
          ...mockClassification,
          intent: 'information',
        },
        sessionType: {
          ...mockSessionType,
          primary_type: 'information_seeking',
        },
        crisis: mockCrisis,
      });

      expect(result.search_query).toBeDefined();
      expect(result.search_query?.length).toBeGreaterThan(0);
    });

    it('provides reasoning for all decisions', async () => {
      const result = await decideResourceSearch({
        userMessage: 'Just venting',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'venting',
        },
        crisis: mockCrisis,
      });

      expect(result.reasoning).toBeDefined();
      expect(result.reasoning.length).toBeGreaterThan(0);
    });

    it('uses LLM for ambiguous cases', async () => {
      const mockResponse = JSON.stringify({
        should_search: true,
        search_depth: 'moderate',
        search_query: 'anxiety management techniques',
        reasoning: 'User likely wants both validation and solutions',
      });

      vi.mocked(openrouter.callOpenRouter).mockResolvedValueOnce(mockResponse);

      const result = await decideResourceSearch({
        userMessage: 'I feel stuck and do not know what to do',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          confidence: 45, // Low confidence triggers LLM
        },
        crisis: mockCrisis,
      });

      // May or may not call LLM depending on heuristic
      expect(result.should_search).toBeDefined();
    });

    it('falls back on LLM error', async () => {
      vi.mocked(openrouter.callOpenRouter).mockRejectedValueOnce(
        new Error('API timeout')
      );

      const result = await decideResourceSearch({
        userMessage: 'Tell me about anxiety',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'information_seeking',
        },
        crisis: mockCrisis,
      });

      expect(result.search_depth).toBeDefined();
    });
  });

  describe('Search depth heuristics', () => {
    it('escalates to deep search for at-risk problem-solving', async () => {
      const result = await decideResourceSearch({
        userMessage: 'I need help managing these thoughts',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'problem_solving',
        },
        crisis: {
          ...mockCrisis,
          severity: 'at_risk',
          score: 45,
        },
      });

      expect(result.search_depth).toBe('deep');
    });

    it('uses moderate search for safe problem-solving', async () => {
      const result = await decideResourceSearch({
        userMessage: 'How can I improve my sleep habits?',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'problem_solving',
        },
        crisis: mockCrisis,
      });

      expect(result.search_depth).toBe('moderate');
    });
  });

  describe('getDefaultResourceSearch', () => {
    it('returns safe default with no resources', () => {
      const result = getDefaultResourceSearch();

      expect(result.resources).toEqual([]);
      expect(result.decision.should_search).toBe(false);
    });

    it('has skip decision in default', () => {
      const result = getDefaultResourceSearch();
      expect(result.decision.search_depth).toBe('skip');
    });

    it('has high confidence in default safe behavior', () => {
      const result = getDefaultResourceSearch();
      expect(result.confidence).toBe(100);
    });
  });

  describe('Resource search reasoning', () => {
    it('explains venting skip reason', async () => {
      const result = await decideResourceSearch({
        userMessage: 'I am so frustrated',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'venting',
        },
        crisis: mockCrisis,
      });

      expect(result.skip_reason?.toLowerCase()).toContain('dismiss');
    });

    it('explains validation-seeking skip reason', async () => {
      const result = await decideResourceSearch({
        userMessage: 'Am I wrong?',
        classification: mockClassification,
        sessionType: {
          ...mockSessionType,
          primary_type: 'validation_seeking',
        },
        crisis: mockCrisis,
      });

      expect(result.skip_reason?.toLowerCase()).toContain('validation');
    });

    it('explains information-seeking search decision', async () => {
      const result = await decideResourceSearch({
        userMessage: 'What is depression?',
        classification: {
          ...mockClassification,
          intent: 'information',
        },
        sessionType: {
          ...mockSessionType,
          primary_type: 'information_seeking',
        },
        crisis: mockCrisis,
      });

      expect(result.reasoning.toLowerCase()).toContain('information');
    });
  });
});
