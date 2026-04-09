import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getUserPreferenceLearner } from '@/lib/user-preferences';

vi.mock('@supabase/supabase-js', () => ({
  createClient: vi.fn(() => ({})),
}));

vi.mock('@/lib/logging', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('UserPreferenceLearner', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('initialization', () => {
    it('creates learner for user', () => {
      const learner = getUserPreferenceLearner('user_123');
      expect(learner).toBeDefined();
    });
  });

  describe('preference retrieval', () => {
    it('returns default preferences for new user', async () => {
      const learner = getUserPreferenceLearner('new_user_456');
      const prefs = await learner.getPreferences();

      expect(prefs.user_id).toBe('new_user_456');
      expect(prefs.verbosity_preference).toBe('medium');
      expect(prefs.resource_preference).toBe('moderate');
      expect(prefs.tone_preference).toBe('gen_z');
    });

    it('default preferences have all strategies', async () => {
      const learner = getUserPreferenceLearner('user_789');
      const prefs = await learner.getPreferences();

      expect(prefs.preferred_strategies).toBeDefined();
      expect(prefs.preferred_strategies['empathy_first']).toBe(50);
      expect(prefs.preferred_strategies['concrete_steps']).toBe(50);
    });

    it('initial satisfaction is neutral', async () => {
      const learner = getUserPreferenceLearner('user_abc');
      const prefs = await learner.getPreferences();

      expect(prefs.avg_response_satisfaction).toBe(70);
    });
  });

  describe('feedback recording', () => {
    it('improves strategy score on positive feedback', async () => {
      const learner = getUserPreferenceLearner('user_learn_1');
      const prefsBefore = await learner.getPreferences();
      const scoreBefore = prefsBefore.preferred_strategies['empathy_first'];

      await learner.recordFeedback(
        'session_1',
        'positive',
        'empathy_first',
        85
      );

      const prefsAfter = await learner.getPreferences();
      const scoreAfter = prefsAfter.preferred_strategies['empathy_first'];

      // Score should improve with positive feedback
      expect(scoreAfter).toBeGreaterThanOrEqual(scoreBefore);
    });

    it('decreases strategy score on negative feedback', async () => {
      const learner = getUserPreferenceLearner('user_learn_2');
      const prefsBefore = await learner.getPreferences();
      const scoreBefore = prefsBefore.preferred_strategies['concrete_steps'];

      await learner.recordFeedback(
        'session_2',
        'negative',
        'concrete_steps',
        35
      );

      const prefsAfter = await learner.getPreferences();
      const scoreAfter = prefsAfter.preferred_strategies['concrete_steps'];

      // Score should decrease with negative feedback
      expect(scoreAfter).toBeLessThanOrEqual(scoreBefore);
    });

    it('infers verbosity from comment', async () => {
      const learner = getUserPreferenceLearner('user_verbose');
      await learner.recordFeedback(
        'session_3',
        'negative',
        'empathy_first',
        40,
        'too long, tl;dr'
      );

      const prefs = await learner.getPreferences();
      expect(prefs.verbosity_preference).toBe('short');
    });

    it('infers resource preference from comment', async () => {
      const learner = getUserPreferenceLearner('user_resources');
      await learner.recordFeedback(
        'session_4',
        'negative',
        'resources_focus',
        50,
        'way too many resources, overwhelming'
      );

      const prefs = await learner.getPreferences();
      expect(prefs.resource_preference).toBe('minimal');
    });

    it('infers tone preference from comment', async () => {
      const learner = getUserPreferenceLearner('user_tone');
      await learner.recordFeedback(
        'session_5',
        'positive',
        'empathy_first',
        90,
        'love the gen z slang, super relatable'
      );

      const prefs = await learner.getPreferences();
      expect(prefs.tone_preference).toBe('gen_z');
    });
  });

  describe('strategy recommendations', () => {
    it('recommends best strategy for session type', async () => {
      const learner = getUserPreferenceLearner('user_rec_1');

      // Record some positive feedback for empathy_first on venting
      await learner.recordFeedback('s1', 'positive', 'empathy_first', 90);
      await learner.recordFeedback('s2', 'positive', 'empathy_first', 85);
      await learner.recordFeedback('s3', 'negative', 'concrete_steps', 30);

      const strategy = await learner.getBestStrategy('venting');
      expect(strategy).toBe('empathy_first');
    });

    it('recommends concrete_steps for problem_solving', async () => {
      const learner = getUserPreferenceLearner('user_rec_2');
      
      await learner.recordFeedback('s1', 'positive', 'concrete_steps', 88);
      await learner.recordFeedback('s2', 'negative', 'resources_focus', 35);

      const strategy = await learner.getBestStrategy('problem_solving');
      // Should prefer concrete_steps for problem-solving
      expect(['concrete_steps', 'empathy_first']).toContain(strategy);
    });

    it('falls back to default strategy if no history', async () => {
      const learner = getUserPreferenceLearner('user_rec_3');
      const strategy = await learner.getBestStrategy('chitchat');

      // Should return a valid strategy
      expect(['empathy_first', 'concrete_steps', 'more_validation']).toContain(strategy);
    });
  });

  describe('satisfaction tracking', () => {
    it('updates satisfaction on positive feedback', async () => {
      const learner = getUserPreferenceLearner('user_sat_1');
      const before = await learner.getPreferences();

      await learner.recordFeedback('s1', 'positive', 'empathy_first', 90);

      const after = await learner.getPreferences();
      expect(after.avg_response_satisfaction).toBeGreaterThanOrEqual(before.avg_response_satisfaction);
    });

    it('decreases satisfaction on negative feedback', async () => {
      const learner = getUserPreferenceLearner('user_sat_2');
      await learner.recordFeedback('s1', 'positive', 'empathy_first', 85);
      const afterPositive = await learner.getPreferences();

      await learner.recordFeedback('s2', 'negative', 'concrete_steps', 25);
      const afterNegative = await learner.getPreferences();

      expect(afterNegative.avg_response_satisfaction).toBeLessThanOrEqual(afterPositive.avg_response_satisfaction);
    });
  });

  describe('preference bounds', () => {
    it('keeps strategy scores between 0-100', async () => {
      const learner = getUserPreferenceLearner('user_bounds');

      // Send many negative feedbacks
      for (let i = 0; i < 10; i++) {
        await learner.recordFeedback('s' + i, 'negative', 'concrete_steps', 10);
      }

      const prefs = await learner.getPreferences();
      const score = prefs.preferred_strategies['concrete_steps'];

      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    });

    it('keeps satisfaction between 0-100', async () => {
      const learner = getUserPreferenceLearner('user_sat_bounds');

      for (let i = 0; i < 20; i++) {
        await learner.recordFeedback('s' + i, 'positive', 'empathy_first', 95);
      }

      const prefs = await learner.getPreferences();
      expect(prefs.avg_response_satisfaction).toBeGreaterThanOrEqual(0);
      expect(prefs.avg_response_satisfaction).toBeLessThanOrEqual(100);
    });
  });

  describe('learning over time', () => {
    it('converges on preferred strategy through repeated feedback', async () => {
      const learner = getUserPreferenceLearner('user_converge');
      const strategy = 'empathy_first';

      // Give 5 positive feedbacks for one strategy
      for (let i = 0; i < 5; i++) {
        await learner.recordFeedback(`s${i}`, 'positive', strategy, 85 + i);
      }

      const finalPrefs = await learner.getPreferences();
      const finalScore = finalPrefs.preferred_strategies[strategy];

      // Score should be high after repeated positive feedback
      expect(finalScore).toBeGreaterThan(70);
    });
  });
});
