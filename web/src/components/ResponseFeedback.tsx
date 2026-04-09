'use client';

import { useState, useEffect } from 'react';

/**
 * ResponseFeedback Component
 * Simple thumbs up/down widget for user feedback on responses
 * Shows optional comment field for negative feedback
 * Persists feedback state to prevent duplicate submissions
 */

interface ResponseFeedbackProps {
  responseId: string;
  sessionId: string;
  onFeedbackSubmitted?: (feedback: FeedbackData) => void;
}

export interface FeedbackData {
  response_id: string;
  session_id: string;
  sentiment: 'positive' | 'negative';
  comment?: string;
  timestamp: string;
}

const STORAGE_KEY_PREFIX = 'feedback_';

export default function ResponseFeedback({
  responseId,
  sessionId,
  onFeedbackSubmitted,
}: ResponseFeedbackProps) {
  const [sentiment, setSentiment] = useState<'positive' | 'negative' | null>(null);
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load persisted feedback on mount
  useEffect(() => {
    const storageKey = `${STORAGE_KEY_PREFIX}${responseId}`;
    const persistedFeedback = localStorage.getItem(storageKey);
    
    if (persistedFeedback) {
      try {
        const parsed = JSON.parse(persistedFeedback);
        setSubmitted(true);
        setSentiment(parsed.sentiment);
        setComment(parsed.comment || '');
        
        // Hide widget after 2 seconds if feedback already submitted
        setTimeout(() => {
          setSubmitted(false);
        }, 2000);
      } catch (err) {
        // Ignore parse errors
      }
    }
  }, [responseId]);

  const handleFeedback = async (newSentiment: 'positive' | 'negative') => {
    setSentiment(newSentiment);
    setError(null);

    // Only show comment field for negative feedback
    if (newSentiment === 'positive') {
      await submitFeedback(newSentiment, '');
    }
  };

  const submitFeedback = async (feedbackSentiment: 'positive' | 'negative', feedbackComment: string) => {
    setIsSubmitting(true);

    try {
      const feedbackData: FeedbackData = {
        response_id: responseId,
        session_id: sessionId,
        sentiment: feedbackSentiment,
        comment: feedbackComment || undefined,
        timestamp: new Date().toISOString(),
      };

      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      // Persist feedback to local storage to prevent duplicate submissions
      const storageKey = `${STORAGE_KEY_PREFIX}${responseId}`;
      localStorage.setItem(storageKey, JSON.stringify(feedbackData));

      setSubmitted(true);
      setSentiment(null);
      setComment('');

      if (onFeedbackSubmitted) {
        onFeedbackSubmitted(feedbackData);
      }

      // Hide widget after 2 seconds
      setTimeout(() => {
        setSubmitted(false);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
        <p className="text-sm text-green-700">Thanks for your feedback! 💚</p>
      </div>
    );
  }

  return (
    <div className="mt-4 p-3 bg-neutral-50 border border-neutral-200 rounded-lg">
      <p className="text-xs text-neutral-600 mb-2">Was this helpful?</p>

      <div className="flex gap-2 items-center">
        {/* Thumbs Up Button */}
        <button
          onClick={() => handleFeedback('positive')}
          disabled={isSubmitting}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            sentiment === 'positive'
              ? 'bg-green-100 text-green-700 border border-green-300'
              : 'bg-white text-neutral-600 border border-neutral-200 hover:bg-neutral-100'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          aria-label="This was helpful"
        >
          👍 Yes
        </button>

        {/* Thumbs Down Button */}
        <button
          onClick={() => handleFeedback('negative')}
          disabled={isSubmitting}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            sentiment === 'negative'
              ? 'bg-red-100 text-red-700 border border-red-300'
              : 'bg-white text-neutral-600 border border-neutral-200 hover:bg-neutral-100'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          aria-label="This was not helpful"
        >
          👎 No
        </button>
      </div>

      {/* Comment Field (shown for negative feedback) */}
      {sentiment === 'negative' && (
        <div className="mt-3 pt-3 border-t border-neutral-200">
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What could have been better? (optional)"
            className="w-full p-2 text-sm border border-neutral-200 rounded resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={2}
            disabled={isSubmitting}
          />

          <div className="mt-2 flex gap-2">
            <button
              onClick={() => submitFeedback('negative', comment)}
              disabled={isSubmitting}
              className="px-3 py-1.5 bg-blue-500 text-white text-sm rounded font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>

            <button
              onClick={() => {
                setSentiment(null);
                setComment('');
              }}
              disabled={isSubmitting}
              className="px-3 py-1.5 text-neutral-600 text-sm rounded font-medium border border-neutral-200 hover:bg-neutral-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  );
}
