import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';

/**
 * Feedback API Endpoint
 * POST /api/feedback
 * Stores user feedback (thumbs up/down) and optional comments
 */

const FeedbackSchema = z.object({
  response_id: z.string(),
  session_id: z.string(),
  sentiment: z.enum(['positive', 'negative']),
  comment: z.string().optional(),
  timestamp: z.string().datetime(),
});

type FeedbackInput = z.infer<typeof FeedbackSchema>;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate input
    const feedback = FeedbackSchema.parse(body);

    // Store in Supabase if configured
    if (isSupabaseConfigured) {
      const supabase = createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey);

      const { error } = await supabase.from('feedback').insert([
        {
          response_id: feedback.response_id,
          session_id: feedback.session_id,
          sentiment: feedback.sentiment,
          comment: feedback.comment || null,
          created_at: feedback.timestamp,
        },
      ]);

      if (error) {
        logger.error(
          {
            error: error.message,
            code: error.code,
          },
          'Failed to store feedback in Supabase'
        );

        // Don't fail the request if Supabase write fails
        // Log it but return success to user
      } else {
        logger.debug(
          {
            response_id: feedback.response_id,
            sentiment: feedback.sentiment,
          },
          'Feedback stored successfully'
        );
      }
    }

    // Log feedback locally
    logger.info(
      {
        response_id: feedback.response_id,
        session_id: feedback.session_id,
        sentiment: feedback.sentiment,
        has_comment: Boolean(feedback.comment),
      },
      'User feedback received'
    );

    return NextResponse.json(
      {
        success: true,
        message: 'Feedback received, thank you!',
      },
      { status: 200 }
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      logger.warn(
        {
          errors: error.issues,
        },
        'Invalid feedback input'
      );

      return NextResponse.json(
        {
          success: false,
          message: 'Invalid feedback data',
          errors: error.issues,
        },
        { status: 400 }
      );
    }

    logger.error(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      'Feedback API error'
    );

    return NextResponse.json(
      {
        success: false,
        message: 'Internal server error',
      },
      { status: 500 }
    );
  }
}
