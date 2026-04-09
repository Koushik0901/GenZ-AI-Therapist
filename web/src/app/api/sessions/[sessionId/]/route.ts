import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';

/**
 * Session Detail API Endpoint
 * GET /api/sessions/:sessionId
 * Returns detailed information about a specific session
 */

interface RouteParams {
  sessionId: string;
}

export async function GET(
  request: NextRequest,
  { params }: { params: RouteParams }
) {
  try {
    if (!isSupabaseConfigured) {
      return NextResponse.json(
        {
          success: false,
          message: 'Supabase not configured',
        },
        { status: 503 }
      );
    }

    const { sessionId } = params;

    if (!sessionId) {
      return NextResponse.json(
        {
          success: false,
          message: 'sessionId is required',
        },
        { status: 400 }
      );
    }

    const supabase = createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey);

    // Get session metadata
    const { data: session, error: sessionError } = await supabase
      .from('session_metadata')
      .select('*')
      .eq('session_id', sessionId)
      .single();

    if (sessionError || !session) {
      logger.warn(
        { error: sessionError?.message, session_id: sessionId },
        'Session not found'
      );

      return NextResponse.json(
        {
          success: false,
          message: 'Session not found',
        },
        { status: 404 }
      );
    }

    // Get feedback for this session
    const { data: feedback, error: feedbackError } = await supabase
      .from('feedback')
      .select('sentiment, comment, created_at')
      .eq('session_id', sessionId);

    if (feedbackError) {
      logger.warn(
        { error: feedbackError.message, session_id: sessionId },
        'Failed to fetch session feedback'
      );
    }

    logger.debug(
      { session_id: sessionId },
      'Session details retrieved'
    );

    return NextResponse.json(
      {
        success: true,
        session: {
          ...session,
          feedback: feedback || [],
          feedback_count: feedback?.length || 0,
        },
      },
      { status: 200 }
    );
  } catch (error) {
    logger.error(
      { error: error instanceof Error ? error.message : String(error) },
      'Session detail API error'
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
