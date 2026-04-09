import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { appEnv, isSupabaseConfigured } from '@/lib/env';
import { logger } from '@/lib/logging';

/**
 * Session History API Endpoint
 * GET /api/sessions?user_id=user_123&limit=10
 * Returns user's session history
 */

export async function GET(request: NextRequest) {
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

    const searchParams = request.nextUrl.searchParams;
    const userId = searchParams.get('user_id');
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 100); // Max 100

    if (!userId) {
      return NextResponse.json(
        {
          success: false,
          message: 'user_id is required',
        },
        { status: 400 }
      );
    }

    const supabase = createClient(appEnv.supabaseUrl, appEnv.supabaseAnonKey);

    const { data, error } = await supabase
      .from('session_metadata')
      .select(
        'session_id, start_time, end_time, message_count, avg_response_quality, crisis_detected, session_type'
      )
      .eq('user_id', userId)
      .order('start_time', { ascending: false })
      .limit(limit);

    if (error) {
      logger.error(
        { error: error.message, user_id: userId },
        'Failed to fetch session history'
      );

      return NextResponse.json(
        {
          success: false,
          message: 'Failed to fetch sessions',
        },
        { status: 500 }
      );
    }

    logger.debug(
      { user_id: userId, count: data?.length || 0 },
      'Session history retrieved'
    );

    return NextResponse.json(
      {
        success: true,
        user_id: userId,
        sessions: data || [],
        count: data?.length || 0,
      },
      { status: 200 }
    );
  } catch (error) {
    logger.error(
      { error: error instanceof Error ? error.message : String(error) },
      'Session history API error'
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
