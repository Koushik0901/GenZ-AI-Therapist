import { NextRequest, NextResponse } from 'next/server';
import { logger } from '@/lib/logging';

/**
 * Session Details API Endpoint
 * GET /api/sessions/[sessionId]
 * Returns detailed information about a specific session
 *
 * Next.js 16+ requires params to be async
 */

interface Params {
  sessionId: string;
}

export async function GET(
  request: NextRequest,
  context: {
    params: Promise<Params>;
  }
) {
  try {
    const params = await context.params;
    const { sessionId } = params;

    logger.debug(
      {
        session_id: sessionId,
      },
      'Fetching session details'
    );

    // For now, return mock data (implementation would query Supabase)
    return NextResponse.json(
      {
        success: true,
        session: {
          id: sessionId,
          title: 'Sample Session',
          user_id: 'user-123',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          messages: [
            {
              id: 'msg-1',
              content: "I'm feeling stressed",
              role: 'user',
              timestamp: new Date().toISOString(),
            },
            {
              id: 'msg-2',
              content: "I hear you. What's on your mind?",
              role: 'assistant',
              timestamp: new Date().toISOString(),
              quality_score: 85,
              strategy_used: 'empathy_first',
            },
          ],
          feedback: [
            {
              message_id: 'msg-2',
              rating: 'helpful',
              timestamp: new Date().toISOString(),
            },
          ],
          quality_score: 85,
          is_crisis: false,
        },
      },
      { status: 200 }
    );
  } catch (error) {
    logger.error(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      'Error fetching session details'
    );

    return NextResponse.json(
      {
        success: false,
        message: 'Failed to fetch session details',
      },
      { status: 500 }
    );
  }
}
