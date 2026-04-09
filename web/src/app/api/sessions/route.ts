import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { logger } from '@/lib/logging';

/**
 * Sessions API Endpoint
 * GET /api/sessions?user_id=user_123&limit=10
 * Returns paginated list of user's session history
 */

const QuerySchema = z.object({
  user_id: z.string(),
  limit: z.coerce.number().max(100).default(10),
  offset: z.coerce.number().default(0),
});

type QueryParams = z.infer<typeof QuerySchema>;

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;

    // Parse and validate query parameters
    const parsed = QuerySchema.parse({
      user_id: searchParams.get('user_id'),
      limit: searchParams.get('limit') || 10,
      offset: searchParams.get('offset') || 0,
    });

    // For now, return mock data (implementation would query Supabase)
    // In production, this would fetch from the database with user_id, order by created_at DESC
    logger.debug(
      {
        user_id: parsed.user_id,
        limit: parsed.limit,
        offset: parsed.offset,
      },
      'Fetching user sessions'
    );

    return NextResponse.json(
      {
        success: true,
        sessions: [
          {
            id: 'session-001',
            user_id: parsed.user_id,
            title: 'Feeling overwhelmed with work',
            created_at: new Date().toISOString(),
            message_count: 12,
            quality_score: 78,
            is_crisis: false,
          },
        ],
        count: 1,
        total: 1,
        limit: parsed.limit,
        offset: parsed.offset,
      },
      { status: 200 }
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        {
          success: false,
          message: 'Invalid query parameters',
          errors: error.issues,
        },
        { status: 400 }
      );
    }

    logger.error(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      'Error fetching sessions'
    );

    return NextResponse.json(
      {
        success: false,
        message: 'Failed to fetch sessions',
      },
      { status: 500 }
    );
  }
}
