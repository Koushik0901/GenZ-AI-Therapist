import { NextRequest, NextResponse } from 'next/server';
import { getMonitoringService } from '@/lib/monitoring';
import { requireAdminAuth } from '@/lib/auth-middleware';

/**
 * Metrics API Endpoint
 * GET /api/metrics
 * Returns current system metrics
 *
 * PROTECTED: Requires admin authentication via Authorization header or x-admin-token
 * Headers:
 *   Authorization: Bearer <admin_token>
 *   OR
 *   x-admin-token: <admin_token>
 */

export async function GET(request: NextRequest) {
  try {
    // Validate admin authentication
    const auth = await requireAdminAuth(request);
    if (!auth.isAuthorized) {
      return auth.error!;
    }

    const monitoring = getMonitoringService();
    const metrics = monitoring.getPerformanceReport();

    return NextResponse.json(
      {
        success: true,
        metrics,
      },
      { status: 200 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: 'Failed to fetch metrics',
      },
      { status: 500 }
    );
  }
}
