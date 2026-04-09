import { NextRequest, NextResponse } from 'next/server';
import { getMonitoringService } from '@/lib/monitoring';
import { requireAdminAuth } from '@/lib/auth-middleware';

/**
 * Alerts API Endpoint
 * GET /api/alerts
 * Returns active system alerts
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
    const alerts = await monitoring.getActiveAlerts();

    return NextResponse.json(
      {
        success: true,
        alerts,
        count: alerts.length,
      },
      { status: 200 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: 'Failed to fetch alerts',
        alerts: [],
      },
      { status: 500 }
    );
  }
}
