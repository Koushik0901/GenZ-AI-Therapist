import { NextResponse } from 'next/server';
import { getMonitoringService } from '@/lib/monitoring';

/**
 * Alerts API Endpoint
 * GET /api/alerts
 * Returns active system alerts
 */

export async function GET() {
  try {
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
