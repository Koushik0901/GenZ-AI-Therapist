import { NextResponse } from 'next/server';
import { getMonitoringService } from '@/lib/monitoring';

/**
 * Metrics API Endpoint
 * GET /api/metrics
 * Returns current system metrics
 */

export async function GET() {
  try {
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
