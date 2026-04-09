import { NextRequest, NextResponse } from 'next/server';
import { getMonitoringService } from '@/lib/monitoring';
import { requireAdminAuth } from '@/lib/auth-middleware';
import { logger } from '@/lib/logging';
import { z } from 'zod';

/**
 * Alert Acknowledgment API Endpoint
 * POST /api/alerts/acknowledge
 * Acknowledges (marks as resolved) a specific alert
 *
 * PROTECTED: Requires admin authentication via Authorization header or x-admin-token
 *
 * Request Body:
 * {
 *   alertId: string (required)
 *   notes?: string (optional admin notes)
 * }
 *
 * Response:
 * {
 *   success: boolean
 *   message: string
 *   alertId?: string
 * }
 */

const AcknowledgeAlertSchema = z.object({
  alertId: z.string().min(1, 'Alert ID is required'),
  notes: z.string().optional(),
});

type AcknowledgeAlertRequest = z.infer<typeof AcknowledgeAlertSchema>;

export async function POST(request: NextRequest) {
  try {
    // Validate admin authentication
    const auth = await requireAdminAuth(request);
    if (!auth.isAuthorized) {
      return auth.error!;
    }

    // Parse and validate request body
    let body: AcknowledgeAlertRequest;
    try {
      const rawBody = await request.json();
      body = AcknowledgeAlertSchema.parse(rawBody);
    } catch (error) {
      return NextResponse.json(
        {
          success: false,
          message:
            error instanceof z.ZodError
              ? `Validation error: ${error.errors[0].message}`
              : 'Invalid request body',
        },
        { status: 400 }
      );
    }

    const monitoring = getMonitoringService();

    // Acknowledge the alert in monitoring service
    const acknowledged = await monitoring.acknowledgeAlert(body.alertId, {
      acknowledgedBy: auth.adminId || 'system',
      notes: body.notes,
    });

    if (!acknowledged) {
      return NextResponse.json(
        {
          success: false,
          message: 'Alert not found or already acknowledged',
        },
        { status: 404 }
      );
    }

    logger.info({
      message: 'Alert acknowledged',
      alertId: body.alertId,
      adminId: auth.adminId,
      notes: body.notes,
    });

    return NextResponse.json(
      {
        success: true,
        message: 'Alert acknowledged successfully',
        alertId: body.alertId,
      },
      { status: 200 }
    );
  } catch (error) {
    logger.error({
      message: 'Error acknowledging alert',
      error: error instanceof Error ? error.message : String(error),
    });

    return NextResponse.json(
      {
        success: false,
        message: 'Failed to acknowledge alert',
      },
      { status: 500 }
    );
  }
}
