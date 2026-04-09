import { NextRequest, NextResponse } from 'next/server';
import { logger } from './logging';

/**
 * Admin Authentication Middleware
 * Protects admin endpoints with token-based authentication
 *
 * Environment Variable:
 * ADMIN_AUTH_TOKEN - Secret token required to access protected endpoints
 *
 * Usage:
 * const response = await requireAdminAuth(request);
 * if (!response.isAuthorized) return response.error;
 * // Continue with endpoint logic
 */

export interface AuthResult {
  isAuthorized: boolean;
  error?: NextResponse;
  adminId?: string;
}

/**
 * Validates admin authentication from request headers
 * Supports two methods:
 * 1. Authorization: Bearer <token> header
 * 2. x-admin-token header
 */
export async function requireAdminAuth(request: NextRequest): Promise<AuthResult> {
  try {
    const authToken = getAuthToken(request);
    const expectedToken = process.env.ADMIN_AUTH_TOKEN;

    // If no token configured, deny all admin requests
    if (!expectedToken) {
      logger.warn({
        message: 'Admin endpoint called without ADMIN_AUTH_TOKEN configured',
        url: request.url,
      });

      return {
        isAuthorized: false,
        error: NextResponse.json(
          {
            success: false,
            message: 'Admin authentication not configured',
          },
          { status: 503 }
        ),
      };
    }

    // If no auth header provided, deny
    if (!authToken) {
      logger.warn({
        message: 'Admin endpoint called without authentication',
        url: request.url,
      });

      return {
        isAuthorized: false,
        error: NextResponse.json(
          {
            success: false,
            message: 'Authentication required',
          },
          { status: 401 }
        ),
      };
    }

    // Validate token (constant-time comparison to prevent timing attacks)
    const isValid = constantTimeCompare(authToken, expectedToken);

    if (!isValid) {
      logger.warn({
        message: 'Admin endpoint called with invalid authentication token',
        url: request.url,
      });

      return {
        isAuthorized: false,
        error: NextResponse.json(
          {
            success: false,
            message: 'Invalid authentication token',
          },
          { status: 403 }
        ),
      };
    }

    // Extract admin ID from token if present (format: token:admin_id)
    const parts = authToken.split(':');
    const adminId = parts.length > 1 ? parts[1] : 'system';

    logger.info({
      message: 'Admin endpoint accessed successfully',
      url: request.url,
      adminId,
    });

    return {
      isAuthorized: true,
      adminId,
    };
  } catch (error) {
    logger.error({
      message: 'Error in admin authentication',
      error: error instanceof Error ? error.message : String(error),
      url: request.url,
    });

    return {
      isAuthorized: false,
      error: NextResponse.json(
        {
          success: false,
          message: 'Authentication error',
        },
        { status: 500 }
      ),
    };
  }
}

/**
 * Extract authentication token from request headers
 * Tries multiple header formats for flexibility
 */
function getAuthToken(request: NextRequest): string | null {
  // Try Authorization header first (Bearer token)
  const authHeader = request.headers.get('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    return authHeader.slice(7);
  }

  // Fall back to custom x-admin-token header
  return request.headers.get('x-admin-token');
}

/**
 * Constant-time string comparison to prevent timing attacks
 * Returns true if strings are equal, false otherwise
 */
function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}
