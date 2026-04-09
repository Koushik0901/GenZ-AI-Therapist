import { runOrchestrator, OrchestratorOutput } from '@/lib/orchestrator';
import { SessionManager } from '@/lib/session-storage';
import { getUserPreferenceLearner } from '@/lib/user-preferences';
import { getABTestManager } from '@/lib/ab-testing';
import { getMonitoringService } from '@/lib/monitoring';
import { logger } from '@/lib/logging';

/**
 * Integrated Orchestrator
 * Orchestrator + Session Storage + User Preferences + A/B Testing + Monitoring
 * This is the complete system integration layer
 */

export interface IntegratedOrchestratorConfig {
  userId?: string;
  sessionId?: string;
  enableABTesting?: boolean;
  enableMetrics?: boolean;
}

export async function runIntegratedOrchestrator(
  userMessage: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  config: IntegratedOrchestratorConfig = {}
) {
  const startTime = Date.now();
  const sessionId = config.sessionId || `session_${Date.now()}`;
  const userId = config.userId;
  const enableABTesting = config.enableABTesting !== false;
  const enableMetrics = config.enableMetrics !== false;

  // Initialize services
  const sessionManager = new SessionManager(sessionId, userId);
  const monitoring = getMonitoringService();

  try {
    // Initialize session
    await sessionManager.initialize();

    // Get user preferences if available
    let userPrefs = null;
    if (userId) {
      const preferenceLearner = getUserPreferenceLearner(userId);
      userPrefs = await preferenceLearner.getPreferences();
    }

    // Select A/B test variant if enabled
    let abTestVariant = null;
    if (enableABTesting) {
      const abTestManager = getABTestManager();
      abTestVariant = await abTestManager.selectVariantForUser(sessionId, 'strategy_selection');
    }

    // Run core orchestrator
    logger.debug(
      { session_id: sessionId, user_id: userId },
      'Running integrated orchestrator'
    );

    const orchestratorResult = await runOrchestrator({
      userMessage,
      history,
      maxAttempts: 3,
    });

    const responseTime = Date.now() - startTime;

    // Add message to session
    await sessionManager.addMessage({
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
      // Note: classification and sessionType are available in orchestratorResult
      // but not typed for SessionMessage.metadata due to complex type requirements
    });

    // Record metrics
    if (enableMetrics) {
      monitoring.recordResponseMetrics(
        orchestratorResult.metadata.confidence_score,
        responseTime,
        true
      );

      if (orchestratorResult.metadata.crisis_severity === 'critical' || orchestratorResult.metadata.crisis_severity === 'high_risk') {
        monitoring.recordCrisisDetection(orchestratorResult.metadata.crisis_severity, sessionId);
        await sessionManager.markCrisisDetected(orchestratorResult.metadata.crisis_severity);
      }

      if (orchestratorResult.metadata.regeneration_attempts > 0) {
        monitoring.recordRegeneration(
          sessionId,
          `Regenerated ${orchestratorResult.metadata.regeneration_attempts} times`
        );
      }
    }

    // Update session quality metrics
    await sessionManager.updateQuality(
      orchestratorResult.metadata.confidence_score,
      orchestratorResult.metadata.classification.confidence
    );

    // Log A/B test result if applicable
    if (abTestVariant && enableABTesting) {
      const abTestManager = getABTestManager();
      await abTestManager.recordResult({
        test_id: `test_${Date.now()}`,
        variant_id: abTestVariant.id,
        session_id: sessionId,
        response_quality: orchestratorResult.metadata.confidence_score,
        user_feedback: null, // Will be updated when user provides feedback
        metrics: {
          response_time_ms: responseTime,
          regeneration_attempts: orchestratorResult.metadata.regeneration_attempts,
        },
      });
    }

    // Add assistant response to session
    await sessionManager.addMessage({
      role: 'assistant',
      content: orchestratorResult.response,
      timestamp: new Date().toISOString(),
      metadata: {
        responseQuality: orchestratorResult.metadata.confidence_score,
      },
    });

    logger.info(
      {
        session_id: sessionId,
        response_time_ms: responseTime,
        quality: orchestratorResult.metadata.confidence_score,
      },
      'Integrated orchestrator completed'
    );

    return {
      ...orchestratorResult,
      session_id: sessionId,
      response_time_ms: responseTime,
      ab_test_variant: abTestVariant?.name || null,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;

    logger.error(
      {
        session_id: sessionId,
        error: error instanceof Error ? error.message : String(error),
      },
      'Integrated orchestrator error'
    );

    // Record error metric
    if (enableMetrics) {
      monitoring.recordAPIError('orchestrator', error instanceof Error ? error.message : String(error), sessionId);
    }

    // Create alert
    await sessionManager.createAlert(
      'api_error',
      'warning',
      'Orchestrator error occurred',
      { error_message: error instanceof Error ? error.message : String(error) }
    );

    throw error;
  }
}

/**
 * Process user feedback and update learning systems
 */
export async function processFeedback(
  sessionId: string,
  userId: string,
  responseId: string,
  sentiment: 'positive' | 'negative',
  comment?: string
) {
  try {
    const preferenceLearner = getUserPreferenceLearner(userId);
    const monitoring = getMonitoringService();

    // Record user satisfaction
    monitoring.recordUserSatisfaction(sentiment === 'positive');

    // Learn from feedback
    await preferenceLearner.recordFeedback(
      sessionId,
      sentiment,
      'empathy_first', // This would be the actual strategy used (simplified)
      sentiment === 'positive' ? 80 : 40,
      comment
    );

    logger.info(
      {
        session_id: sessionId,
        user_id: userId,
        sentiment,
        has_comment: Boolean(comment),
      },
      'User feedback processed'
    );
  } catch (error) {
    logger.error(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      'Error processing feedback'
    );
  }
}

/**
 * Get session summary
 */
export async function getSessionSummary(sessionId: string) {
  try {
    const monitoring = getMonitoringService();
    const health = monitoring.getHealthStatus();
    const alerts = await monitoring.getActiveAlerts();

    return {
      session_id: sessionId,
      system_health: health,
      active_alerts: alerts.length,
      metrics: monitoring.getPerformanceReport(),
    };
  } catch (error) {
    logger.error(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      'Error getting session summary'
    );

    return null;
  }
}
