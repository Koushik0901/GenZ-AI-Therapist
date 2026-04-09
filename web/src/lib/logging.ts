import pino from 'pino';

// Logger instance configuration
const isDevelopment = process.env.NODE_ENV === 'development';

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: isDevelopment
    ? {
        target: 'pino-pretty',
        options: {
          colorize: true,
          translateTime: 'SYS:standard',
          ignore: 'pid,hostname',
          singleLine: false,
        },
      }
    : undefined,
});

/**
 * Log levels:
 * - trace: Very detailed debugging (rarely used)
 * - debug: Debugging information (development)
 * - info: General informational messages (normal)
 * - warn: Warning messages (potential issues)
 * - error: Error messages (problems that occurred)
 * - fatal: Fatal errors (unrecoverable)
 */

// Tool call logging
export interface ToolCallLog {
  tool_name: string;
  input: Record<string, any>;
  output?: Record<string, any>;
  duration_ms: number;
  success: boolean;
  error?: string;
  timestamp: string;
}

export function logToolCall(data: Omit<ToolCallLog, 'timestamp'>): void {
  logger.info(
    {
      type: 'tool_call',
      tool: data.tool_name,
      duration_ms: data.duration_ms,
      success: data.success,
      error: data.error,
    },
    `Tool call: ${data.tool_name} (${data.duration_ms}ms)`
  );
}

// Orchestrator decision logging
export interface OrchestratorDecisionLog {
  message: string;
  decision_path: string;
  tools_called: string[];
  reasoning: string;
  timestamp: string;
}

export function logOrchestratorDecision(data: Omit<OrchestratorDecisionLog, 'timestamp'>): void {
  logger.info(
    {
      type: 'orchestrator_decision',
      decision_path: data.decision_path,
      tools_called: data.tools_called,
    },
    `Orchestrator decision: ${data.decision_path}`
  );
}

// Response evaluation logging
export interface ResponseEvaluationLog {
  response_id: string;
  warmth: number;
  validation: number;
  clarity: number;
  actionability: number;
  overall: number;
  timestamp: string;
}

export function logResponseEvaluation(data: Omit<ResponseEvaluationLog, 'timestamp'>): void {
  logger.info(
    {
      type: 'response_evaluation',
      response_id: data.response_id,
      overall_score: data.overall,
      scores: {
        warmth: data.warmth,
        validation: data.validation,
        clarity: data.clarity,
        actionability: data.actionability,
      },
    },
    `Response evaluation: ${data.overall}/100`
  );
}

// Crisis evaluation logging
export interface CrisisEvaluationLog {
  severity: number;
  confidence: number;
  factors: string[];
  escalation_level: 'immediate' | 'escalate' | 'moderate' | 'concern' | 'none';
  timestamp: string;
}

export function logCrisisEvaluation(data: Omit<CrisisEvaluationLog, 'timestamp'>): void {
  logger.warn(
    {
      type: 'crisis_evaluation',
      severity: data.severity,
      confidence: data.confidence,
      escalation_level: data.escalation_level,
      factors: data.factors,
    },
    `Crisis evaluation: severity=${data.severity}, escalation=${data.escalation_level}`
  );
}

// User feedback logging
export interface FeedbackLog {
  response_id: string;
  session_id: string;
  helpful: boolean;
  timestamp: string;
}

export function logFeedback(data: Omit<FeedbackLog, 'timestamp'>): void {
  logger.info(
    {
      type: 'user_feedback',
      response_id: data.response_id,
      session_id: data.session_id,
      helpful: data.helpful,
    },
    `User feedback: ${data.helpful ? 'helpful' : 'not helpful'}`
  );
}

// Error logging
export function logError(error: Error, context: Record<string, any> = {}): void {
  logger.error(
    {
      type: 'error',
      error: {
        message: error.message,
        stack: error.stack,
      },
      context,
    },
    `Error occurred: ${error.message}`
  );
}

// Warning logging
export function logWarning(message: string, context: Record<string, any> = {}): void {
  logger.warn(
    {
      type: 'warning',
      context,
    },
    message
  );
}

// Debug logging
export function logDebug(message: string, data: Record<string, any> = {}): void {
  logger.debug(
    {
      type: 'debug',
      ...data,
    },
    message
  );
}
