-- Feedback System Migration (Phase 3+)
-- Creates additional tables for advanced analytics
-- Note: Basic feedback table is created in 0002_agentic_logging.sql

-- Feedback Summary: aggregated metrics (NEW)
CREATE TABLE IF NOT EXISTS public.feedback_summary (
  id BIGSERIAL PRIMARY KEY,
  response_id TEXT NOT NULL,
  positive_count BIGINT DEFAULT 0,
  negative_count BIGINT DEFAULT 0,
  total_count BIGINT DEFAULT 0,
  positive_ratio NUMERIC(5, 2) DEFAULT 0.00,
  last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(response_id)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_feedback_summary_response_id ON public.feedback_summary(response_id);
CREATE INDEX IF NOT EXISTS idx_feedback_summary_ratio ON public.feedback_summary(positive_ratio DESC);

-- Session tracking: multi-turn conversation metadata
CREATE TABLE IF NOT EXISTS public.session_metadata (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  user_id TEXT,
  start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  end_time TIMESTAMP WITH TIME ZONE,
  message_count BIGINT DEFAULT 0,
  avg_classification_confidence NUMERIC(5, 2),
  avg_response_quality NUMERIC(5, 2),
  crisis_detected BOOLEAN DEFAULT FALSE,
  session_type TEXT,
  overall_satisfaction TEXT,
  notes TEXT,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_session_metadata_session_id ON public.session_metadata(session_id);
CREATE INDEX IF NOT EXISTS idx_session_metadata_user_id ON public.session_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_session_metadata_start_time ON public.session_metadata(start_time DESC);

-- Strategy Performance: tracks which strategies work best
CREATE TABLE IF NOT EXISTS public.strategy_performance (
  id BIGSERIAL PRIMARY KEY,
  strategy_name TEXT NOT NULL,
  session_type TEXT,
  attempt_number INT DEFAULT 1,
  quality_before NUMERIC(5, 2),
  quality_after NUMERIC(5, 2),
  user_feedback TEXT,
  used_count BIGINT DEFAULT 1,
  success_count BIGINT DEFAULT 0,
  avg_quality_improvement NUMERIC(5, 2),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(strategy_name, session_type, attempt_number)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy ON public.strategy_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_session_type ON public.strategy_performance(session_type);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_success_rate ON public.strategy_performance(success_count DESC);

-- A/B Test Variants: track different strategy combinations
CREATE TABLE IF NOT EXISTS public.ab_test_variants (
  id BIGSERIAL PRIMARY KEY,
  variant_name TEXT NOT NULL UNIQUE,
  variant_type TEXT NOT NULL CHECK (variant_type IN ('strategy_selection', 'resource_search', 'response_eval')),
  description TEXT,
  config JSONB NOT NULL,
  active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for active variants
CREATE INDEX IF NOT EXISTS idx_ab_test_variants_active ON public.ab_test_variants(active);
CREATE INDEX IF NOT EXISTS idx_ab_test_variants_type ON public.ab_test_variants(variant_type);

-- A/B Test Results: track performance of variants
CREATE TABLE IF NOT EXISTS public.ab_test_results (
  id BIGSERIAL PRIMARY KEY,
  test_id TEXT NOT NULL,
  variant_id BIGINT REFERENCES public.ab_test_variants(id),
  session_id TEXT NOT NULL,
  response_quality NUMERIC(5, 2),
  user_feedback TEXT,
  metrics JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_ab_test_results_test_id ON public.ab_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_variant_id ON public.ab_test_results(variant_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_created_at ON public.ab_test_results(created_at DESC);

-- Alerts: for critical patterns or issues
CREATE TABLE IF NOT EXISTS public.system_alerts (
  id BIGSERIAL PRIMARY KEY,
  alert_type TEXT NOT NULL CHECK (alert_type IN ('crisis_escalation', 'quality_decline', 'api_error', 'pattern_detected', 'user_support_needed')),
  severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
  session_id TEXT,
  message TEXT NOT NULL,
  details JSONB,
  acknowledged BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  acknowledged_at TIMESTAMP WITH TIME ZONE,
  acknowledged_by TEXT
);

-- Index for active alerts
CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON public.system_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_system_alerts_acknowledged ON public.system_alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_system_alerts_created_at ON public.system_alerts(created_at DESC);

-- Grant permissions for anon access (if needed)
GRANT SELECT ON public.feedback_summary TO anon;
GRANT SELECT, INSERT, UPDATE ON public.session_metadata TO anon;
GRANT SELECT ON public.strategy_performance TO anon;
GRANT SELECT ON public.ab_test_variants TO anon;
GRANT SELECT, INSERT ON public.ab_test_results TO anon;
GRANT SELECT ON public.system_alerts TO anon;
