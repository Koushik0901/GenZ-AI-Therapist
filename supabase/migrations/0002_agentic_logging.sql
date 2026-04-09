-- Agentic Logging Tables (Phase 0)

-- Tool calls logging
CREATE TABLE IF NOT EXISTS public.tool_calls (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tool_name text NOT NULL,
  input jsonb NOT NULL,
  output jsonb,
  duration_ms integer,
  success boolean NOT NULL DEFAULT false,
  error text,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tool_calls_tool_name_created 
  ON public.tool_calls(tool_name, created_at DESC);

CREATE INDEX IF NOT EXISTS tool_calls_success_created 
  ON public.tool_calls(success, created_at DESC);

-- Orchestrator decisions logging
CREATE TABLE IF NOT EXISTS public.orchestrator_decisions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message text NOT NULL,
  decision_path text NOT NULL,
  tools_called text[] NOT NULL,
  reasoning text,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS orchestrator_decisions_decision_path_created 
  ON public.orchestrator_decisions(decision_path, created_at DESC);

-- Response evaluations logging
CREATE TABLE IF NOT EXISTS public.response_evaluations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  response_id uuid NOT NULL,
  warmth integer CHECK (warmth >= 0 AND warmth <= 100),
  validation integer CHECK (validation >= 0 AND validation <= 100),
  clarity integer CHECK (clarity >= 0 AND clarity <= 100),
  actionability integer CHECK (actionability >= 0 AND actionability <= 100),
  overall integer CHECK (overall >= 0 AND overall <= 100) NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS response_evaluations_overall_created 
  ON public.response_evaluations(overall, created_at DESC);

CREATE INDEX IF NOT EXISTS response_evaluations_response_id 
  ON public.response_evaluations(response_id);

-- Crisis evaluations logging
CREATE TABLE IF NOT EXISTS public.crisis_evaluations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  severity integer CHECK (severity >= 0 AND severity <= 100) NOT NULL,
  confidence integer CHECK (confidence >= 0 AND confidence <= 100) NOT NULL,
  factors text[] NOT NULL,
  escalation_level text NOT NULL CHECK (escalation_level IN ('immediate', 'escalate', 'moderate', 'concern', 'none')),
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS crisis_evaluations_severity_created 
  ON public.crisis_evaluations(severity, created_at DESC);

CREATE INDEX IF NOT EXISTS crisis_evaluations_escalation_created 
  ON public.crisis_evaluations(escalation_level, created_at DESC);

-- User feedback
CREATE TABLE IF NOT EXISTS public.feedback (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users ON DELETE CASCADE,
  session_id uuid REFERENCES public.chat_sessions ON DELETE CASCADE,
  response_id uuid NOT NULL,
  helpful boolean NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_session_user_created 
  ON public.feedback(session_id, user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS feedback_helpful_created 
  ON public.feedback(helpful, created_at DESC);

-- RLS Policies for logging tables (read-only, no public access)
ALTER TABLE public.tool_calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.orchestrator_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.response_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.crisis_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;

-- Tool calls: only service role can write, nobody can read from client
CREATE POLICY "tool_calls_insert_service" ON public.tool_calls
  FOR INSERT WITH CHECK (auth.jwt() ->> 'role' = 'service_role');

-- Orchestrator decisions: only service role can write
CREATE POLICY "orchestrator_insert_service" ON public.orchestrator_decisions
  FOR INSERT WITH CHECK (auth.jwt() ->> 'role' = 'service_role');

-- Response evaluations: only service role can write
CREATE POLICY "response_eval_insert_service" ON public.response_evaluations
  FOR INSERT WITH CHECK (auth.jwt() ->> 'role' = 'service_role');

-- Crisis evaluations: only service role can write
CREATE POLICY "crisis_eval_insert_service" ON public.crisis_evaluations
  FOR INSERT WITH CHECK (auth.jwt() ->> 'role' = 'service_role');

-- Feedback: users can insert and view their own feedback
CREATE POLICY "feedback_select_own" ON public.feedback
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "feedback_insert_own" ON public.feedback
  FOR INSERT WITH CHECK (auth.uid() = user_id);
