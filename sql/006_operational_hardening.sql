-- ============================================================
-- 006_operational_hardening.sql
-- Production-readiness tables for API observability, review,
-- and agent tool governance.
-- Run: psql $DATABASE_URL -f sql/006_operational_hardening.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- API request tracing and latency/error dashboard source.
CREATE TABLE IF NOT EXISTS api_request_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id text NOT NULL UNIQUE,
    method text NOT NULL,
    path text NOT NULL,
    status_code int,
    latency_ms int,
    user_id text,
    client_host text,
    error_message text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_api_request_logs_created_at
    ON api_request_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_request_logs_path_created_at
    ON api_request_logs (path, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_request_logs_status_created_at
    ON api_request_logs (status_code, created_at DESC);

-- Human-in-the-loop review for important facts in generated answers.
CREATE TABLE IF NOT EXISTS answer_reviews (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_log_id uuid REFERENCES query_logs(id) ON DELETE SET NULL,
    reviewer_id text,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'needs_revision')),
    reviewed_fact text,
    corrected_fact text,
    source_citation text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    reviewed_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_answer_reviews_status_created_at
    ON answer_reviews (status, created_at DESC);

-- Agent/tool governance trace. Each row is one tool call attempt.
CREATE TABLE IF NOT EXISTS agent_tool_traces (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id text,
    thread_id text,
    user_id text,
    tool_name text NOT NULL,
    tool_input jsonb NOT NULL DEFAULT '{}'::jsonb,
    tool_output jsonb,
    status text NOT NULL DEFAULT 'started'
        CHECK (status IN ('started', 'success', 'error', 'timeout', 'blocked')),
    latency_ms int,
    error_message text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_tool_traces_request_id
    ON agent_tool_traces (request_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_traces_tool_created_at
    ON agent_tool_traces (tool_name, created_at DESC);
