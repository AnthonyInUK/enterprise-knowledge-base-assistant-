CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS available_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS locked_at timestamptz,
    ADD COLUMN IF NOT EXISTS locked_by text,
    ADD COLUMN IF NOT EXISTS max_retries int NOT NULL DEFAULT 3,
    ADD COLUMN IF NOT EXISTS result jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_queue_claim
    ON ingestion_jobs (status, available_at, created_at)
    WHERE status IN ('queued', 'retry');

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_created_at
    ON ingestion_jobs (status, created_at DESC);
