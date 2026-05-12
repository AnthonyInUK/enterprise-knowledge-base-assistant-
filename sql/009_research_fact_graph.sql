CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS research_facts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company text NOT NULL,
    metric text NOT NULL,
    value text NOT NULL,
    unit text,
    period text,
    source_citation text NOT NULL,
    source_chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
    confidence numeric(5,4) NOT NULL DEFAULT 0.8000,
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected', 'needs_revision')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    reviewed_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_research_facts_unique_fact
    ON research_facts (company, metric, COALESCE(period, ''), value, source_citation);

CREATE INDEX IF NOT EXISTS idx_research_facts_company_metric
    ON research_facts (company, metric);

CREATE INDEX IF NOT EXISTS idx_research_facts_review_status
    ON research_facts (review_status, created_at DESC);
