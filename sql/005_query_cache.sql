-- ============================================================
-- 005_query_cache.sql
-- Semantic query cache: pgvector similarity search over past queries.
-- Queries with cosine similarity >= threshold skip the full RAG pipeline.
-- Run: psql $DATABASE_URL -f sql/005_query_cache.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS query_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT NOT NULL,
    query_embedding vector(1024),           -- BAAI/bge-m3 embedding
    answer          TEXT NOT NULL,
    intent          TEXT,
    sources         TEXT[],
    react_steps     JSONB NOT NULL DEFAULT '[]',
    latency_ms      FLOAT,                  -- original pipeline latency (for display)
    hit_count       INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_hit_at     TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days')
);

-- Fast ANN index for cosine similarity lookups
CREATE INDEX IF NOT EXISTS query_cache_embedding_idx
    ON query_cache
    USING ivfflat (query_embedding vector_cosine_ops)
    WITH (lists = 20);

-- Exact text dedup (avoid storing identical queries twice)
CREATE UNIQUE INDEX IF NOT EXISTS query_cache_text_idx
    ON query_cache (query_text);
