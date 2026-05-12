-- ============================================================
-- 004_agent_memory.sql
-- Long-term memory store for the LangGraph Agent layer
-- Run: psql $DATABASE_URL -f sql/004_agent_memory.sql
-- ============================================================

-- Enable pgvector if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ── agent_memories ────────────────────────────────────────────────────────────
-- Stores distilled facts extracted from past conversations.
-- Each row = one atomic memory unit (a key fact, user preference, or conclusion).
-- Retrieved at query time via pgvector cosine similarity.
CREATE TABLE IF NOT EXISTS agent_memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id       TEXT NOT NULL,          -- conversation thread that produced this memory
    user_id         TEXT NOT NULL DEFAULT 'default',
    content         TEXT NOT NULL,          -- the actual memory text
    embedding       vector(1024),           -- BAAI/bge-m3 embedding of content
    importance      FLOAT NOT NULL DEFAULT 0.5,  -- 0.0–1.0, higher = retrieved first
    source_turn     INT,                    -- which turn of the conversation (for debugging)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed   TIMESTAMPTZ,
    access_count    INT NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- Cosine similarity index for fast semantic recall
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx
    ON agent_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- Fast lookup by user / thread
CREATE INDEX IF NOT EXISTS agent_memories_user_idx    ON agent_memories (user_id);
CREATE INDEX IF NOT EXISTS agent_memories_thread_idx  ON agent_memories (thread_id);
CREATE INDEX IF NOT EXISTS agent_memories_created_idx ON agent_memories (created_at DESC);

-- ── conversation_threads ───────────────────────────────────────────────────────
-- Lightweight thread registry.  LangGraph's MemorySaver handles per-turn state;
-- this table tracks thread-level metadata (intent history, turn count, etc.)
CREATE TABLE IF NOT EXISTS conversation_threads (
    thread_id       TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default',
    title           TEXT,                   -- auto-generated from first question
    intent_history  TEXT[] NOT NULL DEFAULT '{}',
    turn_count      INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS conv_threads_user_idx ON conversation_threads (user_id);
