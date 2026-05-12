CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS query_embedding_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash text NOT NULL,
    query_text text NOT NULL,
    embedding_model text NOT NULL,
    embedding_version text NOT NULL DEFAULT 'v1',
    dimension int NOT NULL,
    embedding vector(1024) NOT NULL,
    hit_count int NOT NULL DEFAULT 0,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(query_hash, embedding_model, embedding_version)
);

CREATE INDEX IF NOT EXISTS idx_query_embedding_cache_hash_model
    ON query_embedding_cache(query_hash, embedding_model, embedding_version);

CREATE TABLE IF NOT EXISTS retrieval_result_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key text NOT NULL UNIQUE,
    query_hash text NOT NULL,
    query_text text NOT NULL,
    corpus_fingerprint text NOT NULL,
    strategy_fingerprint text NOT NULL,
    top_k int NOT NULL,
    chunk_ids uuid[] NOT NULL,
    hit_count int NOT NULL DEFAULT 0,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_retrieval_result_cache_query_hash
    ON retrieval_result_cache(query_hash);

CREATE TABLE IF NOT EXISTS answer_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key text NOT NULL UNIQUE,
    query_hash text NOT NULL,
    query_text text NOT NULL,
    corpus_fingerprint text NOT NULL,
    answer_text text NOT NULL,
    prompt text NOT NULL DEFAULT '',
    sources text[] NOT NULL DEFAULT '{}'::text[],
    retrieved_chunks jsonb NOT NULL DEFAULT '[]'::jsonb,
    used_llm boolean NOT NULL DEFAULT false,
    llm_backend text NOT NULL DEFAULT '',
    latency_ms double precision,
    hit_count int NOT NULL DEFAULT 0,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_answer_cache_query_hash
    ON answer_cache(query_hash);

CREATE TABLE IF NOT EXISTS document_parse_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash text NOT NULL,
    parser_fingerprint text NOT NULL,
    file_name text,
    parse_method text NOT NULL,
    ocr_used boolean NOT NULL DEFAULT false,
    ocr_reason text,
    extracted_char_count int,
    ocr_char_count int,
    quality jsonb NOT NULL DEFAULT '{}'::jsonb,
    blocks jsonb NOT NULL DEFAULT '[]'::jsonb,
    hit_count int NOT NULL DEFAULT 0,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(file_hash, parser_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_document_parse_cache_file_hash
    ON document_parse_cache(file_hash);
