CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type text NOT NULL,
    source_url text,
    title text NOT NULL,
    file_name text,
    file_hash text NOT NULL UNIQUE,
    doc_type text NOT NULL DEFAULT 'unknown',
    language text NOT NULL DEFAULT 'zh',
    status text NOT NULL DEFAULT 'new',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_no int NOT NULL DEFAULT 1,
    raw_path text NOT NULL,
    parsed_path text,
    content_hash text NOT NULL UNIQUE,
    published_at timestamptz,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(document_id, version_no)
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_version_id uuid NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    parent_chunk_id uuid,
    chunk_level text NOT NULL DEFAULT 'paragraph',
    section_title text,
    page_start int,
    page_end int,
    char_start int,
    char_end int,
    token_count int,
    chunk_type text NOT NULL DEFAULT 'paragraph',
    text text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(document_version_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_version_id ON chunks(document_version_id);

CREATE TABLE IF NOT EXISTS embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding_model text NOT NULL,
    embedding_version text NOT NULL DEFAULT 'v1',
    dimension int NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(embedding_model);
-- Build the HNSW vector index after replacing hash-bootstrap rows with real embeddings:
-- CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email text UNIQUE,
    display_name text NOT NULL,
    dept text,
    status text NOT NULL DEFAULT 'active',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS roles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    description text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id uuid NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS doc_acl (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    principal_type text NOT NULL CHECK (principal_type IN ('user', 'role', 'department')),
    principal_id text NOT NULL,
    permission_level text NOT NULL CHECK (permission_level IN ('view', 'annotate', 'admin')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(document_id, principal_type, principal_id)
);

CREATE TABLE IF NOT EXISTS query_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid REFERENCES users(id) ON DELETE SET NULL,
    query_text text NOT NULL,
    rewritten_query text,
    retrieved_chunk_ids uuid[] NOT NULL DEFAULT '{}'::uuid[],
    answer_text text,
    latency_ms int,
    model_name text,
    total_cost numeric(10, 4),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS rerank_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash text NOT NULL,
    chunk_id uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    rerank_model text NOT NULL,
    score double precision NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(query_hash, chunk_id, rerank_model)
);

CREATE INDEX IF NOT EXISTS idx_rerank_cache_query_model ON rerank_cache(query_hash, rerank_model);

CREATE TABLE IF NOT EXISTS feedback (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid REFERENCES users(id) ON DELETE SET NULL,
    query_log_id uuid REFERENCES query_logs(id) ON DELETE CASCADE,
    chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
    rating smallint NOT NULL CHECK (rating BETWEEN -1 AND 1),
    comment text,
    correct_answer text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_sets (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    description text,
    version text NOT NULL DEFAULT 'v1',
    created_by text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_set_id uuid NOT NULL REFERENCES eval_sets(id) ON DELETE CASCADE,
    question text NOT NULL,
    gold_answer text,
    gold_citations text[] NOT NULL DEFAULT '{}'::text[],
    gold_chunk_ids uuid[] NOT NULL DEFAULT '{}'::uuid[],
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_set_id uuid NOT NULL REFERENCES eval_sets(id) ON DELETE CASCADE,
    run_name text NOT NULL,
    model_name text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS eval_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id uuid NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    eval_case_id uuid NOT NULL REFERENCES eval_cases(id) ON DELETE CASCADE,
    retrieval_score numeric(6, 4),
    answer_score numeric(6, 4),
    citation_score numeric(6, 4),
    pass_fail boolean,
    raw_output jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(eval_run_id, eval_case_id)
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    job_type text NOT NULL,
    status text NOT NULL DEFAULT 'queued',
    retries int NOT NULL DEFAULT 0,
    error_message text,
    started_at timestamptz,
    finished_at timestamptz,
    parse_method text,
    ocr_used boolean NOT NULL DEFAULT false,
    ocr_reason text,
    extracted_char_count int,
    ocr_char_count int,
    page_count int,
    text_page_count int,
    ocr_page_count int,
    quality_score numeric(5,4),
    needs_review boolean NOT NULL DEFAULT false,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS entities (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    entity_type text NOT NULL,
    normalized_name text NOT NULL,
    source_chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(normalized_name, entity_type)
);

CREATE TABLE IF NOT EXISTS relations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    head_entity_id uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type text NOT NULL,
    tail_entity_id uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    source_chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
    confidence numeric(5, 4),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(head_entity_id, relation_type, tail_entity_id, source_chunk_id)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    start_offset int,
    end_offset int,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(entity_id, chunk_id, start_offset, end_offset)
);
