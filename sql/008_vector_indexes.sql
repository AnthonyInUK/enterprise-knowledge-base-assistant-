CREATE EXTENSION IF NOT EXISTS vector;

-- Speeds up pgvector cosine nearest-neighbor search used by hybrid retrieval.
-- Build after real embeddings have replaced bootstrap vectors.
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_hnsw
    ON embeddings
    USING hnsw (embedding vector_cosine_ops);

-- Helps model-filtered vector queries before ANN ordering.
CREATE INDEX IF NOT EXISTS idx_embeddings_model_chunk
    ON embeddings (embedding_model, chunk_id);
