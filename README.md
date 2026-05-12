# Enterprise Knowledge Base RAG Assistant

This repo implements an enterprise knowledge base RAG / GraphRAG assistant for renewable-energy research. It includes document ingestion, PDF/HTML parsing, chunking, real embeddings, hybrid retrieval, BGE reranking, grounded answer generation, and evaluation tooling.

## Current Status

- Corpus: 12,486 chunks from renewable-energy company reports and industry sources.
- Embeddings: 100% covered with `BAAI/bge-m3` in the current local database.
- Retrieval: BM25 + vector search + RRF + BGE reranking.
- Answering: Ollama/Qwen generation with direct-extractive guardrails for high-confidence numeric facts.
- Caching: PostgreSQL-backed query embedding, retrieval result, answer, and document parse caches.
- API: FastAPI chat routes and streaming endpoint.

Latest local retrieval regression on `data/golden_dataset.json`:

```text
Hybrid+Rerank:
Cases   51
Hit@5   100.0%
MRR     0.915
P.Hit@5 100.0%

Answer keyword eval:
10-sample keyword hit rate: 90.0%
```

## What is included

- PostgreSQL schema for documents, chunks, embeddings, permissions, feedback, eval, and graph tables
- Redis for cache and temporary job state
- Sample seed docs for a新能源 scenario
- Bootstrap scripts to initialize the database, collect energy data, ingest collected energy data, ingest sample docs, and extract graph data

## Quick start

1. Copy `.env.example` to `.env`
2. Install Python dependencies:

```bash
python3 -m pip install -e .
```

3. Start services:

```bash
docker compose up -d
```

4. Initialize schema:

```bash
python3 scripts/bootstrap_db.py
```

5. Seed sample documents:

```bash
python3 scripts/seed_sample_data.py
```

6. Extract graph entities and relations:

```bash
python3 scripts/extract_graph.py
```

7. Collect public新能源 source documents:

```bash
python3 scripts/collect_energy_data.py
```

8. Ingest collected energy documents:

```bash
python3 scripts/ingest_energy_data.py
```

## Notes

- Early bootstrap runs used deterministic hash-based embeddings; the current local setup uses real `BAAI/bge-m3` embeddings.
- The graph extraction script currently uses heuristics and a small domain glossary. A model-based extractor can be added later.
- The energy collection script reads from `data/sources/energy_sources.json` and stores downloaded pages and PDFs under `data/raw/energy/`.
- The energy ingestion script parses HTML and PDFs into chapter-level and paragraph-level chunks, removes repeated headers/footers and common disclaimer noise, preserves table-like blocks when possible, caps long paragraphs into smaller overlapping chunks, uses `pdftotext` first, falls back to OCR only when extraction is too sparse, records page-level parsing quality on each ingestion job, and writes chunks into PostgreSQL.

## Reranking

The retrieval service supports two rerank modes:

- `heuristic`: lightweight local rules, no model dependency.
- `bge`: a real cross-encoder reranker through `sentence-transformers`, for example `BAAI/bge-reranker-base`.

Install rerank dependencies when you want to run the model locally:

```bash
python3 -m pip install -e '.[rerank]'
```

Run eval with the bge backend:

```bash
RAG_ENABLE_RERANK=1 \
RAG_RERANK_BACKEND=bge \
RAG_RERANK_MODEL=BAAI/bge-reranker-base \
RAG_RERANK_CANDIDATES=30 \
RAG_RERANK_RECALL_WEIGHT=0.5 \
.venv/bin/python -m rag_assistant.eval_runner --top-k 6
```

For offline runs, download/cache the model first and set:

```bash
RAG_RERANK_LOCAL_FILES_ONLY=1
```

If the bge model or dependency is unavailable, the service records `debug.rerank_error` and falls back to the heuristic reranker unless `RAG_RERANK_FALLBACK=none` is set.

### Rerank Cache And Candidate Tuning

The `bge` reranker stores pairwise scores in PostgreSQL:

```text
query_hash + chunk_id + rerank_model -> score
```

This avoids repeated cross-encoder inference for the same query/chunk pair. Cache statistics are exposed in `debug.rerank_cache` for each answer/eval result.

Run candidate tuning:

```bash
.venv/bin/python -m rag_assistant.tune_rerank_candidates --candidates 20,30,50
```

Current local retrieval regression on `data/golden_dataset.json` with answer/retrieval caches disabled:

```text
candidates=20 Hit@5=1.000 MRR=0.904 P.Hit@5=1.000
candidates=30 Hit@5=1.000 MRR=0.908 P.Hit@5=1.000
```

The default `RAG_RERANK_CANDIDATES` is therefore set to `30`, which keeps Hit@5 at 100% while reducing first-query latency.

## Cold Query Optimization

For first-time queries that cannot use the final answer cache, the project includes:

- Lower default cross-encoder rerank candidates: `30` instead of the previous larger candidate pool.
- pgvector HNSW index for ANN vector search.
- Optional app startup warmup for chunks, IDF, embedding model, and reranker model.
- MPS-aware embedding model loading on Apple Silicon.

Apply the vector indexes:

```bash
make vector-index
```

Warm local models manually:

```bash
make warmup
```

Or warm them when the FastAPI app starts:

```bash
export RAG_WARMUP_ON_STARTUP=1
make api
```

Recent local query check with answer/retrieval caches disabled:

```text
Tesla automotive revenue query:
before cache/latency tuning: ~15.3s
after candidates=30 + HNSW + cached query/rerank pairs: ~2.2s
answer cache hit: ~57ms
```

## Operational Hardening

Apply the optional production-readiness tables:

```bash
psql "$DATABASE_URL" -f sql/006_operational_hardening.sql
```

Available controls:

```bash
# Optional comma-separated API keys. If unset, local dev remains open.
export RAG_API_KEYS="dev-key-1,dev-key-2"

# Per-client request cap. Set 0 to disable.
export RAG_RATE_LIMIT_PER_MIN=60
```

The FastAPI app records request IDs, latency, status codes, and recent errors in `api_request_logs` when the ops schema is installed. A lightweight dashboard endpoint is available at:

```text
GET /v1/ops/metrics
```

Useful local commands:

```bash
make test          # unit tests for guardrails
make regression    # retrieval regression gate on golden_dataset.json
make cache-schema  # install query/retrieval/answer/document parse cache tables
make vector-index  # install pgvector HNSW indexes
make fact-graph-schema # install structured fact graph tables
make job-schema    # install async job queue columns/indexes
make warmup        # preload chunks/IDF/models for lower first-request latency
make worker        # run the PostgreSQL-backed background worker
make api           # start FastAPI locally
make frontend      # start React research workbench on port 5174
```

## Async Job Queue

Long-running backend work should not block API requests. The project includes a PostgreSQL-backed queue using `ingestion_jobs`.

Install queue columns and indexes:

```bash
make job-schema
```

Create a job:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"job_type":"smoke_test","payload":{"source":"api-test"},"max_retries":1}'
```

Run one worker iteration:

```bash
python3 scripts/job_worker.py --once
```

Supported job types:

- `smoke_test`: verifies the queue path.
- `embed_missing_chunks`: runs background embedding for chunks without current embeddings.
- `ingest_pdf`: parses/chunks a local PDF path asynchronously.

The worker claims jobs with `FOR UPDATE SKIP LOCKED`, records `running/succeeded/retry/failed`, supports retry backoff, and writes structured `result` JSON back to `ingestion_jobs`.

## GraphRAG Fact Loop And Human Review

Install the structured fact graph table:

```bash
make fact-graph-schema
```

When `POST /v1/research/ask` returns a high-confidence numeric answer, the API extracts a small GraphRAG fact:

```text
company -> metric -> value -> source
Tesla -> total automotive revenues -> 82,419 USD million -> Tesla 2023 Annual Report p.51
```

The fact is stored in `research_facts` and a pending human review item is created in `answer_reviews`.

Useful endpoints:

```text
GET   /v1/graph/facts?company=Tesla
GET   /v1/graph/companies
GET   /v1/reviews?status=pending
PATCH /v1/reviews/{review_id}
```

Approving a review updates the linked `research_facts.review_status`, so downstream report generation can choose to use only approved facts.

## RAG Caching

The project includes a local PostgreSQL-backed cache layer for the expensive parts of the RAG pipeline:

- `query_embedding_cache`: reuses query embeddings by normalized query + embedding model/version.
- `retrieval_result_cache`: reuses top-k chunk IDs by query + retrieval strategy + corpus fingerprint.
- `answer_cache`: reuses final answers by query + corpus fingerprint + generation/retrieval strategy.
- `document_parse_cache`: reuses PDF parse/OCR output by file hash + parser fingerprint.

Install the cache tables:

```bash
make cache-schema
```

The caches are enabled by default and can be disabled independently:

```bash
export RAG_QUERY_EMBEDDING_CACHE=0
export RAG_RETRIEVAL_RESULT_CACHE=0
export RAG_ANSWER_CACHE=0
export RAG_DOCUMENT_PARSE_CACHE=0
```

Retrieval and answer cache keys include a corpus fingerprint, so document/chunk/embedding changes naturally produce new cache keys instead of returning stale evidence.

## CI Pipeline

This repo includes a GitHub Actions workflow at `.github/workflows/ci.yml`.

It runs on push, pull request, or manual trigger:

- Python dependency install
- Python compile check for key backend modules
- Unit tests for RAG guardrails
- Frontend dependency install with `npm ci`
- React production build

The heavier RAG retrieval regression is intentionally kept as a local gate for now because it depends on the seeded local PostgreSQL/pgvector dataset and optional reranker/model cache:

```bash
make regression
```

Production CD is intentionally not enabled yet. The next practical step would be adding Docker image build/push after the API and worker deployment target is decided.

## React Research Workbench

The `frontend/` app is a Vite + React workbench for the research agent. It calls:

- `POST /v1/research/ask` for grounded answers, citations, retrieved chunks, and debug metadata.
- `GET /v1/ops/metrics` for request/error/latency dashboard cards.

Run locally:

```bash
make api
make frontend
```

Then open:

```text
http://localhost:5174
```
