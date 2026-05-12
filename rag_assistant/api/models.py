"""
models.py — Pydantic request / response schemas.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── request schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:  str  = Field(..., min_length=1, max_length=2000,
                            description="User question in any language")
    thread_id: str  = Field(default="default",
                            description="Conversation thread ID for multi-turn memory")
    user_id:   str  = Field(default="default",
                            description="User identifier for long-term memory scoping")
    top_k:     int  = Field(default=6, ge=1, le=20,
                            description="Number of chunks to retrieve")

    model_config = {"json_schema_extra": {
        "example": {
            "question":  "What was Vestas total revenue in 2023?",
            "thread_id": "session-abc123",
            "user_id":   "analyst-1",
        }
    }}


# ── response schemas ──────────────────────────────────────────────────────────

class ReActStepSchema(BaseModel):
    thought:      str
    action:       str
    action_input: Any
    observation:  str


class ChatResponse(BaseModel):
    answer:           str
    intent:           str
    intent_reason:    str
    sources:          list[str]
    react_steps:      list[ReActStepSchema] = []
    latency_ms:       float
    cache_hit:        bool        = False
    cache_similarity: float | None = None
    thread_id:        str
    user_id:          str


class RetrievedChunkSchema(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_title: str
    chunk_level: str
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    text: str
    matched_terms: list[str] = []
    reason: str = ""


class ResearchAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunkSchema]
    latency_ms: float
    used_llm: bool
    debug: dict[str, Any]
    facts: list[dict[str, Any]] = []


class FactReviewUpdate(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected|needs_revision|pending)$")
    reviewer_id: str = "analyst-ui"
    corrected_fact: str | None = None


class FactReviewCreate(BaseModel):
    reviewed_fact: str = Field(..., min_length=1, max_length=2000)
    source_citation: str | None = None
    reviewer_id: str | None = None
    metadata: dict[str, Any] = {}


class JobCreateRequest(BaseModel):
    job_type: str = Field(..., min_length=1, max_length=80)
    payload: dict[str, Any] = {}
    document_id: str | None = None
    max_retries: int = Field(default=3, ge=0, le=10)


class JobResponse(BaseModel):
    id: str
    document_id: str | None = None
    job_type: str
    status: str
    retries: int = 0
    max_retries: int = 3
    error_message: str | None = None
    metadata: dict[str, Any] = {}
    result: dict[str, Any] = {}
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ThreadSummary(BaseModel):
    thread_id:      str
    title:          str | None
    turn_count:     int
    intent_history: list[str]
    updated_at:     str | None


class MemoryRecord(BaseModel):
    content:      str
    importance:   float
    thread_id:    str
    created_at:   str | None


class CacheStats(BaseModel):
    session_hits:      int
    session_misses:    int
    session_hit_rate:  str
    db_total_entries:  int
    db_total_hits:     int
    threshold:         float


class HealthResponse(BaseModel):
    status:    str          # "ok" | "degraded"
    db:        bool
    llm:       str          # "ollama" | "anthropic" | "none"
    cache_entries: int
    version:   str = "0.2.0"


class OpsMetrics(BaseModel):
    request_count_24h: int
    error_count_24h: int
    avg_latency_ms_24h: float | None
    p95_latency_ms_24h: float | None
    query_count_24h: int
    avg_query_latency_ms_24h: float | None
    pending_reviews: int
    recent_errors: list[dict[str, Any]] = []


# ── SSE event envelope ────────────────────────────────────────────────────────
# Sent as:  data: <json>\n\n

class SSEEvent(BaseModel):
    """
    Streaming event types:

    intent   — intent classification complete
    memory   — long-term memory recall result
    progress — pipeline stage update (e.g. "Retrieving documents…")
    token    — incremental answer text (for word-by-word streaming)
    done     — final payload (sources, latency, cache_hit)
    error    — pipeline error
    """
    type:    str
    payload: dict[str, Any]
