"""
cache.py — Semantic query cache backed by pgvector.

How it works
------------
1. Incoming query → embed with BAAI/bge-m3
2. Cosine similarity search against query_cache table
3. If best match >= threshold (default 0.95) → return cached answer instantly
4. Otherwise → run full pipeline, then persist result to cache

Latency impact
--------------
Cache MISS  : full pipeline  ~30-60 s  (Ollama 7B)
Cache HIT   : pgvector ANN   ~5-20 ms

The high threshold (0.95) ensures only genuine paraphrases hit the cache.
e.g. "What was Vestas revenue in 2023?" ≈ "Vestas 2023 annual revenue?" → HIT
     "What was Vestas revenue in 2023?" vs "What was Vestas profit in 2023?" → MISS
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_assistant.core import Database


@dataclass
class CacheEntry:
    query_text: str
    answer: str
    intent: str
    sources: list[str]
    react_steps: list[dict]
    latency_ms: float          # original pipeline latency
    similarity: float          # cosine similarity that triggered this hit
    hit_count: int


class SemanticCache:
    """pgvector-backed semantic cache for RAG query results."""

    def __init__(
        self,
        db: "Database",
        embed_model: str = "BAAI/bge-m3",
        threshold: float = 0.95,
    ) -> None:
        self.db = db
        self.embed_model = embed_model
        self.threshold = threshold
        self._model = None          # lazy-loaded SentenceTransformer
        self._stats = {"hits": 0, "misses": 0}

    # ── embedding ─────────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.embed_model, device=device)
        return self._model

    def _embed(self, text: str) -> list[float]:
        return self._get_model().encode(text, normalize_embeddings=True).tolist()

    def _vec_str(self, vec: list[float]) -> str:
        return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

    # ── get ───────────────────────────────────────────────────────────────────

    def get(self, query: str) -> CacheEntry | None:
        """
        Look up a query in the cache.
        Returns CacheEntry if similarity >= threshold, else None.
        """
        t0 = time.perf_counter()
        vec = self._embed(query)
        vec_str = self._vec_str(vec)

        row = self.db.fetch_one(
            """
            SELECT
                id::text,
                query_text,
                answer,
                intent,
                sources,
                react_steps,
                latency_ms,
                hit_count,
                1 - (query_embedding <=> %s::vector) AS similarity
            FROM query_cache
            WHERE query_embedding IS NOT NULL
              AND expires_at > NOW()
            ORDER BY query_embedding <=> %s::vector
            LIMIT 1
            """,
            (vec_str, vec_str),
        )

        lookup_ms = (time.perf_counter() - t0) * 1000

        if row is None or float(row["similarity"]) < self.threshold:
            self._stats["misses"] += 1
            return None

        # Update hit stats
        self.db.execute(
            """
            UPDATE query_cache
            SET hit_count   = hit_count + 1,
                last_hit_at = NOW()
            WHERE id = %s
            """,
            (row["id"],),
        )

        self._stats["hits"] += 1

        react_steps = row["react_steps"]
        if isinstance(react_steps, str):
            react_steps = json.loads(react_steps)

        return CacheEntry(
            query_text  = row["query_text"],
            answer      = row["answer"],
            intent      = row["intent"] or "general",
            sources     = list(row["sources"] or []),
            react_steps = list(react_steps or []),
            latency_ms  = float(row["latency_ms"] or 0),
            similarity  = float(row["similarity"]),
            hit_count   = int(row["hit_count"]) + 1,
        )

    # ── set ───────────────────────────────────────────────────────────────────

    def set(self, query: str, result: dict[str, Any]) -> None:
        """
        Persist a query result to the cache.
        Silently ignores duplicates (ON CONFLICT DO NOTHING).
        """
        answer = result.get("answer", "")
        if not answer or len(answer) < 10:
            return  # don't cache empty / error responses

        vec = self._embed(query)
        vec_str = self._vec_str(vec)

        self.db.execute(
            """
            INSERT INTO query_cache
                (query_text, query_embedding, answer, intent, sources,
                 react_steps, latency_ms)
            VALUES
                (%s, %s::vector, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (query_text) DO NOTHING
            """,
            (
                query,
                vec_str,
                answer,
                result.get("intent", "general"),
                result.get("sources", []),
                json.dumps(result.get("react_steps", [])),
                result.get("latency_ms", 0),
            ),
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return cache statistics for this session + DB totals."""
        session_total = self._stats["hits"] + self._stats["misses"]
        session_rate  = (
            self._stats["hits"] / session_total if session_total > 0 else 0.0
        )

        row = self.db.fetch_one(
            """
            SELECT
                COUNT(*)            AS total_entries,
                SUM(hit_count)      AS total_hits,
                AVG(hit_count)      AS avg_hits_per_entry,
                MAX(last_hit_at)    AS last_hit
            FROM query_cache
            WHERE expires_at > NOW()
            """
        )

        return {
            "session_hits":     self._stats["hits"],
            "session_misses":   self._stats["misses"],
            "session_hit_rate": f"{session_rate:.1%}",
            "db_total_entries": int(row["total_entries"] or 0),
            "db_total_hits":    int(row["total_hits"] or 0),
            "threshold":        self.threshold,
        }

    def clear_expired(self) -> int:
        """Delete expired cache entries. Returns count deleted."""
        result = self.db.fetch_one(
            "DELETE FROM query_cache WHERE expires_at <= NOW() RETURNING id"
        )
        return 1 if result else 0

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from cache. Returns True if found."""
        result = self.db.fetch_one(
            "DELETE FROM query_cache WHERE query_text = %s RETURNING id",
            (query,),
        )
        return result is not None
