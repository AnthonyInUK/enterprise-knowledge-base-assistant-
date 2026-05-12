"""
routers/health.py

GET /v1/health       — system health check
GET /v1/cache/stats  — semantic cache statistics
DELETE /v1/cache     — clear expired cache entries
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from ..deps import CacheDep, DbDep
from ..models import CacheStats, HealthResponse

router = APIRouter(tags=["health & cache"])


@router.get("/v1/health", response_model=HealthResponse,
            summary="System health check")
async def health(db: DbDep, cache: CacheDep):
    # DB check
    db_ok = False
    try:
        db.fetch_one("SELECT 1")
        db_ok = True
    except Exception:
        pass

    # LLM backend
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-"):
        llm_backend = "anthropic"
    elif os.getenv("OLLAMA_URL"):
        llm_backend = "ollama"
    else:
        llm_backend = "none"

    # Cache entries
    cache_entries = 0
    if cache is not None:
        try:
            stats = cache.stats()
            cache_entries = stats["db_total_entries"]
        except Exception:
            pass

    return HealthResponse(
        status        = "ok" if db_ok else "degraded",
        db            = db_ok,
        llm           = llm_backend,
        cache_entries = cache_entries,
    )


@router.get("/v1/cache/stats", response_model=CacheStats,
            summary="Semantic cache statistics")
async def cache_stats(cache: CacheDep):
    if cache is None:
        return CacheStats(
            session_hits=0, session_misses=0, session_hit_rate="0.0%",
            db_total_entries=0, db_total_hits=0, threshold=0.0,
        )
    s = cache.stats()
    return CacheStats(**s)


@router.delete("/v1/cache", summary="Clear expired cache entries")
async def clear_cache(cache: CacheDep):
    if cache is None:
        return {"deleted": 0}
    n = cache.clear_expired()
    return {"deleted": n, "message": "Expired entries cleared"}
