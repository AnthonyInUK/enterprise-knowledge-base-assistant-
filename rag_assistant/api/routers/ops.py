"""
routers/ops.py

Operational metrics for a lightweight dashboard.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import DbDep
from ..models import OpsMetrics

router = APIRouter(prefix="/v1/ops", tags=["ops"])


def _table_exists(db, table_name: str) -> bool:
    row = db.fetch_one(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables WHERE table_name = %s
        ) AS exists
        """,
        (table_name,),
    )
    return bool(row and row["exists"])


@router.get("/metrics", response_model=OpsMetrics, summary="Operational metrics")
async def metrics(db: DbDep):
    request_count = 0
    error_count = 0
    avg_latency = None
    p95_latency = None
    recent_errors: list[dict] = []

    if _table_exists(db, "api_request_logs"):
        row = db.fetch_one(
            """
            SELECT
                COUNT(*)::int AS request_count,
                COUNT(*) FILTER (WHERE status_code >= 500)::int AS error_count,
                ROUND(AVG(latency_ms)::numeric, 2)::float AS avg_latency,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::float AS p95_latency
            FROM api_request_logs
            WHERE created_at >= now() - interval '24 hours'
            """
        )
        if row:
            request_count = int(row["request_count"] or 0)
            error_count = int(row["error_count"] or 0)
            avg_latency = row["avg_latency"]
            p95_latency = row["p95_latency"]
        recent_errors = db.fetch_all(
            """
            SELECT request_id, method, path, status_code, latency_ms, error_message,
                   created_at::text AS created_at
            FROM api_request_logs
            WHERE status_code >= 400
            ORDER BY created_at DESC
            LIMIT 10
            """
        )

    query_count = 0
    avg_query_latency = None
    if _table_exists(db, "query_logs"):
        row = db.fetch_one(
            """
            SELECT COUNT(*)::int AS query_count,
                   ROUND(AVG(latency_ms)::numeric, 2)::float AS avg_latency
            FROM query_logs
            WHERE created_at >= now() - interval '24 hours'
            """
        )
        if row:
            query_count = int(row["query_count"] or 0)
            avg_query_latency = row["avg_latency"]

    pending_reviews = 0
    if _table_exists(db, "answer_reviews"):
        row = db.fetch_one("SELECT COUNT(*)::int AS n FROM answer_reviews WHERE status = 'pending'")
        pending_reviews = int(row["n"] or 0) if row else 0

    return OpsMetrics(
        request_count_24h=request_count,
        error_count_24h=error_count,
        avg_latency_ms_24h=avg_latency,
        p95_latency_ms_24h=p95_latency,
        query_count_24h=query_count,
        avg_query_latency_ms_24h=avg_query_latency,
        pending_reviews=pending_reviews,
        recent_errors=[dict(row) for row in recent_errors],
    )
