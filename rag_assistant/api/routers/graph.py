from __future__ import annotations

from fastapi import APIRouter, Query

from ..deps import DbDep

router = APIRouter(prefix="/v1/graph", tags=["graph"])


@router.get("/facts", summary="List structured research facts")
async def list_facts(
    db: DbDep,
    company: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    filters = []
    params: list[object] = []
    if company:
        filters.append("company ILIKE %s")
        params.append(f"%{company}%")
    if status:
        filters.append("review_status = %s")
        params.append(status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    rows = db.fetch_all(
        f"""
        SELECT
            id::text, company, metric, value, unit, period,
            source_citation, source_chunk_id::text, confidence::float,
            review_status, metadata, created_at::text, updated_at::text
        FROM research_facts
        {where}
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (*params, limit),
    )
    return [dict(row) for row in rows]


@router.get("/companies", summary="List companies with structured facts")
async def list_companies(db: DbDep):
    rows = db.fetch_all(
        """
        SELECT company, COUNT(*)::int AS fact_count,
               COUNT(*) FILTER (WHERE review_status = 'approved')::int AS approved_count
        FROM research_facts
        GROUP BY company
        ORDER BY fact_count DESC, company
        """
    )
    return [dict(row) for row in rows]
