from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from psycopg.types.json import Json

from ..deps import DbDep
from ..models import FactReviewCreate, FactReviewUpdate

router = APIRouter(prefix="/v1/reviews", tags=["reviews"])


@router.get("", summary="List human review items")
async def list_reviews(
    db: DbDep,
    status: str = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=200),
):
    rows = db.fetch_all(
        """
        SELECT
            id::text, query_log_id::text, reviewer_id, status,
            reviewed_fact, corrected_fact, source_citation,
            metadata, created_at::text, reviewed_at::text
        FROM answer_reviews
        WHERE status = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (status, limit),
    )
    return [dict(row) for row in rows]


@router.post("", summary="Create a review item")
async def create_review(req: FactReviewCreate, db: DbDep):
    row = db.fetch_one(
        """
        INSERT INTO answer_reviews (
            reviewer_id, reviewed_fact, source_citation, metadata
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id::text, status, reviewed_fact, source_citation, metadata, created_at::text
        """,
        (req.reviewer_id, req.reviewed_fact, req.source_citation, Json(req.metadata)),
    )
    return dict(row) if row else {}


@router.patch("/{review_id}", summary="Approve/reject a review item")
async def update_review(review_id: str, req: FactReviewUpdate, db: DbDep):
    row = db.fetch_one(
        """
        UPDATE answer_reviews
        SET status = %s,
            reviewer_id = %s,
            corrected_fact = COALESCE(%s, corrected_fact),
            reviewed_at = CASE WHEN %s = 'pending' THEN NULL ELSE now() END
        WHERE id = %s
        RETURNING id::text, status, reviewed_fact, corrected_fact,
                  source_citation, metadata, reviewed_at::text
        """,
        (req.status, req.reviewer_id, req.corrected_fact, req.status, review_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    metadata = dict(row.get("metadata") or {})
    fact_id = metadata.get("research_fact_id")
    if fact_id:
        db.execute(
            """
            UPDATE research_facts
            SET review_status = %s,
                metadata = metadata || %s::jsonb,
                reviewed_at = CASE WHEN %s = 'pending' THEN NULL ELSE now() END,
                updated_at = now()
            WHERE id = %s
            """,
            (
                req.status,
                Json({"review_id": review_id, "reviewer_id": req.reviewer_id, "corrected_fact": req.corrected_fact}),
                req.status,
                fact_id,
            ),
        )
    return dict(row)
