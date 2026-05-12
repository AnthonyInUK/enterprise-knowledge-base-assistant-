from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from rag_assistant.jobs import enqueue_job, get_job, list_jobs

from ..deps import DbDep
from ..models import JobCreateRequest, JobResponse

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


ALLOWED_JOB_TYPES = {"smoke_test", "embed_missing_chunks", "ingest_pdf"}


@router.post("", response_model=JobResponse, summary="Enqueue an async job")
async def create_job(req: JobCreateRequest, db: DbDep):
    if req.job_type not in ALLOWED_JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported job_type: {req.job_type}")
    if req.job_type == "ingest_pdf":
        missing = [key for key in ("pdf_path", "title") if key not in req.payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing payload fields: {', '.join(missing)}")
    job = enqueue_job(
        db,
        job_type=req.job_type,
        payload=req.payload,
        document_id=req.document_id,
        max_retries=req.max_retries,
    )
    return JobResponse(**job)


@router.get("", response_model=list[JobResponse], summary="List async jobs")
async def list_async_jobs(
    db: DbDep,
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    return [JobResponse(**job) for job in list_jobs(db, status=status, limit=limit)]


@router.get("/{job_id}", response_model=JobResponse, summary="Get async job status")
async def get_async_job(job_id: str, db: DbDep):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job)
