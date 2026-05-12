from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json

from rag_assistant.core import Database


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


@dataclass(slots=True)
class JobResult:
    ok: bool
    payload: dict[str, Any]


def enqueue_job(
    db: Database,
    job_type: str,
    payload: dict[str, Any],
    document_id: str | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    row = db.fetch_one(
        """
        INSERT INTO ingestion_jobs (
            document_id, job_type, status, max_retries, metadata, available_at
        )
        VALUES (%s, %s, 'queued', %s, %s, now())
        RETURNING id::text, document_id::text, job_type, status, retries,
                  max_retries, metadata, created_at::text
        """,
        (document_id, job_type, max_retries, Json({"payload": payload})),
    )
    return dict(row) if row else {}


def get_job(db: Database, job_id: str) -> dict[str, Any] | None:
    row = db.fetch_one(
        """
        SELECT id::text, document_id::text, job_type, status, retries, max_retries,
               error_message, started_at::text, finished_at::text,
               available_at::text, locked_at::text, locked_by,
               metadata, result, created_at::text
        FROM ingestion_jobs
        WHERE id = %s
        """,
        (job_id,),
    )
    return dict(row) if row else None


def list_jobs(db: Database, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if status:
        rows = db.fetch_all(
            """
            SELECT id::text, document_id::text, job_type, status, retries, max_retries,
                   error_message, started_at::text, finished_at::text,
                   metadata, result, created_at::text
            FROM ingestion_jobs
            WHERE status = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (status, limit),
        )
    else:
        rows = db.fetch_all(
            """
            SELECT id::text, document_id::text, job_type, status, retries, max_retries,
                   error_message, started_at::text, finished_at::text,
                   metadata, result, created_at::text
            FROM ingestion_jobs
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
    return [dict(row) for row in rows]


class JobWorker:
    def __init__(self, db: Database | None = None, worker_id: str | None = None) -> None:
        self.db = db or Database()
        self.worker_id = worker_id or f"{socket.gethostname()}:{time.time_ns()}"

    def claim_next(self) -> dict[str, Any] | None:
        with self.db.connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH next_job AS (
                    SELECT id
                    FROM ingestion_jobs
                    WHERE status IN ('queued', 'retry')
                      AND available_at <= now()
                    ORDER BY created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ingestion_jobs j
                SET status = 'running',
                    locked_at = now(),
                    locked_by = %s,
                    started_at = COALESCE(started_at, now()),
                    error_message = NULL
                FROM next_job
                WHERE j.id = next_job.id
                RETURNING j.id::text, j.job_type, j.retries, j.max_retries,
                          j.metadata, j.document_id::text
                """,
                (self.worker_id,),
            )
            row = cur.fetchone()
            conn.commit()
        return dict(row) if row else None

    def run_once(self) -> dict[str, Any] | None:
        job = self.claim_next()
        if not job:
            return None
        try:
            result = self._run_job(job)
            if result.ok:
                self._mark_succeeded(job["id"], result.payload)
            else:
                self._mark_failed_or_retry(job, RuntimeError(result.payload.get("error", "job failed")))
            return {"job_id": job["id"], "status": "succeeded" if result.ok else "failed", "result": result.payload}
        except Exception as exc:
            self._mark_failed_or_retry(job, exc)
            return {"job_id": job["id"], "status": "error", "error": str(exc)}

    def _run_job(self, job: dict[str, Any]) -> JobResult:
        metadata = dict(job.get("metadata") or {})
        payload = dict(metadata.get("payload") or {})
        job_type = str(job.get("job_type"))

        if job_type == "smoke_test":
            return JobResult(True, {"message": "worker ok", "payload": payload})

        if job_type == "embed_missing_chunks":
            from scripts.embed_chunks import embed_chunks
            result = embed_chunks(
                batch_size=int(payload.get("batch_size", 32)),
                limit=payload.get("limit"),
                force=bool(payload.get("force", False)),
                verbose=bool(payload.get("verbose", False)),
                device=payload.get("device"),
            )
            return JobResult(True, result)

        if job_type == "ingest_pdf":
            from scripts.ingest_single_pdf import ingest_pdf
            pdf_path = Path(str(payload["pdf_path"]))
            result = ingest_pdf(
                pdf_path=pdf_path,
                title=str(payload["title"]),
                company=str(payload.get("company") or ""),
                source_url=str(payload.get("source_url") or ""),
                dry_run=False,
            )
            return JobResult(True, result)

        return JobResult(False, {"error": f"Unsupported job_type: {job_type}"})

    def _mark_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        self.db.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'succeeded',
                finished_at = now(),
                locked_at = NULL,
                locked_by = NULL,
                result = %s
            WHERE id = %s
            """,
            (Json(result), job_id),
        )

    def _mark_failed_or_retry(self, job: dict[str, Any], exc: Exception) -> None:
        retries = int(job.get("retries") or 0)
        max_retries = int(job.get("max_retries") or 3)
        next_retries = retries + 1
        if next_retries <= max_retries:
            self.db.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'retry',
                    retries = %s,
                    error_message = %s,
                    locked_at = NULL,
                    locked_by = NULL,
                    available_at = now() + make_interval(secs => LEAST(300, POWER(2, %s)::int))
                WHERE id = %s
                """,
                (next_retries, str(exc), next_retries, job["id"]),
            )
            return
        self.db.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'failed',
                retries = %s,
                error_message = %s,
                finished_at = now(),
                locked_at = NULL,
                locked_by = NULL
            WHERE id = %s
            """,
            (next_retries, str(exc), job["id"]),
        )
