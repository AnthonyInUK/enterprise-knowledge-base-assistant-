"""
app.py — FastAPI application factory.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg.types.json import Json

from .deps import lifespan
from .routers import chat, graph, health, jobs, ops, research, reviews, threads


_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _configured_api_keys() -> set[str]:
    raw = os.getenv("RAG_API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def _client_id(request: Request) -> str:
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        return f"key:{api_key[:8]}"
    if request.client:
        return request.client.host
    return "unknown"


def _rate_limited(client_id: str) -> bool:
    limit = int(os.getenv("RAG_RATE_LIMIT_PER_MIN", "60"))
    if limit <= 0:
        return False
    now = time.time()
    bucket = _RATE_LIMIT_BUCKETS[client_id]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= limit:
        return True
    bucket.append(now)
    return False


def _log_api_request(
    app: FastAPI,
    request: Request,
    request_id: str,
    status_code: int,
    latency_ms: int,
    error_message: str | None = None,
) -> None:
    db = getattr(app.state, "db", None)
    if db is None:
        return
    try:
        columns = db.fetch_all(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'api_request_logs'
            """
        )
        if not columns:
            return
        db.execute(
            """
            INSERT INTO api_request_logs
                (request_id, method, path, status_code, latency_ms, user_id,
                 client_host, error_message, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (request_id) DO NOTHING
            """,
            (
                request_id,
                request.method,
                request.url.path,
                status_code,
                latency_ms,
                request.headers.get("x-user-id"),
                request.client.host if request.client else None,
                error_message,
                Json({"query": str(request.url.query), "user_agent": request.headers.get("user-agent", "")}),
            ),
        )
    except Exception:
        return


def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Enterprise RAG Knowledge Base API",
        description = (
            "Hybrid BM25+Vector search, BGE reranking, LangGraph agent, "
            "long-term memory, and semantic cache for renewable energy research."
        ),
        version     = "0.2.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # CORS — open for local dev; restrict origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    @app.middleware("http")
    async def operational_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        api_keys = _configured_api_keys()
        if api_keys and request.url.path not in {"/", "/v1/health"}:
            supplied = request.headers.get("x-api-key", "")
            if supplied not in api_keys:
                response = JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
                response.headers["x-request-id"] = request_id
                _log_api_request(app, request, request_id, 401, int((time.perf_counter() - started) * 1000))
                return response

        client_id = _client_id(request)
        if _rate_limited(client_id):
            response = JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
            response.headers["x-request-id"] = request_id
            _log_api_request(app, request, request_id, 429, int((time.perf_counter() - started) * 1000))
            return response

        try:
            response = await call_next(request)
            status_code = response.status_code
            error_message = None
        except Exception as exc:
            status_code = 500
            error_message = str(exc)
            response = JSONResponse({"detail": "Internal server error", "request_id": request_id}, status_code=500)

        latency_ms = int((time.perf_counter() - started) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = str(latency_ms)
        _log_api_request(app, request, request_id, status_code, latency_ms, error_message)
        return response

    # Routers
    app.include_router(chat.router)
    app.include_router(research.router)
    app.include_router(graph.router)
    app.include_router(reviews.router)
    app.include_router(jobs.router)
    app.include_router(threads.router)
    app.include_router(health.router)
    app.include_router(ops.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Enterprise RAG API",
            "version": "0.2.0",
            "docs":    "/docs",
        }

    return app


# Module-level `app` instance for uvicorn
app = create_app()
