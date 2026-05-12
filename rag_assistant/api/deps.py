"""
deps.py — App-level shared state and FastAPI dependency injectors.

Single instances of DB / graph / cache are created once at startup
(via lifespan) and shared across all requests.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request


# ── app state holder ──────────────────────────────────────────────────────────

class AppState:
    db    = None
    graph = None
    cache = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once on startup, clean up on shutdown."""
    from rag_assistant.core import Database, RetrievalService
    from rag_assistant.agent.graph import build_agent_graph
    from rag_assistant.agent.cache import SemanticCache

    _state.db    = Database()
    _state.graph = build_agent_graph(_state.db)
    _state.cache = SemanticCache(_state.db, threshold=0.95)
    app.state.db = _state.db
    app.state.graph = _state.graph
    app.state.cache = _state.cache

    if os.getenv("RAG_WARMUP_ON_STARTUP", "0") == "1":
        service = RetrievalService(_state.db)
        app.state.warmup = service.warmup()
        print(f"🔥 RAG warmup complete: {app.state.warmup}")

    print("✅ Agent graph ready")
    yield

    # teardown (connections are managed by psycopg pool)
    print("👋 Shutting down")


# ── dependency functions ──────────────────────────────────────────────────────

def get_graph(request: Request):
    return _state.graph


def get_cache(request: Request):
    return _state.cache


def get_db(request: Request):
    return _state.db


GraphDep = Annotated[object, Depends(get_graph)]
CacheDep = Annotated[object, Depends(get_cache)]
DbDep    = Annotated[object, Depends(get_db)]
