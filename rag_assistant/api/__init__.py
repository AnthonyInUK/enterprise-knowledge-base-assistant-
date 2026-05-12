"""
rag_assistant.api
=================
FastAPI service layer for the enterprise RAG knowledge base assistant.

Endpoints:
  POST   /v1/chat              — synchronous question answering
  GET    /v1/chat/stream       — SSE streaming (intent → tokens → done)
  GET    /v1/threads           — list conversation threads
  GET    /v1/threads/{id}      — thread metadata
  GET    /v1/memories          — long-term memory entries
  GET    /v1/health            — system health check
  GET    /v1/cache/stats       — semantic cache statistics
  DELETE /v1/cache             — clear expired entries

Start:
  python scripts/serve.py
  uvicorn rag_assistant.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from .app import create_app

__all__ = ["create_app"]
