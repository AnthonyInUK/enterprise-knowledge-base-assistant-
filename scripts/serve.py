#!/usr/bin/env python3
"""
serve.py — Start the FastAPI server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --reload

Quick test after startup:
    # Health check
    curl http://localhost:8000/v1/health

    # Synchronous chat
    curl -X POST http://localhost:8000/v1/chat \
         -H "Content-Type: application/json" \
         -d '{"question":"What was Vestas revenue in 2023?","thread_id":"t1"}'

    # SSE streaming
    curl -N "http://localhost:8000/v1/chat/stream?question=Vestas+2023+revenue&thread_id=t1"

    # List threads
    curl http://localhost:8000/v1/threads

    # Cache stats
    curl http://localhost:8000/v1/cache/stats

    # Interactive API docs
    open http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the RAG API server")
    parser.add_argument("--host",   default="0.0.0.0",   help="Bind host")
    parser.add_argument("--port",   default=8000, type=int, help="Bind port")
    parser.add_argument("--reload", action="store_true",  help="Auto-reload on code changes")
    parser.add_argument("--workers", default=1, type=int, help="Number of worker processes")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║  Enterprise RAG Knowledge Base API                   ║
║  http://{args.host}:{args.port}                              ║
║  Docs: http://localhost:{args.port}/docs                     ║
╚══════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "rag_assistant.api.app:app",
        host        = args.host,
        port        = args.port,
        reload      = args.reload,
        workers     = args.workers if not args.reload else 1,
        log_level   = "info",
    )


if __name__ == "__main__":
    main()
