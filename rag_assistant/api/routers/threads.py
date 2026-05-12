"""
routers/threads.py

GET /v1/threads                     — list recent threads for a user
GET /v1/threads/{thread_id}         — thread metadata
GET /v1/threads/{thread_id}/history — conversation message history
GET /v1/memories                    — list long-term memories
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from rag_assistant.agent.memory import LongTermMemory

from ..deps import DbDep
from ..models import MemoryRecord, ThreadSummary

router = APIRouter(tags=["threads & memory"])


@router.get("/v1/threads", response_model=list[ThreadSummary],
            summary="List recent conversation threads")
async def list_threads(
    db: DbDep,
    user_id: str = Query(default="default"),
    limit:   int = Query(default=20, ge=1, le=100),
):
    mem = LongTermMemory(db)
    threads = mem.list_threads(user_id=user_id, limit=limit)
    return [
        ThreadSummary(
            thread_id      = t["thread_id"],
            title          = t.get("title"),
            turn_count     = t.get("turn_count", 0),
            intent_history = list(t.get("intent_history") or []),
            updated_at     = str(t["updated_at"]) if t.get("updated_at") else None,
        )
        for t in threads
    ]


@router.get("/v1/threads/{thread_id}", response_model=ThreadSummary,
            summary="Get thread metadata")
async def get_thread(thread_id: str, db: DbDep, user_id: str = Query(default="default")):
    row = db.fetch_one(
        "SELECT * FROM conversation_threads WHERE thread_id = %s AND user_id = %s",
        (thread_id, user_id),
    )
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadSummary(
        thread_id      = row["thread_id"],
        title          = row.get("title"),
        turn_count     = row.get("turn_count", 0),
        intent_history = list(row.get("intent_history") or []),
        updated_at     = str(row["updated_at"]) if row.get("updated_at") else None,
    )


@router.get("/v1/memories", response_model=list[MemoryRecord],
            summary="List long-term memories for a user")
async def list_memories(
    db: DbDep,
    user_id: str = Query(default="default"),
    limit:   int = Query(default=20, ge=1, le=100),
):
    rows = db.fetch_all(
        """
        SELECT content, importance, thread_id, created_at
        FROM agent_memories
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (user_id, limit),
    )
    return [
        MemoryRecord(
            content    = r["content"],
            importance = float(r["importance"]),
            thread_id  = r["thread_id"],
            created_at = str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in rows
    ]
