"""
routers/chat.py

POST /v1/chat          — synchronous, returns full JSON
GET  /v1/chat/stream   — Server-Sent Events, streams pipeline progress + answer

SSE event sequence:
  {"type":"progress", "payload":{"stage":"classifying intent"}}
  {"type":"intent",   "payload":{"intent":"financial","reason":"..."}}
  {"type":"memory",   "payload":{"recalled":true/false}}
  {"type":"progress", "payload":{"stage":"generating answer"}}
  {"type":"token",    "payload":{"text":"word "}}   ← word-by-word
  {"type":"done",     "payload":{"sources":[...],"latency_ms":123,"cache_hit":false}}
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from rag_assistant.agent.graph import run_query

from ..deps import CacheDep, GraphDep
from ..models import ChatRequest, ChatResponse, ReActStepSchema

router = APIRouter(prefix="/v1/chat", tags=["chat"])


# ── POST /v1/chat ─────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse, summary="Ask a question (synchronous)")
async def chat(req: ChatRequest, graph: GraphDep, cache: CacheDep):
    """
    Full RAG + Agent pipeline.  Returns complete answer in one response.
    For streaming progress, use GET /v1/chat/stream instead.
    """
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_query(
            graph,
            req.question,
            thread_id=req.thread_id,
            user_id=req.user_id,
            cache=cache,
        ),
    )

    return ChatResponse(
        answer           = result["answer"],
        intent           = result["intent"],
        intent_reason    = result["intent_reason"],
        sources          = result.get("sources", []),
        react_steps      = [ReActStepSchema(**s) for s in result.get("react_steps", [])],
        latency_ms       = result["latency_ms"],
        cache_hit        = result.get("cache_hit", False),
        cache_similarity = result.get("cache_similarity"),
        thread_id        = req.thread_id,
        user_id          = req.user_id,
    )


# ── GET /v1/chat/stream ───────────────────────────────────────────────────────

def _sse(event_type: str, payload: dict) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps({'type': event_type, 'payload': payload}, ensure_ascii=False)}\n\n"


async def _stream_pipeline(
    graph, cache, question: str, thread_id: str, user_id: str
) -> AsyncGenerator[str, None]:
    """
    Run the agent pipeline and yield SSE events.

    Uses LangGraph's graph.stream() with stream_mode='updates' to get
    real-time node-by-node state updates, then streams the final answer
    word-by-word for a natural reading experience.
    """
    from langchain_core.messages import HumanMessage

    # ── 1. cache check ────────────────────────────────────────────────────────
    yield _sse("progress", {"stage": "checking cache"})
    if cache is not None:
        entry = await asyncio.get_event_loop().run_in_executor(
            None, lambda: cache.get(question)
        )
        if entry is not None:
            yield _sse("intent",   {"intent": entry.intent, "reason": "(cache hit)"})
            yield _sse("memory",   {"recalled": False})
            yield _sse("progress", {"stage": "streaming cached answer"})
            # stream answer word-by-word
            for word in entry.answer.split(" "):
                yield _sse("token", {"text": word + " "})
                await asyncio.sleep(0.01)
            yield _sse("done", {
                "sources":          entry.sources,
                "latency_ms":       entry.latency_ms,
                "cache_hit":        True,
                "cache_similarity": entry.similarity,
                "react_steps":      entry.react_steps,
            })
            return

    # ── 2. full pipeline via LangGraph stream ─────────────────────────────────
    t0 = time.perf_counter()
    yield _sse("progress", {"stage": "classifying intent"})

    config = {"configurable": {"thread_id": thread_id}}
    init_state = {
        "messages":    [HumanMessage(content=question)],
        "thread_id":   thread_id,
        "user_id":     user_id,
        "iterations":  0,
        "react_steps": [],
    }

    final_answer = ""
    sources: list[str] = []
    intent = "general"
    intent_reason = ""
    react_steps: list[dict] = []
    memory_recalled = False

    # LangGraph stream yields (node_name, state_delta) per node completion
    def _run_stream():
        return list(graph.stream(init_state, config=config, stream_mode="updates"))

    node_updates = await asyncio.get_event_loop().run_in_executor(None, _run_stream)

    for node_name, delta in node_updates:
        if node_name == "recall_memory":
            memory_recalled = bool(delta.get("memory_context", ""))
            yield _sse("memory", {"recalled": memory_recalled})

        elif node_name == "classify_intent":
            intent        = delta.get("intent", "general")
            intent_reason = delta.get("intent_reason", "")
            yield _sse("intent", {"intent": intent, "reason": intent_reason})
            yield _sse("progress", {"stage": "retrieving documents"})

        elif node_name in ("financial_agent", "comparison_agent",
                           "summary_agent", "react_planner"):
            final_answer = delta.get("final_answer", "")
            sources      = delta.get("sources", [])
            react_steps  = delta.get("react_steps", [])
            yield _sse("progress", {"stage": "streaming answer"})
            # stream answer word-by-word
            for word in final_answer.split(" "):
                yield _sse("token", {"text": word + " "})
                await asyncio.sleep(0.008)

        elif node_name == "save_memory":
            pass  # silent

    latency = (time.perf_counter() - t0) * 1000

    # persist to cache
    if cache is not None and final_answer:
        result_dict = {
            "answer":      final_answer,
            "intent":      intent,
            "sources":     sources,
            "react_steps": react_steps,
            "latency_ms":  latency,
        }
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: cache.set(question, result_dict)
        )

    yield _sse("done", {
        "sources":     sources,
        "latency_ms":  latency,
        "cache_hit":   False,
        "react_steps": react_steps,
    })


@router.get("/stream", summary="Ask a question (Server-Sent Events)")
async def chat_stream(
    graph: GraphDep,
    cache: CacheDep,
    question:  str = Query(..., min_length=1, max_length=2000),
    thread_id: str = Query(default="default"),
    user_id:   str = Query(default="default"),
):
    """
    Stream the agent pipeline via Server-Sent Events.

    Connect with:
        curl -N "http://localhost:8000/v1/chat/stream?question=What+was+Vestas+revenue"

    Each event is a JSON object with `type` and `payload` fields.
    Event types: progress | intent | memory | token | done | error
    """
    return StreamingResponse(
        _stream_pipeline(graph, cache, question, thread_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
