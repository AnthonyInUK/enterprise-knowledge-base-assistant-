"""
routers/research.py

Direct research workbench endpoint: returns grounded answer plus retrieved
evidence chunks for UI inspection.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from rag_assistant.core import RetrievalService
from rag_assistant.facts import extract_fact_candidates, persist_fact_candidates

from ..deps import DbDep
from ..models import ChatRequest, ResearchAnswerResponse, RetrievedChunkSchema

router = APIRouter(prefix="/v1/research", tags=["research"])


@router.post("/ask", response_model=ResearchAnswerResponse, summary="Ask with evidence")
async def ask(req: ChatRequest, db: DbDep):
    service = RetrievalService(db)
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: service.answer(req.question, top_k=req.top_k)
    )
    chunks = list(result.retrieved_chunks)
    fact_candidates = extract_fact_candidates(
        result.question, result.answer, result.sources, chunks
    )
    facts = persist_fact_candidates(db, fact_candidates) if fact_candidates else []
    return ResearchAnswerResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        retrieved_chunks=[
            RetrievedChunkSchema(**chunk) for chunk in chunks
        ],
        latency_ms=result.latency_ms,
        used_llm=result.used_llm,
        debug=result.debug,
        facts=facts,
    )
