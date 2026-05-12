"""
tools.py — LangChain @tool functions for the ReAct agent.

Each tool wraps the existing RAG core (RetrievalService / AnswerResult).
Tools are stateless functions; shared service/db instances are injected
via module-level globals set by graph.py at startup.

Available tools:
  retrieve_docs       — hybrid RAG retrieval for any query
  compare_companies   — retrieve for two companies and return side-by-side chunks
  extract_financials  — targeted financial metric retrieval
  summarize_document  — get all top chunks from a single document
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from rag_assistant.core import RetrievalService

# These are set by graph.py before the graph is compiled
_service: "RetrievalService | None" = None


def set_service(service: "RetrievalService") -> None:
    global _service
    _service = service


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_chunks(chunks: list[dict], max_chunks: int = 5) -> str:
    if not chunks:
        return "No relevant content found."
    parts = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        title = c.get("document_title", "Unknown")
        p_start = c.get("page_start", "?")
        p_end = c.get("page_end") or p_start
        text = c.get("text", "")[:600]
        parts.append(f"[{i}] {title} p.{p_start}-{p_end}\n{text}")
    return "\n\n".join(parts)


def _run_rag(query: str, top_k: int = 6, doc_filter: str = "") -> tuple[list[dict], str]:
    """Call RAG and return (chunks, formatted_text)."""
    assert _service is not None, "RetrievalService not initialised"
    old_backend = os.environ.get("RAG_LLM_BACKEND")
    os.environ["RAG_LLM_BACKEND"] = "none"  # skip LLM, retrieval only
    try:
        result = _service.answer(query, top_k=top_k)
        chunks = list(result.retrieved_chunks)
        # apply optional document filter
        if doc_filter:
            chunks = [c for c in chunks if doc_filter.lower() in c.get("document_title", "").lower()]
        return chunks, _format_chunks(chunks)
    finally:
        if old_backend is None:
            os.environ.pop("RAG_LLM_BACKEND", None)
        else:
            os.environ["RAG_LLM_BACKEND"] = old_backend


# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def retrieve_docs(query: str, top_k: int = 6) -> str:
    """
    Retrieve the most relevant document chunks for any query using hybrid
    BM25 + vector search with BGE reranking.

    Use this as your primary knowledge retrieval tool.
    Returns ranked text passages with source document and page numbers.

    Args:
        query:  The search query in English or Chinese.
        top_k:  Number of chunks to return (default 6, max 10).
    """
    top_k = min(int(top_k), 10)
    _, formatted = _run_rag(query, top_k=top_k)
    return formatted


@tool
def compare_companies(company_a: str, company_b: str, aspect: str) -> str:
    """
    Retrieve relevant passages for two companies and return them side-by-side
    to support a comparison analysis.

    Use this when the user asks to compare two companies (e.g. CATL vs BYD,
    Tesla vs Vestas) on a specific aspect like revenue, technology, or market.

    Args:
        company_a:  First company name (e.g. "CATL", "宁德时代").
        company_b:  Second company name (e.g. "BYD", "比亚迪").
        aspect:     What to compare (e.g. "revenue 2023", "battery technology").
    """
    query_a = f"{company_a} {aspect}"
    query_b = f"{company_b} {aspect}"

    chunks_a, text_a = _run_rag(query_a, top_k=4, doc_filter=company_a)
    chunks_b, text_b = _run_rag(query_b, top_k=4, doc_filter=company_b)

    return (
        f"=== {company_a} — {aspect} ===\n{text_a}\n\n"
        f"=== {company_b} — {aspect} ===\n{text_b}"
    )


@tool
def extract_financials(company: str, metric: str, year: str = "") -> str:
    """
    Extract a specific financial metric for a company from annual reports.

    Use this for precise financial queries like revenue, net profit, EBIT,
    gross margin, R&D spend, capex, etc.

    Args:
        company:  Company name (e.g. "Tesla", "Vestas", "宁德时代").
        metric:   Financial metric (e.g. "total revenue", "net income",
                  "EBIT margin", "R&D expenditure").
        year:     Optional fiscal year (e.g. "2023", "2024").
    """
    year_str = f" {year}" if year else ""
    query = f"{company}{year_str} {metric} financial results annual report"
    chunks, text = _run_rag(query, top_k=5, doc_filter=company)
    return text or f"No financial data found for {company} {metric}{year_str}."


@tool
def summarize_document(document_name: str) -> str:
    """
    Retrieve the most representative chunks from a single document to support
    a high-level summary of its content.

    Use this when the user asks to summarise a specific report or document.

    Args:
        document_name:  Full or partial document title
                        (e.g. "Vestas", "IRENA 2024", "Tesla 2023").
    """
    query = f"{document_name} key findings overview summary"
    chunks, text = _run_rag(query, top_k=8, doc_filter=document_name)
    return text or f"No content found for document: {document_name}."


# ── tool registry (used by graph.py) ─────────────────────────────────────────
ALL_TOOLS = [retrieve_docs, compare_companies, extract_financials, summarize_document]
