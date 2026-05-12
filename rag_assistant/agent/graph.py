"""
graph.py — Assemble and compile the LangGraph Agent graph.

Graph topology:
                        ┌─────────────┐
                        │ recall_mem  │  ← fetch long-term memories
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │classify_int │  ← LLM intent classification
                        └──────┬──────┘
                               │  (conditional routing)
               ┌───────────────┼──────────────────┐
               ▼               ▼                  ▼
       ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
       │financial_agent│ │comparison_ag │ │ summary_agent│
       └───────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                  │
               └────────────────┼──────────────────┘
                                │       (also from react_planner)
                         ┌──────▼──────┐
                         │ react_plan  │  ← ReAct loop (tools)
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ save_memory │  ← distil + persist facts
                         └──────┬──────┘
                                │
                              END

Checkpointing: MemorySaver (in-process, thread-safe).
Swap to PostgresSaver for production persistence:
  from langgraph.checkpoint.postgres import PostgresSaver
  checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
"""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    classify_intent_node,
    comparison_agent_node,
    financial_agent_node,
    react_planner_node,
    recall_memory_node,
    route_by_intent,
    save_memory_node,
    summary_agent_node,
)
from .state import AgentState, Intent
from .tools import ALL_TOOLS, set_service

if TYPE_CHECKING:
    from rag_assistant.core import Database


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm(with_tools: bool = False):
    """
    Build the LLM client. Priority:
      1. Claude API (ANTHROPIC_API_KEY in env)
      2. Ollama local (OLLAMA_MODEL, default qwen2.5:7b-instruct)

    Returns a plain LLM; if with_tools=True returns an LLM bound to ALL_TOOLS
    for the ReAct tool-calling loop.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if anthropic_key and anthropic_key.startswith("sk-"):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
            max_tokens=2048,
        )
    else:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
        )

    if with_tools:
        return llm.bind_tools(ALL_TOOLS)
    return llm


# ── graph builder ──────────────────────────────────────────────────────────────

def build_agent_graph(db: "Database"):
    """
    Build and compile the full agent graph.

    Returns a CompiledGraph that can be invoked as:
        graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "my-thread-1"}},
        )
    """
    from rag_assistant.core import RetrievalService
    from .memory import LongTermMemory

    # Shared service instances
    service = RetrievalService(db)
    memory  = LongTermMemory(db)

    # Register service with tools module (tools are stateless, inject here)
    set_service(service)

    # LLM instances
    llm            = _build_llm(with_tools=False)
    llm_with_tools = _build_llm(with_tools=True)

    # ── build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("recall_memory",    partial(recall_memory_node,    memory=memory))
    builder.add_node("classify_intent",  partial(classify_intent_node,  llm=llm))
    builder.add_node("financial_agent",  partial(financial_agent_node,  service=service, llm=llm))
    builder.add_node("comparison_agent", partial(comparison_agent_node, llm=llm))
    builder.add_node("summary_agent",    partial(summary_agent_node,    service=service, llm=llm))
    builder.add_node("react_planner",    partial(react_planner_node,    llm_with_tools=llm_with_tools, llm=llm))
    builder.add_node("save_memory",      partial(save_memory_node,      memory=memory, llm=llm))

    # Edges: linear start
    builder.add_edge(START, "recall_memory")
    builder.add_edge("recall_memory", "classify_intent")

    # Conditional routing after intent classification
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "financial_agent":  "financial_agent",
            "comparison_agent": "comparison_agent",
            "summary_agent":    "summary_agent",
            "react_planner":    "react_planner",
        },
    )

    # All agent nodes → save_memory → END
    for agent_node in ["financial_agent", "comparison_agent", "summary_agent", "react_planner"]:
        builder.add_edge(agent_node, "save_memory")
    builder.add_edge("save_memory", END)

    # ── checkpointer (session persistence) ───────────────────────────────────
    # MemorySaver: in-process, suitable for dev and single-instance deployments.
    # For multi-instance / durable execution, swap with:
    #   from langgraph.checkpoint.postgres import PostgresSaver
    #   checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ── convenience: run a single query ──────────────────────────────────────────

def run_query(
    graph,
    question: str,
    thread_id: str = "default",
    user_id: str = "default",
    cache=None,          # optional SemanticCache instance
) -> dict:
    """
    Invoke the compiled graph for a single question.

    If a SemanticCache is provided:
      - HIT  → return cached answer in ~5-20 ms (skip entire pipeline)
      - MISS → run full pipeline, persist result to cache

    Returns a dict with: answer, intent, intent_reason, react_steps,
                         sources, latency_ms, memory_used, cache_hit,
                         cache_similarity (on hit).
    """
    from langchain_core.messages import HumanMessage

    # ── cache lookup ──────────────────────────────────────────────────────────
    if cache is not None:
        entry = cache.get(question)
        if entry is not None:
            return {
                "answer":           entry.answer,
                "intent":           entry.intent,
                "intent_reason":    "(cache hit)",
                "react_steps":      entry.react_steps,
                "sources":          entry.sources,
                "latency_ms":       entry.latency_ms,   # original pipeline latency
                "memory_used":      False,
                "cache_hit":        True,
                "cache_similarity": entry.similarity,
                "cache_hit_count":  entry.hit_count,
            }

    # ── full pipeline ─────────────────────────────────────────────────────────
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages":    [HumanMessage(content=question)],
            "thread_id":   thread_id,
            "user_id":     user_id,
            "iterations":  0,
            "react_steps": [],
        },
        config=config,
    )

    output = {
        "answer":           result.get("final_answer", ""),
        "intent":           result.get("intent", "general"),
        "intent_reason":    result.get("intent_reason", ""),
        "react_steps":      result.get("react_steps", []),
        "sources":          result.get("sources", []),
        "latency_ms":       result.get("latency_ms", 0),
        "memory_used":      bool(result.get("memory_context", "")),
        "cache_hit":        False,
        "cache_similarity": None,
    }

    # ── persist to cache ──────────────────────────────────────────────────────
    if cache is not None:
        cache.set(question, output)

    return output
