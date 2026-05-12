"""
state.py — AgentState shared across all LangGraph nodes.

Design principles:
  - All fields are optional / have defaults so any node can be the entry point.
  - `messages` uses LangGraph's add_messages reducer (append-only, thread-safe).
  - `intent` drives the conditional routing after classification.
  - `react_steps` records Thought/Action/Observation for debugging + interview demo.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class Intent(str, Enum):
    FINANCIAL   = "financial"    # revenue, profit, EBIT, margin
    COMPARISON  = "comparison"   # compare company A vs B
    TECHNICAL   = "technical"    # tech stack, product, roadmap
    MARKET      = "market"       # market share, regions, overseas
    SUMMARY     = "summary"      # summarize a document
    COMPLEX     = "complex"      # multi-step, needs ReAct planning
    GENERAL     = "general"      # catch-all


class ReActStep(TypedDict):
    thought:     str
    action:      str             # tool name
    action_input: Any
    observation: str


class AgentState(TypedDict, total=False):
    # ── conversation ──────────────────────────────────────────────────────────
    messages:        Annotated[list[BaseMessage], add_messages]
    thread_id:       str
    user_id:         str

    # ── intent classification ─────────────────────────────────────────────────
    intent:          str          # Intent enum value
    intent_reason:   str          # LLM's one-line explanation

    # ── long-term memory ──────────────────────────────────────────────────────
    memory_context:  str          # recalled memories injected into prompt

    # ── retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks: list[dict]  # raw chunk dicts from RAG core

    # ── ReAct planning ────────────────────────────────────────────────────────
    react_steps:     list[ReActStep]
    iterations:      int          # guard against infinite loops (max=5)

    # ── final output ──────────────────────────────────────────────────────────
    final_answer:    str
    sources:         list[str]    # cited document titles
    latency_ms:      float
