"""
rag_assistant.agent
===================
LangGraph-based Agent layer on top of the RAG core.

Components:
  state   — AgentState TypedDict (shared across all graph nodes)
  memory  — LongTermMemory: pgvector-backed cross-session recall + save
  tools   — @tool functions: retrieve_docs, compare_companies, extract_financials
  nodes   — LangGraph node functions (recall, classify, specialized agents, ReAct)
  graph   — build_agent_graph() → compiled LangGraph with MemorySaver checkpointer
"""

from .graph import build_agent_graph
from .memory import LongTermMemory
from .state import AgentState, Intent

__all__ = ["build_agent_graph", "LongTermMemory", "AgentState", "Intent"]
