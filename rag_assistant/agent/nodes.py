"""
nodes.py — LangGraph node functions.

Each node is a pure function: AgentState → dict (partial state update).
Nodes are wired together in graph.py.

Node pipeline:
  recall_memory → classify_intent → [route] → agent_node → save_memory

Routing:
  financial  → financial_agent   (direct RAG + structured prompt)
  comparison → comparison_agent  (dual retrieval, side-by-side)
  summary    → summary_agent     (full-doc retrieval + long-form summary)
  technical / market / general → react_planner  (ReAct loop with tools)
  complex    → react_planner
"""

from __future__ import annotations

import json
import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .state import AgentState, Intent

if TYPE_CHECKING:
    from rag_assistant.agent.memory import LongTermMemory
    from rag_assistant.core import RetrievalService

MAX_REACT_ITERATIONS = 5


# ── helpers ───────────────────────────────────────────────────────────────────

def _last_human_message(state: AgentState) -> str:
    """Extract the most recent HumanMessage content from state."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _resolve_query(question: str, state: AgentState) -> str:
    """
    Expand pronoun-heavy follow-up questions using recent conversation context.
    e.g. "它的研发费用" → "宁德时代 研发费用" if previous turn was about CATL.
    """
    pronouns = {"它", "该公司", "这家公司", "its", "their", "the company", "they"}
    has_pronoun = any(p in question for p in pronouns)
    if not has_pronoun:
        return question

    # Find the last AI answer to extract subject
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # pull first ~80 chars as topic hint
            hint = msg.content[:80].split("\n")[0]
            return f"{hint} — {question}"
    return question


def _build_rag_prompt(
    question: str,
    context: str,
    memory_context: str = "",
    system_prefix: str = "",
) -> list:
    system_text = system_prefix or (
        "You are an expert research analyst for the renewable energy industry. "
        "Answer the question using ONLY the provided document excerpts. "
        "Always cite the source document and page number. "
        "If the answer is not in the excerpts, say so clearly."
    )
    if memory_context:
        system_text += f"\n\n{memory_context}"

    return [
        SystemMessage(content=system_text),
        HumanMessage(content=(
            f"Document excerpts:\n{context}\n\n"
            f"Question: {question}"
        )),
    ]


# ── node: recall_memory ───────────────────────────────────────────────────────

def recall_memory_node(state: AgentState, memory: "LongTermMemory") -> dict:
    """Fetch relevant long-term memories and inject as context."""
    question = _last_human_message(state)
    user_id  = state.get("user_id", "default")

    memory_context = ""
    if question:
        try:
            memory_context = memory.recall_as_context(question, user_id=user_id)
        except Exception:
            pass  # DB might not have the table yet; degrade gracefully

    return {"memory_context": memory_context}


# ── node: classify_intent ─────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are an intent classifier for an enterprise knowledge-base assistant covering \
the renewable energy industry (solar, wind, batteries, EVs).

Classify the user's question into exactly ONE of these intents:
  financial   — revenue, profit, EBIT, margin, earnings, financial results
  comparison  — compare two companies or products
  technical   — technology, product specs, R&D, patents, roadmap
  market      — market share, regions, overseas expansion, customers
  summary     — summarise a document or report
  complex     — multi-step question requiring several lookups or calculations
  general     — anything else

Respond with a JSON object only:
{"intent": "<one of the intents above>", "reason": "<one sentence>"}
"""


def classify_intent_node(state: AgentState, llm) -> dict:
    """Use LLM to classify the user's intent."""
    question = _last_human_message(state)
    try:
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=question),
        ])
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        data = json.loads(text)
        intent = data.get("intent", Intent.GENERAL).lower()
        reason = data.get("reason", "")
    except Exception as e:
        intent = Intent.GENERAL
        reason = f"classification failed: {e}"

    return {"intent": intent, "intent_reason": reason}


# ── routing edge function ─────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """Conditional edge: returns next node name based on classified intent."""
    intent = state.get("intent", Intent.GENERAL)
    routing = {
        Intent.FINANCIAL:  "financial_agent",
        Intent.COMPARISON: "comparison_agent",
        Intent.SUMMARY:    "summary_agent",
        Intent.TECHNICAL:  "react_planner",
        Intent.MARKET:     "react_planner",
        Intent.COMPLEX:    "react_planner",
        Intent.GENERAL:    "react_planner",
    }
    return routing.get(intent, "react_planner")


# ── node: financial_agent ─────────────────────────────────────────────────────

def financial_agent_node(state: AgentState, service: "RetrievalService", llm) -> dict:
    """Specialised agent for financial metric questions — direct RAG + structured prompt."""
    question       = _last_human_message(state)
    resolved_query = _resolve_query(question, state)   # expand pronouns
    memory_context = state.get("memory_context", "")

    t0 = time.perf_counter()
    # retrieval using resolved query for better recall
    old = os.environ.get("RAG_LLM_BACKEND")
    os.environ["RAG_LLM_BACKEND"] = "none"
    try:
        result = service.answer(resolved_query, top_k=6)
        chunks = list(result.retrieved_chunks)
    finally:
        if old is None:
            os.environ.pop("RAG_LLM_BACKEND", None)
        else:
            os.environ["RAG_LLM_BACKEND"] = old

    context = "\n\n".join(
        f"[{c.get('document_title','?')} p.{c.get('page_start','?')}]\n{c.get('text','')[:500]}"
        for c in chunks[:6]
    )

    system_prefix = (
        "You are a financial analyst specialising in renewable energy companies. "
        "Extract the exact numerical answer from the document excerpts. "
        "Format numbers clearly (e.g. '€15,382 million', 'CNY 402.7 billion'). "
        "Cite the source document and page number for each figure."
    )
    messages = _build_rag_prompt(question, context, memory_context, system_prefix)
    response = llm.invoke(messages)

    sources = list({c.get("document_title", "") for c in chunks if c.get("document_title")})
    latency = (time.perf_counter() - t0) * 1000

    return {
        "final_answer":    response.content,
        "retrieved_chunks": chunks,
        "sources":         sources,
        "latency_ms":      latency,
        "messages":        [AIMessage(content=response.content)],
    }


# ── node: comparison_agent ────────────────────────────────────────────────────

def comparison_agent_node(state: AgentState, llm) -> dict:
    """Specialised agent for cross-company comparison — dual retrieval."""
    from .tools import compare_companies

    question      = _last_human_message(state)
    memory_context = state.get("memory_context", "")

    t0 = time.perf_counter()

    # Ask LLM to extract company names + aspect
    extract_prompt = (
        "Extract the two company names and the comparison aspect from this question. "
        'Reply with JSON only: {"company_a": "...", "company_b": "...", "aspect": "..."}\n\n'
        f"Question: {question}"
    )
    try:
        extraction = llm.invoke([HumanMessage(content=extract_prompt)])
        text = extraction.content.strip().lstrip("```json").rstrip("```")
        params = json.loads(text)
    except Exception:
        params = {"company_a": "", "company_b": "", "aspect": question}

    context = compare_companies.invoke(params)

    system_prefix = (
        "You are a research analyst comparing renewable energy companies. "
        "Use the retrieved excerpts to give a structured, balanced comparison. "
        "Use a table or bullet points where helpful. Cite sources."
    )
    messages = _build_rag_prompt(question, context, memory_context, system_prefix)
    response = llm.invoke(messages)
    latency  = (time.perf_counter() - t0) * 1000

    # Build sources from both companies' names (avoid inheriting stale state)
    sources = []
    for name in [params.get("company_a", ""), params.get("company_b", "")]:
        if name:
            sources.append(name)

    return {
        "final_answer":    response.content,
        "retrieved_chunks": [],   # reset to avoid stale chunks from prior turn
        "sources":         sources,
        "latency_ms":      latency,
        "messages":        [AIMessage(content=response.content)],
    }


# ── node: summary_agent ───────────────────────────────────────────────────────

def summary_agent_node(state: AgentState, service: "RetrievalService", llm) -> dict:
    """Specialised agent for document summarisation."""
    from .tools import summarize_document

    question      = _last_human_message(state)
    memory_context = state.get("memory_context", "")

    t0 = time.perf_counter()
    context   = summarize_document.invoke({"document_name": question})
    system_prefix = (
        "You are a research analyst producing an executive summary. "
        "Structure your answer with: Key Findings, Financial Highlights, "
        "Strategic Outlook. Keep it under 300 words. Cite sources."
    )
    messages  = _build_rag_prompt(question, context, memory_context, system_prefix)
    response  = llm.invoke(messages)
    latency   = (time.perf_counter() - t0) * 1000

    return {
        "final_answer": response.content,
        "latency_ms":   latency,
        "messages":     [AIMessage(content=response.content)],
    }


# ── node: react_planner ───────────────────────────────────────────────────────

_REACT_SYSTEM = """\
You are a research analyst with access to a renewable energy knowledge base.
Solve the user's question step by step using the available tools.

Available tools:
  - retrieve_docs(query, top_k): general hybrid RAG retrieval
  - compare_companies(company_a, company_b, aspect): side-by-side comparison
  - extract_financials(company, metric, year): targeted financial lookup
  - summarize_document(document_name): full-doc summary

Think out loud before each action. When you have enough information, write your
final answer starting with "FINAL ANSWER:".
"""


def react_planner_node(state: AgentState, llm_with_tools, llm) -> dict:
    """
    ReAct planning loop: Thought → Action (tool call) → Observation → repeat.
    Uses LangChain tool-calling interface. Max MAX_REACT_ITERATIONS iterations.
    """
    question      = _last_human_message(state)
    memory_context = state.get("memory_context", "")
    iterations    = state.get("iterations", 0)
    react_steps   = list(state.get("react_steps", []))

    t0 = time.perf_counter()

    system_content = _REACT_SYSTEM
    if memory_context:
        system_content += f"\n\n{memory_context}"

    messages: list = [
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ]

    # ── ReAct loop ────────────────────────────────────────────────────────────
    from .tools import ALL_TOOLS
    tool_map = {t.name: t for t in ALL_TOOLS}

    while iterations < MAX_REACT_ITERATIONS:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If LLM called a tool
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_name  = tc["name"]
                tool_input = tc["args"]
                tool_fn    = tool_map.get(tool_name)

                if tool_fn is None:
                    observation = f"Unknown tool: {tool_name}"
                else:
                    try:
                        observation = tool_fn.invoke(tool_input)
                        if len(observation) > 2000:
                            observation = observation[:2000] + "\n...[truncated]"
                    except Exception as e:
                        observation = f"Tool error: {e}"

                react_steps.append({
                    "thought":      getattr(response, "content", ""),
                    "action":       tool_name,
                    "action_input": tool_input,
                    "observation":  observation,
                })
                messages.append(ToolMessage(
                    content=observation,
                    tool_call_id=tc["id"],
                ))

            iterations += 1
            continue

        # No tool calls → LLM has finished
        final_answer = response.content
        break
    else:
        # Hit iteration limit — ask LLM to summarise what it found
        summary_msg = HumanMessage(content=(
            "You've reached the maximum number of tool calls. "
            "Please give your best answer based on what you've found so far."
        ))
        messages.append(summary_msg)
        final_response = llm.invoke(messages)
        final_answer   = final_response.content

    latency = (time.perf_counter() - t0) * 1000

    return {
        "final_answer": final_answer,
        "react_steps":  react_steps,
        "iterations":   iterations,
        "latency_ms":   latency,
        "messages":     [AIMessage(content=final_answer)],
    }


# ── node: save_memory ─────────────────────────────────────────────────────────

def save_memory_node(state: AgentState, memory: "LongTermMemory", llm) -> dict:
    """
    Extract key facts from this conversation turn and persist to long-term memory.
    Also updates the conversation_threads table.
    """
    thread_id = state.get("thread_id", "default")
    user_id   = state.get("user_id", "default")
    question  = _last_human_message(state)
    answer    = state.get("final_answer", "")
    intent    = state.get("intent", "general")

    if not answer:
        return {}

    # Upsert thread metadata
    try:
        memory.upsert_thread(
            thread_id=thread_id,
            user_id=user_id,
            title=question[:80] if question else None,
            intent=intent,
        )
    except Exception:
        pass

    # Extract + save memories (best-effort)
    try:
        convo_text = f"Q: {question}\nA: {answer}"
        memory.extract_and_save(
            conversation_text=convo_text,
            thread_id=thread_id,
            user_id=user_id,
            llm=llm,
        )
    except Exception:
        pass

    return {}
