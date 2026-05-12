"""
memory.py — Long-term memory backed by pgvector.

Architecture
------------
Each memory is a distilled fact extracted from a past conversation turn.
On session end, key facts are LLM-summarised → embedded → stored in agent_memories.
On new query, top-k semantically similar memories are recalled and injected into prompt.

This is separate from LangGraph's checkpointer (which stores the full graph state
for session resumption). Long-term memory crosses session boundaries and is
retrieved semantically, not replayed verbatim.

Table: agent_memories (see sql/004_agent_memory.sql)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_assistant.core import Database


@dataclass
class MemoryRecord:
    id: str
    content: str
    importance: float
    thread_id: str
    similarity: float = 0.0


class LongTermMemory:
    """Semantic memory store using pgvector cosine similarity."""

    def __init__(self, db: "Database", embed_model: str = "BAAI/bge-m3") -> None:
        self.db = db
        self.embed_model = embed_model
        self._model = None  # lazy-loaded SentenceTransformer

    # ── embedding ─────────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.embed_model, device=device)
        return self._model

    def _embed(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _vec_str(self, vec: list[float]) -> str:
        return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

    # ── recall ────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 4,
        min_similarity: float = 0.60,
    ) -> list[MemoryRecord]:
        """Return top-k memories relevant to the query."""
        vec = self._embed(query)
        vec_str = self._vec_str(vec)

        rows = self.db.fetch_all(
            """
            SELECT
                id::text,
                content,
                importance,
                thread_id,
                1 - (embedding <=> %s::vector) AS similarity
            FROM agent_memories
            WHERE user_id = %s
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY (importance * (1 - (embedding <=> %s::vector))) DESC
            LIMIT %s
            """,
            (vec_str, user_id, vec_str, min_similarity, vec_str, top_k),
        )

        # update access stats
        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ", ".join(f"'{i}'" for i in ids)
            self.db.execute(
                f"""
                UPDATE agent_memories
                SET last_accessed = NOW(), access_count = access_count + 1
                WHERE id::text IN ({placeholders})
                """
            )

        return [
            MemoryRecord(
                id=r["id"],
                content=r["content"],
                importance=r["importance"],
                thread_id=r["thread_id"],
                similarity=float(r["similarity"]),
            )
            for r in rows
        ]

    def recall_as_context(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 4,
    ) -> str:
        """Return recalled memories formatted as a prompt context block."""
        memories = self.recall(query, user_id=user_id, top_k=top_k)
        if not memories:
            return ""
        lines = ["[Long-term memory — from past conversations]"]
        for m in memories:
            lines.append(f"- {m.content}  (relevance: {m.similarity:.2f})")
        return "\n".join(lines)

    # ── save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        content: str,
        thread_id: str,
        user_id: str = "default",
        importance: float = 0.5,
        source_turn: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Embed and persist one memory. Returns the new memory id."""
        vec = self._embed(content)
        vec_str = self._vec_str(vec)
        mem_id = str(uuid.uuid4())

        self.db.execute(
            """
            INSERT INTO agent_memories
                (id, thread_id, user_id, content, embedding, importance,
                 source_turn, metadata)
            VALUES
                (%s, %s, %s, %s, %s::vector, %s, %s, %s)
            """,
            (
                mem_id,
                thread_id,
                user_id,
                content,
                vec_str,
                importance,
                source_turn,
                json.dumps(metadata or {}),
            ),
        )
        return mem_id

    def save_batch(
        self,
        facts: list[str],
        thread_id: str,
        user_id: str = "default",
        importance: float = 0.5,
    ) -> list[str]:
        """Save multiple memory facts at once. Returns list of ids."""
        return [
            self.save(fact, thread_id=thread_id, user_id=user_id, importance=importance)
            for fact in facts
            if fact.strip()
        ]

    # ── LLM-assisted extraction ───────────────────────────────────────────────

    def extract_and_save(
        self,
        conversation_text: str,
        thread_id: str,
        user_id: str = "default",
        llm=None,
    ) -> list[str]:
        """
        Ask an LLM to extract key facts from a conversation, then save them.
        Falls back to a simple heuristic if no LLM is provided.
        """
        if llm is not None:
            from langchain_core.messages import HumanMessage, SystemMessage
            system = SystemMessage(content=(
                "You are a memory extraction assistant. "
                "Extract 2-4 distinct, concrete facts from the conversation that "
                "would be useful to remember for future interactions. "
                "Focus on: specific numbers, company names, user preferences, "
                "key conclusions. "
                "Return ONLY a JSON array of strings, e.g. "
                '[\"CATL 2023 revenue was 402.7 billion CNY\", ...]'
            ))
            human = HumanMessage(content=f"Conversation:\n{conversation_text}")
            try:
                response = llm.invoke([system, human])
                text = response.content.strip()
                # strip markdown code block if present
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                facts = json.loads(text)
                if isinstance(facts, list):
                    return self.save_batch(facts, thread_id, user_id, importance=0.7)
            except Exception:
                pass  # fall through to heuristic

        # Heuristic fallback: save the last assistant message as-is
        lines = [ln.strip() for ln in conversation_text.splitlines() if ln.strip()]
        # find lines that look like assistant answers
        facts = [ln for ln in lines if len(ln) > 40][:3]
        return self.save_batch(facts, thread_id, user_id, importance=0.4)

    # ── thread management ─────────────────────────────────────────────────────

    def upsert_thread(
        self,
        thread_id: str,
        user_id: str = "default",
        title: str | None = None,
        intent: str | None = None,
    ) -> None:
        """Create or update a conversation thread record."""
        self.db.execute(
            """
            INSERT INTO conversation_threads (thread_id, user_id, title, turn_count, updated_at)
            VALUES (%s, %s, %s, 1, NOW())
            ON CONFLICT (thread_id) DO UPDATE
                SET turn_count  = conversation_threads.turn_count + 1,
                    updated_at  = NOW(),
                    title       = COALESCE(%s, conversation_threads.title),
                    intent_history = CASE
                        WHEN %s IS NOT NULL
                        THEN array_append(conversation_threads.intent_history, %s)
                        ELSE conversation_threads.intent_history
                    END
            """,
            (thread_id, user_id, title, title, intent, intent),
        )

    def list_threads(self, user_id: str = "default", limit: int = 10) -> list[dict]:
        rows = self.db.fetch_all(
            """
            SELECT thread_id, title, turn_count, intent_history, updated_at
            FROM conversation_threads
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        return [dict(r) for r in rows]
