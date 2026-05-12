#!/usr/bin/env python3
"""
run_agent.py — Interactive CLI for the LangGraph Agent.

Usage:
    python scripts/run_agent.py
    python scripts/run_agent.py --thread my-session-1
    python scripts/run_agent.py --user alice --thread project-alpha

Features:
  - Persistent conversation via LangGraph checkpointer (MemorySaver)
  - Long-term memory recall shown before each answer
  - Intent classification displayed per turn
  - ReAct steps printed when planner is used
  - /history   — show conversation history
  - /memories  — list saved long-term memories
  - /threads   — list past conversation threads
  - /new       — start a new thread
  - /quit      — exit
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"
DIM    = "\033[2m"
MAGENTA = "\033[35m"

# Intent → colour
INTENT_COLORS = {
    "financial":  GREEN,
    "comparison": CYAN,
    "technical":  BLUE,
    "market":     YELLOW,
    "summary":    MAGENTA,
    "complex":    YELLOW,
    "general":    DIM,
}


def _color_intent(intent: str) -> str:
    color = INTENT_COLORS.get(intent, "")
    return f"{color}{BOLD}{intent.upper()}{RESET}"


def print_banner(thread_id: str, user_id: str) -> None:
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  🤖 Enterprise RAG Agent  |  LangGraph + BGE + Hybrid Search{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"  Thread : {CYAN}{thread_id}{RESET}")
    print(f"  User   : {user_id}")
    print(f"  Commands: /history  /memories  /threads  /new  /quit")
    print(f"{BOLD}{'='*70}{RESET}\n")


def print_react_steps(steps: list[dict]) -> None:
    if not steps:
        return
    print(f"\n{DIM}── ReAct trace ────────────────────────────────{RESET}")
    for i, step in enumerate(steps, 1):
        thought = step.get("thought", "").strip()
        action  = step.get("action", "")
        inp     = step.get("action_input", {})
        obs     = step.get("observation", "")[:200]
        if thought:
            print(f"{DIM}[{i}] Thought: {thought[:120]}{RESET}")
        print(f"{DIM}    → {YELLOW}{action}{RESET}{DIM}({inp}){RESET}")
        print(f"{DIM}    ← {obs}...{RESET}")
    print()


def handle_slash_command(cmd: str, memory, thread_id: str, user_id: str) -> str | None:
    """Handle /commands. Returns new thread_id if /new, else None."""
    cmd = cmd.strip().lower()

    if cmd == "/quit" or cmd == "/exit":
        print("Goodbye!")
        sys.exit(0)

    elif cmd == "/history":
        print(f"\n{BOLD}Conversation history is managed by LangGraph checkpointer.{RESET}")
        print("Use /threads to see past sessions.\n")

    elif cmd == "/memories":
        try:
            from rag_assistant.core import Database
            from rag_assistant.agent.memory import LongTermMemory
            db  = Database()
            mem = LongTermMemory(db)
            rows = db.fetch_all(
                "SELECT content, importance, created_at FROM agent_memories "
                "WHERE user_id = %s ORDER BY created_at DESC LIMIT 10",
                (user_id,),
            )
            if rows:
                print(f"\n{BOLD}Long-term memories ({user_id}):{RESET}")
                for r in rows:
                    print(f"  [{r['importance']:.1f}] {r['content'][:100]}")
                    print(f"        {DIM}{r['created_at']}{RESET}")
            else:
                print(f"\n{DIM}No memories yet for user '{user_id}'.{RESET}\n")
        except Exception as e:
            print(f"Error: {e}")

    elif cmd == "/threads":
        try:
            from rag_assistant.core import Database
            from rag_assistant.agent.memory import LongTermMemory
            db  = Database()
            mem = LongTermMemory(db)
            threads = mem.list_threads(user_id=user_id, limit=10)
            if threads:
                print(f"\n{BOLD}Recent threads:{RESET}")
                for t in threads:
                    title  = t.get("title", "")[:60] or "—"
                    turns  = t.get("turn_count", 0)
                    intents = ", ".join((t.get("intent_history") or [])[-3:])
                    print(f"  {CYAN}{t['thread_id']}{RESET}  [{turns} turns]  {title}")
                    if intents:
                        print(f"       intents: {DIM}{intents}{RESET}")
            else:
                print(f"\n{DIM}No threads yet.{RESET}\n")
        except Exception as e:
            print(f"Error: {e}")

    elif cmd == "/new":
        new_id = str(uuid.uuid4())[:8]
        print(f"\n{GREEN}New thread: {new_id}{RESET}\n")
        return new_id

    else:
        print(f"{DIM}Unknown command: {cmd}{RESET}")

    return None


def run_interactive(thread_id: str, user_id: str) -> None:
    print_banner(thread_id, user_id)

    # Build graph (connects to DB, loads models lazily)
    print(f"{DIM}Initialising agent...{RESET}", end=" ", flush=True)
    from rag_assistant.core import Database
    from rag_assistant.agent.graph import build_agent_graph, run_query
    from rag_assistant.agent.cache import SemanticCache

    db    = Database()
    graph = build_agent_graph(db)
    cache = SemanticCache(db, threshold=0.95)
    print(f"{GREEN}ready{RESET}")
    print(f"  {DIM}Semantic cache: ON  (threshold={cache.threshold})  "
          f"Commands: /cache{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}You>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            if user_input.strip().lower() == "/cache":
                if cache is not None:
                    s = cache.stats()
                    print(f"\n{BOLD}Semantic Cache Stats{RESET}")
                    print(f"  Session hits   : {GREEN}{s['session_hits']}{RESET}")
                    print(f"  Session misses : {s['session_misses']}")
                    print(f"  Session rate   : {BOLD}{s['session_hit_rate']}{RESET}")
                    print(f"  DB total entries: {s['db_total_entries']}")
                    print(f"  DB total hits  : {s['db_total_hits']}")
                    print(f"  Threshold      : {s['threshold']}\n")
                else:
                    print(f"{DIM}Cache is disabled.{RESET}")
                continue
            new_id = handle_slash_command(user_input, None, thread_id, user_id)
            if new_id:
                thread_id = new_id
                print_banner(thread_id, user_id)
            continue

        # Run the agent
        print(f"\n{DIM}Thinking...{RESET}", flush=True)
        try:
            result = run_query(graph, user_input, thread_id=thread_id,
                               user_id=user_id, cache=cache)
        except Exception as e:
            print(f"{YELLOW}Agent error: {e}{RESET}\n")
            continue

        # ── display results ────────────────────────────────────────────────
        intent      = result.get("intent", "general")
        reason      = result.get("intent_reason", "")
        memory_used = result.get("memory_used", False)
        steps       = result.get("react_steps", [])
        answer      = result.get("answer", "")
        sources     = result.get("sources", [])
        latency     = result.get("latency_ms", 0)
        cache_hit   = result.get("cache_hit", False)
        similarity  = result.get("cache_similarity")

        # Cache hit badge (shown instead of intent when cache hits)
        if cache_hit:
            hit_count = result.get("cache_hit_count", 1)
            print(f"\n  {GREEN}{BOLD}⚡ Cache HIT{RESET}  "
                  f"{DIM}similarity={similarity:.4f}  "
                  f"hit #{hit_count}  "
                  f"(original latency: {latency:.0f} ms){RESET}")
        else:
            # Intent badge
            print(f"\n  Intent: {_color_intent(intent)}", end="")
            if reason:
                print(f"  {DIM}({reason}){RESET}", end="")
            print()

        # Memory recall indicator
        if memory_used:
            print(f"  {MAGENTA}✦ Long-term memory recalled{RESET}")

        # ReAct trace (for complex/general intents)
        if steps and not cache_hit:
            print_react_steps(steps)

        # Final answer
        print(f"\n{BOLD}Agent>{RESET} {answer}")

        # Sources + latency
        if sources:
            src_str = " · ".join(sources[:4])
            print(f"\n  {DIM}Sources: {src_str}{RESET}")

        if cache_hit:
            print(f"  {GREEN}⚡ cache  (skipped pipeline){RESET}\n")
        else:
            print(f"  {DIM}⏱  {latency:.0f} ms{RESET}\n")
        print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RAG Agent interactive CLI")
    parser.add_argument("--thread", default=None,
                        help="Thread ID for conversation persistence (default: random)")
    parser.add_argument("--user",   default="default",
                        help="User ID for long-term memory (default: 'default')")
    args = parser.parse_args()

    thread_id = args.thread or str(uuid.uuid4())[:8]
    run_interactive(thread_id=thread_id, user_id=args.user)


if __name__ == "__main__":
    main()
