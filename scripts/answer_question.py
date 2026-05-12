from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json

from rag_assistant.core import RetrievalService, format_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the renewable energy retrieval assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    service = RetrievalService()
    result = service.answer(args.question, top_k=args.top_k)
    if args.json:
        print(
            json.dumps(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "sources": result.sources,
                    "retrieved_chunks": result.retrieved_chunks,
                    "latency_ms": result.latency_ms,
                    "used_llm": result.used_llm,
                    "debug": result.debug,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_answer(result))


if __name__ == "__main__":
    main()

