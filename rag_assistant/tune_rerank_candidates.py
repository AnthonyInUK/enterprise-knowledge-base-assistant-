from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time
from typing import Any

from rag_assistant.eval_runner import run_eval


def tune_candidates(eval_set_name: str, candidates: list[int], top_k: int, backend: str, recall_weight: float) -> dict[str, Any]:
    old_env = {key: os.environ.get(key) for key in [
        "RAG_ENABLE_RERANK",
        "RAG_RERANK_BACKEND",
        "RAG_RERANK_CANDIDATES",
        "RAG_RERANK_RECALL_WEIGHT",
    ]}
    rows: list[dict[str, Any]] = []
    try:
        os.environ["RAG_ENABLE_RERANK"] = "1"
        os.environ["RAG_RERANK_BACKEND"] = backend
        os.environ["RAG_RERANK_RECALL_WEIGHT"] = str(recall_weight)
        for candidate_k in candidates:
            os.environ["RAG_RERANK_CANDIDATES"] = str(candidate_k)
            started = time.perf_counter()
            summary = run_eval(eval_set_name=eval_set_name, top_k=top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "candidate_k": candidate_k,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "elapsed_per_case_ms": round(elapsed_ms / max(1, summary["count"]), 2),
                    "retrieval_mean": summary["retrieval_mean"],
                    "answer_mean": summary["answer_mean"],
                    "citation_mean": summary["citation_mean"],
                    "pass_rate": summary["pass_rate"],
                    "run_id": summary["run_id"],
                }
            )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"eval_set": eval_set_name, "top_k": top_k, "backend": backend, "recall_weight": recall_weight, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune rerank candidate count against eval metrics and latency.")
    parser.add_argument("--eval-set", default="energy_starter_v1")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates", default="20,30,50")
    parser.add_argument("--backend", default="bge")
    parser.add_argument("--recall-weight", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    candidates = [int(item.strip()) for item in args.candidates.split(",") if item.strip()]
    summary = tune_candidates(args.eval_set, candidates, args.top_k, args.backend, args.recall_weight)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(f"eval set: {summary['eval_set']}")
    print(f"backend: {summary['backend']}")
    print(f"top_k: {summary['top_k']}")
    print(f"recall_weight: {summary['recall_weight']}")
    for row in summary["results"]:
        print(
            f"- candidates={row['candidate_k']} elapsed={row['elapsed_ms']:.1f}ms "
            f"per_case={row['elapsed_per_case_ms']:.1f}ms retrieval={row['retrieval_mean']:.4f} "
            f"answer={row['answer_mean']:.4f} citation={row['citation_mean']:.4f} pass={row['pass_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
