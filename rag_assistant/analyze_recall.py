from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from statistics import median
from typing import Any

from rag_assistant.core import Database, RetrievalService
from rag_assistant.eval_runner import _page_overlap


def _load_cases(db: Database, eval_set_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_set = db.fetch_one("SELECT id, name FROM eval_sets WHERE name = %s", (eval_set_name,))
    if not eval_set:
        raise RuntimeError(f"Eval set {eval_set_name!r} not found. Run eval_seed first.")

    cases = db.fetch_all(
        """
        SELECT id, question, gold_chunk_ids, metadata
        FROM eval_cases
        WHERE eval_set_id = %s
        ORDER BY created_at NULLS LAST, question
        """,
        (eval_set["id"],),
    )
    if not cases:
        raise RuntimeError(f"Eval set {eval_set_name!r} has no cases.")
    return eval_set, cases


def _nearby_rank(hits: list[Any], metadata: dict[str, Any]) -> int | None:
    gold_title = str(metadata.get("source_title") or "").lower()
    gold_section = str(metadata.get("section_title") or "").lower()
    gold_page_start = metadata.get("page_start")
    gold_page_end = metadata.get("page_end")

    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        same_doc = bool(gold_title and chunk.document_title.lower() == gold_title)
        same_page = _page_overlap(chunk.page_start, chunk.page_end, gold_page_start, gold_page_end)
        same_section = bool(gold_section and (chunk.section_title or "").lower() == gold_section)
        if same_doc and (same_page or same_section):
            return rank
    return None


def _hit_preview(hit: Any | None) -> dict[str, Any] | None:
    if hit is None:
        return None
    return {
        "rank": hit.rank,
        "score": round(hit.score, 4),
        "chunk_id": hit.chunk.id,
        "document_title": hit.chunk.document_title,
        "section_title": hit.chunk.section_title,
        "page_start": hit.chunk.page_start,
        "page_end": hit.chunk.page_end,
        "reason": hit.reason,
        "snippet": " ".join(hit.chunk.text.split())[:180],
    }


def analyze_recall(eval_set_name: str = "energy_starter_v1", top_k: int = 6, candidate_k: int = 50) -> dict[str, Any]:
    db = Database()
    service = RetrievalService(db)
    eval_set, cases = _load_cases(db, eval_set_name)

    rows: list[dict[str, Any]] = []
    for case in cases:
        hits = service.retrieve(case["question"], top_k=candidate_k)
        rank_by_chunk_id = {hit.chunk.id: rank for rank, hit in enumerate(hits, start=1)}
        gold_chunk_ids = [str(chunk_id) for chunk_id in (case.get("gold_chunk_ids") or [])]
        gold_ranks = [rank_by_chunk_id[chunk_id] for chunk_id in gold_chunk_ids if chunk_id in rank_by_chunk_id]
        gold_rank = min(gold_ranks) if gold_ranks else None
        nearby = _nearby_rank(hits, dict(case.get("metadata") or {}))

        if gold_rank is not None and gold_rank <= top_k:
            status = "exact_top_k"
        elif gold_rank is not None:
            status = "rerank_opportunity"
        elif nearby is not None and nearby <= candidate_k:
            status = "nearby_only"
        else:
            status = "recall_failure"

        top_hit = hits[0] if hits else None
        rows.append(
            {
                "question": case["question"],
                "status": status,
                "gold_rank": gold_rank,
                "nearby_rank": nearby,
                "gold_chunk_ids": gold_chunk_ids,
                "gold_source_title": (case.get("metadata") or {}).get("source_title"),
                "gold_section_title": (case.get("metadata") or {}).get("section_title"),
                "gold_page_start": (case.get("metadata") or {}).get("page_start"),
                "gold_page_end": (case.get("metadata") or {}).get("page_end"),
                "top_hit": _hit_preview(top_hit),
            }
        )

    count = len(rows)
    exact_top_k = sum(1 for row in rows if row["gold_rank"] is not None and row["gold_rank"] <= top_k)
    exact_top_10 = sum(1 for row in rows if row["gold_rank"] is not None and row["gold_rank"] <= 10)
    exact_top_candidate = sum(1 for row in rows if row["gold_rank"] is not None)
    rerank_opportunity = sum(1 for row in rows if row["status"] == "rerank_opportunity")
    nearby_only = sum(1 for row in rows if row["status"] == "nearby_only")
    recall_failure = sum(1 for row in rows if row["status"] == "recall_failure")
    gold_ranks = [row["gold_rank"] for row in rows if row["gold_rank"] is not None]

    return {
        "eval_set": eval_set["name"],
        "case_count": count,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "exact_top_k_rate": round(exact_top_k / count, 4),
        "exact_top_10_rate": round(exact_top_10 / count, 4),
        "exact_top_candidate_rate": round(exact_top_candidate / count, 4),
        "rerank_opportunity_rate": round(rerank_opportunity / count, 4),
        "nearby_only_rate": round(nearby_only / count, 4),
        "recall_failure_rate": round(recall_failure / count, 4),
        "median_gold_rank": median(gold_ranks) if gold_ranks else None,
        "status_counts": {
            "exact_top_k": exact_top_k,
            "rerank_opportunity": rerank_opportunity,
            "nearby_only": nearby_only,
            "recall_failure": recall_failure,
        },
        "samples": rows[:8],
        "failures": [row for row in rows if row["status"] in {"rerank_opportunity", "nearby_only", "recall_failure"}][:10],
    }


def _print_human(summary: dict[str, Any]) -> None:
    print(f"eval set: {summary['eval_set']}")
    print(f"case_count: {summary['case_count']}")
    print(f"top_k: {summary['top_k']}")
    print(f"candidate_k: {summary['candidate_k']}")
    print(f"exact_top_k_rate: {summary['exact_top_k_rate']}")
    print(f"exact_top_10_rate: {summary['exact_top_10_rate']}")
    print(f"exact_top_candidate_rate: {summary['exact_top_candidate_rate']}")
    print(f"rerank_opportunity_rate: {summary['rerank_opportunity_rate']}")
    print(f"nearby_only_rate: {summary['nearby_only_rate']}")
    print(f"recall_failure_rate: {summary['recall_failure_rate']}")
    print(f"median_gold_rank: {summary['median_gold_rank']}")
    print(f"status_counts: {summary['status_counts']}")
    print("\nproblem samples:")
    for row in summary["failures"][:8]:
        top = row.get("top_hit") or {}
        print(f"- {row['status']} | gold_rank={row['gold_rank']} | nearby_rank={row['nearby_rank']}")
        print(f"  q: {row['question']}")
        print(f"  gold: {row['gold_source_title']} p.{row['gold_page_start']}-{row['gold_page_end']} / {row['gold_section_title']}")
        print(f"  top1: {top.get('document_title')} p.{top.get('page_start')}-{top.get('page_end')} / {top.get('section_title')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether eval gold chunks enter the retrieval candidate pool.")
    parser.add_argument("--eval-set", default="energy_starter_v1")
    parser.add_argument("--top-k", type=int, default=6, help="The final answer retrieval size to compare against.")
    parser.add_argument("--candidate-k", type=int, default=50, help="The wide candidate pool used to test recall.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = analyze_recall(eval_set_name=args.eval_set, top_k=args.top_k, candidate_k=args.candidate_k)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_human(summary)


if __name__ == "__main__":
    main()
