from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Json

from rag_assistant.core import Database, RetrievalService

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _answer_overlap(prediction: str, gold: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    return overlap / math.sqrt(len(pred_tokens) * len(gold_tokens))


def _page_overlap(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> bool:
    if a_start is None or b_start is None:
        return False
    a_end = a_end or a_start
    b_end = b_end or b_start
    return max(a_start, b_start) <= min(a_end, b_end)


def _citation_score(sources: list[str], gold_citations: list[str], metadata: dict[str, Any]) -> float:
    if not sources:
        return 0.0
    source_text = " ".join(sources).lower()
    exact = 0.0
    if gold_citations:
        exact = sum(1 for citation in gold_citations if citation.lower() in source_text) / len(gold_citations)
    source_title = str(metadata.get("source_title") or "").lower()
    if source_title and source_title in source_text:
        return max(exact, 0.7)
    return exact


def _retrieval_score(retrieved_chunks: list[dict[str, Any]], gold_chunk_ids: list[str], metadata: dict[str, Any]) -> float:
    if not retrieved_chunks:
        return 0.0
    gold = {str(chunk_id) for chunk_id in gold_chunk_ids}
    gold_title = str(metadata.get("source_title") or "").lower()
    gold_section = str(metadata.get("section_title") or "").lower()
    gold_page_start = metadata.get("page_start")
    gold_page_end = metadata.get("page_end")
    best = 0.0
    for index, chunk in enumerate(retrieved_chunks, start=1):
        rank_discount = 1.0 / index
        if str(chunk.get("chunk_id")) in gold:
            best = max(best, 1.0 * rank_discount)
            continue
        chunk_title = str(chunk.get("document_title") or "").lower()
        chunk_section = str(chunk.get("section_title") or "").lower()
        same_doc = bool(gold_title and chunk_title == gold_title)
        same_page = _page_overlap(chunk.get("page_start"), chunk.get("page_end"), gold_page_start, gold_page_end)
        same_section = bool(gold_section and chunk_section == gold_section)
        if same_doc and same_page:
            best = max(best, 0.7 * rank_discount)
        elif same_doc and same_section:
            best = max(best, 0.5 * rank_discount)
        elif same_doc:
            best = max(best, 0.2 * rank_discount)
    return best


def _table_columns(db: Database, table_name: str) -> set[str]:
    rows = db.fetch_all(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """,
        (table_name,),
    )
    return {row["column_name"] for row in rows}


def _insert_filtered(db: Database, table_name: str, payload: dict[str, Any]) -> None:
    columns = _table_columns(db, table_name)
    insert_cols = [key for key in payload if key in columns]
    if not insert_cols:
        return
    sql = f"INSERT INTO {table_name} ({', '.join(insert_cols)}) VALUES ({', '.join(f'%({key})s' for key in insert_cols)})"
    db.execute(sql, {key: Json(payload[key]) if isinstance(payload[key], dict) else payload[key] for key in insert_cols})


def run_eval(eval_set_name: str = "energy_starter_v1", top_k: int = 6) -> dict[str, Any]:
    db = Database()
    service = RetrievalService(db)

    eval_set = db.fetch_one("SELECT id, name FROM eval_sets WHERE name = %s", (eval_set_name,))
    if not eval_set:
        raise RuntimeError(f"Eval set {eval_set_name!r} not found. Run eval_seed first.")

    cases = db.fetch_all(
        """
        SELECT id, question, gold_answer, gold_citations, gold_chunk_ids, metadata
        FROM eval_cases
        WHERE eval_set_id = %s
        ORDER BY created_at NULLS LAST, question
        """,
        (eval_set["id"],),
    )
    if not cases:
        raise RuntimeError(f"Eval set {eval_set_name!r} has no cases.")

    run_id = str(uuid.uuid4())
    model_name = os.getenv("OLLAMA_MODEL") or "extractive-baseline"
    _insert_filtered(
        db,
        "eval_runs",
        {
            "id": run_id,
            "eval_set_id": eval_set["id"],
            "run_name": f"{eval_set_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            "model_name": model_name,
            "started_at": datetime.now(timezone.utc),
            "metadata": {"eval_set_name": eval_set_name, "top_k": top_k},
        },
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        result = service.answer(case["question"], top_k=top_k)
        gold_citations = list(case.get("gold_citations") or [])
        gold_chunk_ids = [str(chunk_id) for chunk_id in (case.get("gold_chunk_ids") or [])]
        metadata = dict(case.get("metadata") or {})
        retrieval_score = _retrieval_score(result.retrieved_chunks, gold_chunk_ids, metadata)
        answer_score = _answer_overlap(result.answer, case.get("gold_answer") or "")
        citation_score = _citation_score(result.sources, gold_citations, metadata)
        pass_fail = retrieval_score > 0 and citation_score >= 0.5 and answer_score >= 0.15

        _insert_filtered(
            db,
            "eval_results",
            {
                "id": str(uuid.uuid4()),
                "eval_run_id": run_id,
                "eval_case_id": case["id"],
                "retrieval_score": round(retrieval_score, 4),
                "answer_score": round(answer_score, 4),
                "citation_score": round(citation_score, 4),
                "pass_fail": pass_fail,
                "raw_output": {
                    "question": case["question"],
                    "answer": result.answer,
                    "sources": result.sources,
                    "retrieved_chunks": result.retrieved_chunks,
                    "latency_ms": round(result.latency_ms, 2),
                    "used_llm": result.used_llm,
                    "debug": result.debug,
                },
            },
        )

        rows.append(
            {
                "question": case["question"],
                "retrieval_score": round(retrieval_score, 4),
                "answer_score": round(answer_score, 4),
                "citation_score": round(citation_score, 4),
                "pass_fail": pass_fail,
            }
        )

    if "finished_at" in _table_columns(db, "eval_runs"):
        db.execute("UPDATE eval_runs SET finished_at = %s WHERE id = %s", (datetime.now(timezone.utc), run_id))

    return {
        "eval_set": eval_set_name,
        "run_id": run_id,
        "model_name": model_name,
        "count": len(rows),
        "retrieval_mean": round(sum(r["retrieval_score"] for r in rows) / len(rows), 4),
        "answer_mean": round(sum(r["answer_score"] for r in rows) / len(rows), 4),
        "citation_mean": round(sum(r["citation_score"] for r in rows) / len(rows), 4),
        "pass_rate": round(sum(1 for r in rows if r["pass_fail"]) / len(rows), 4),
        "samples": rows[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the starter retrieval eval set.")
    parser.add_argument("--eval-set", default="energy_starter_v1")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_eval(eval_set_name=args.eval_set, top_k=args.top_k)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"eval set: {summary['eval_set']}")
        print(f"model: {summary['model_name']}")
        print(f"count: {summary['count']}")
        print(f"retrieval_mean: {summary['retrieval_mean']}")
        print(f"answer_mean: {summary['answer_mean']}")
        print(f"citation_mean: {summary['citation_mean']}")
        print(f"pass_rate: {summary['pass_rate']}")
        for sample in summary["samples"]:
            print(
                f"- {sample['question']} :: retrieval={sample['retrieval_score']:.3f} "
                f"answer={sample['answer_score']:.3f} citation={sample['citation_score']:.3f}"
            )


if __name__ == "__main__":
    main()
