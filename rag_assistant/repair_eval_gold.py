from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import re
from typing import Any

from psycopg.types.json import Json

from rag_assistant.core import (
    Database,
    PREFERRED_SECTION_TERMS,
    RERANK_CATEGORY_TERMS,
    RetrievalService,
)

SENTENCE_RE = re.compile(r"(?<=[。！？!?\.])\s+")


def _citation(title: str, page_start: int | None, page_end: int | None) -> str:
    if page_start and page_end and page_start == page_end:
        return f"{title} p.{page_start}"
    if page_start and page_end:
        return f"{title} p.{page_start}-{page_end}"
    if page_start:
        return f"{title} p.{page_start}"
    return title


def _clean_answer(text: str, limit: int = 2) -> str:
    text = " ".join(text.split())
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return text[:360]
    return " ".join(sentences[:limit])[:420]


def _load_cases(db: Database, eval_set_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_set = db.fetch_one("SELECT id, name FROM eval_sets WHERE name = %s", (eval_set_name,))
    if not eval_set:
        raise RuntimeError(f"Eval set {eval_set_name!r} not found.")
    cases = db.fetch_all(
        """
        SELECT id, question, gold_answer, gold_citations, gold_chunk_ids, metadata
        FROM eval_cases
        WHERE eval_set_id = %s
        ORDER BY created_at NULLS LAST, question
        """,
        (eval_set["id"],),
    )
    return eval_set, cases


def _score_gold_candidate(service: RetrievalService, question: str, chunk: Any, category: str) -> tuple[float, list[str]]:
    query_tokens = service._build_query_tokens(question)
    chunk_tokens = service._chunk_tokens(chunk)
    idf = service._term_idf()
    matched_terms = sorted(set(query_tokens) & set(chunk_tokens), key=lambda token: idf.get(token, 0.0), reverse=True)
    weighted_overlap = sum(idf.get(token, 0.0) * min(query_tokens[token], chunk_tokens[token]) for token in matched_terms)
    weighted_query = sum(idf.get(token, 0.0) * count for token, count in query_tokens.items())
    overlap_score = weighted_overlap / max(0.01, weighted_query)

    searchable = " ".join([chunk.document_title, chunk.section_title or "", chunk.text]).lower()
    category_terms = RERANK_CATEGORY_TERMS.get(category, [])
    category_hits = [term for term in category_terms if term in searchable]
    preferred_hits = [term for term in PREFERRED_SECTION_TERMS.get(category, []) if term in (chunk.section_title or "").lower()]

    length = len(" ".join(chunk.text.split()))
    score = overlap_score * 2.0 + len(category_hits) * 0.45 + len(preferred_hits) * 0.25
    if 140 <= length <= 900:
        score += 0.35
    elif length < 80:
        score -= 0.5
    elif length > 1600:
        score -= 0.25
    if re.search(r"\b(20\d{2}|\d+(?:\.\d+)?%|\$?\d+(?:,\d{3})*(?:\.\d+)?)\b", chunk.text):
        score += 0.15
    if any(noisy in searchable for noisy in ["table of contents", "safe harbor", "forward-looking"]):
        score -= 0.35
    if category != "summary" and not category_hits:
        score -= 1.0
    return score, matched_terms[:8]


def repair_eval_gold(eval_set_name: str = "energy_starter_v1", apply: bool = False, min_margin: float = 0.05) -> dict[str, Any]:
    db = Database()
    service = RetrievalService(db)
    eval_set, cases = _load_cases(db, eval_set_name)
    chunks = service.load_chunks()
    chunks_by_doc: dict[str, list[Any]] = {}
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    for chunk in chunks:
        if chunk.chunk_level != "paragraph" or not chunk.text.strip():
            continue
        chunks_by_doc.setdefault(chunk.document_title, []).append(chunk)

    changes: list[dict[str, Any]] = []
    for case in cases:
        metadata = dict(case.get("metadata") or {})
        source_title = str(metadata.get("source_title") or "")
        candidates = chunks_by_doc.get(source_title, [])
        if not candidates:
            continue
        category = service._question_category(case["question"])
        old_ids = [str(chunk_id) for chunk_id in (case.get("gold_chunk_ids") or [])]
        old_chunk = chunks_by_id.get(old_ids[0]) if old_ids else None
        old_score = _score_gold_candidate(service, case["question"], old_chunk, category)[0] if old_chunk else -math.inf

        scored = [(_score_gold_candidate(service, case["question"], chunk, category), chunk) for chunk in candidates]
        scored.sort(key=lambda item: item[0][0], reverse=True)
        (new_score, matched_terms), new_chunk = scored[0]
        if new_chunk.id in old_ids and old_score >= new_score - min_margin:
            continue
        if new_score < old_score + min_margin and old_score > 0:
            continue

        old_metadata = dict(metadata)
        metadata.update(
            {
                "source_title": new_chunk.document_title,
                "chunk_level": new_chunk.chunk_level,
                "page_start": new_chunk.page_start,
                "page_end": new_chunk.page_end,
                "section_title": new_chunk.section_title,
                "gold_repair": {
                    "method": "doc_category_bm25_repair",
                    "question_category": category,
                    "old_gold_chunk_ids": old_ids,
                    "old_metadata": old_metadata,
                    "old_score": round(old_score, 4) if old_score != -math.inf else None,
                    "new_score": round(new_score, 4),
                    "matched_terms": matched_terms,
                },
            }
        )
        change = {
            "case_id": str(case["id"]),
            "question": case["question"],
            "category": category,
            "old_gold_chunk_ids": old_ids,
            "new_gold_chunk_id": new_chunk.id,
            "old_score": round(old_score, 4) if old_score != -math.inf else None,
            "new_score": round(new_score, 4),
            "old_citation": (case.get("gold_citations") or [None])[0],
            "new_citation": _citation(new_chunk.document_title, new_chunk.page_start, new_chunk.page_end),
            "new_section_title": new_chunk.section_title,
            "new_snippet": " ".join(new_chunk.text.split())[:220],
        }
        changes.append(change)

        if apply:
            db.execute(
                """
                UPDATE eval_cases
                SET gold_answer = %s,
                    gold_citations = %s,
                    gold_chunk_ids = %s::uuid[],
                    metadata = %s
                WHERE id = %s
                """,
                (
                    _clean_answer(new_chunk.text),
                    [_citation(new_chunk.document_title, new_chunk.page_start, new_chunk.page_end)],
                    [new_chunk.id],
                    Json(metadata),
                    case["id"],
                ),
            )

    return {
        "eval_set": eval_set["name"],
        "apply": apply,
        "case_count": len(cases),
        "changed_count": len(changes),
        "samples": changes[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair noisy auto-generated eval gold chunks.")
    parser.add_argument("--eval-set", default="energy_starter_v1")
    parser.add_argument("--apply", action="store_true", help="Write repaired gold chunks back to eval_cases.")
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = repair_eval_gold(eval_set_name=args.eval_set, apply=args.apply, min_margin=args.min_margin)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(f"eval set: {summary['eval_set']}")
    print(f"apply: {summary['apply']}")
    print(f"case_count: {summary['case_count']}")
    print(f"changed_count: {summary['changed_count']}")
    for sample in summary["samples"]:
        print(f"- {sample['category']} | {sample['question']}")
        print(f"  old: {sample['old_citation']} score={sample['old_score']}")
        print(f"  new: {sample['new_citation']} score={sample['new_score']} / {sample['new_section_title']}")
        print(f"  snippet: {sample['new_snippet']}")


if __name__ == "__main__":
    main()
