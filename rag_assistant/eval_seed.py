from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Json

from rag_assistant.core import Database, RetrievalService

QUESTION_PATTERNS = {
    "financial": [
        "这份资料里关于{subject}的营收/收入怎么说？",
        "{subject} 在资料里提到的财务表现是什么？",
        "{subject} 的收入、利润或经营结果有哪些信息？",
    ],
    "capacity": [
        "{subject} 的产能或装机信息是什么？",
        "这份资料提到的 {subject} 产能扩张有哪些内容？",
        "{subject} 在产能、制造或部署上有哪些进展？",
    ],
    "market": [
        "{subject} 的海外市场或地区布局是什么？",
        "{subject} 在哪些市场有业务或项目？",
        "这份资料里 {subject} 的市场分布怎么写？",
    ],
    "technology": [
        "{subject} 的技术路线或产品方案是什么？",
        "资料中 {subject} 提到哪些技术、产品或解决方案？",
        "{subject} 有哪些技术升级或路线变化？",
    ],
    "project": [
        "{subject} 的项目、订单或合作内容是什么？",
        "这份资料提到 {subject} 哪些项目进展？",
        "{subject} 相关的合作方和项目情况有哪些？",
    ],
    "summary": [
        "这份资料的核心结论是什么？",
        "请总结这份资料最重要的三点。",
        "这份资料主要讲了什么？",
    ],
}

CATEGORY_KEYWORDS = {
    "financial": ["revenue", "income", "profit", "earnings", "margin", "financial"],
    "capacity": ["capacity", "manufacturing", "production", "installed", "deployment", "gw", "mwh"],
    "market": ["market", "region", "overseas", "international", "global", "country"],
    "technology": ["technology", "product", "solution", "platform", "roadmap", "r&d", "innov"],
    "project": ["project", "order", "contract", "agreement", "partner", "cooperation", "delivery"],
}

ALIASES = {
    "CATL": "宁德时代",
    "BYD": "比亚迪",
    "Sungrow": "阳光电源",
    "LONGi": "隆基绿能",
    "Tesla": "特斯拉",
    "First Solar": "First Solar",
    "IEA": "IEA",
    "IRENA": "IRENA",
}


def _subject_from_title(title: str) -> str:
    text = title.strip()
    for key, value in ALIASES.items():
        if key.lower() in text.lower():
            return value
    return text.split()[0] if text else "这家公司"


def _clean_summary(text: str, limit: int = 2) -> str:
    text = " ".join(text.split())
    parts = re.split(r"(?<=[。！？!?\.])\s+", text)
    picked = [part.strip() for part in parts if part.strip()]
    if not picked:
        return text[:280]
    return " ".join(picked[:limit])[:320]


def _pick_category(text: str) -> str:
    lowered = text.lower()
    scores = {
        category: sum(1 for keyword in keywords if keyword.lower() in lowered)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    return max(scores, key=scores.get) if scores else "summary"


def _build_question(title: str, text: str) -> str:
    subject = _subject_from_title(title)
    category = _pick_category(text)
    pattern = random.choice(QUESTION_PATTERNS.get(category, QUESTION_PATTERNS["summary"]))
    return pattern.format(subject=subject)


def _citation(title: str, page_start: int | None, page_end: int | None) -> str:
    if page_start and page_end and page_start == page_end:
        return f"{title} p.{page_start}"
    if page_start and page_end:
        return f"{title} p.{page_start}-{page_end}"
    if page_start:
        return f"{title} p.{page_start}"
    return title


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


def seed_eval_set(name: str = "energy_starter_v1", limit: int = 40) -> dict[str, Any]:
    db = Database()
    service = RetrievalService(db)
    chunks = service.load_chunks()
    paragraphs = [chunk for chunk in chunks if chunk.chunk_level == "paragraph" and len(chunk.text.strip()) > 140]
    random.Random(42).shuffle(paragraphs)

    selected: list[dict[str, Any]] = []
    title_counts: Counter[str] = Counter()
    unique_titles = {chunk.document_title for chunk in paragraphs}
    max_per_title = max(2, (limit + max(1, len(unique_titles)) - 1) // max(1, len(unique_titles)))
    for chunk in paragraphs:
        if len(selected) >= limit:
            break
        if title_counts[chunk.document_title] >= max_per_title:
            continue
        title_counts[chunk.document_title] += 1
        selected.append(
            {
                "id": str(uuid.uuid4()),
                "question": _build_question(chunk.document_title, chunk.text),
                "gold_answer": _clean_summary(chunk.text),
                "gold_citations": [_citation(chunk.document_title, chunk.page_start, chunk.page_end)],
                "gold_chunk_ids": [chunk.id],
                "metadata": {
                    "source_title": chunk.document_title,
                    "chunk_level": chunk.chunk_level,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_title": chunk.section_title,
                    "seed_reason": "auto-generated from paragraph chunk",
                },
            }
        )

    existing = db.fetch_one("SELECT id FROM eval_sets WHERE name = %s", (name,))
    if existing:
        eval_set_id = existing["id"]
    else:
        eval_set_id = str(uuid.uuid4())
        _insert_filtered(
            db,
            "eval_sets",
            {
                "id": eval_set_id,
                "name": name,
                "description": "Auto-seeded renewable-energy eval set for baseline retrieval QA.",
                "version": "v1",
                "created_by": "codex",
                "metadata": {"generated_at": datetime.now(timezone.utc).isoformat(), "limit": limit},
            },
        )

    db.execute("DELETE FROM eval_cases WHERE eval_set_id = %s", (eval_set_id,))
    for item in selected:
        _insert_filtered(
            db,
            "eval_cases",
            {
                "id": item["id"],
                "eval_set_id": eval_set_id,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_citations": item["gold_citations"],
                "gold_chunk_ids": item["gold_chunk_ids"],
                "metadata": item["metadata"],
            },
        )

    return {"eval_set_id": eval_set_id, "name": name, "case_count": len(selected), "samples": selected[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a starter eval set from the renewable energy corpus.")
    parser.add_argument("--name", default="energy_starter_v1")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = seed_eval_set(name=args.name, limit=args.limit)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"seeded eval set {result['name']} with {result['case_count']} cases")
        for sample in result["samples"]:
            print(f"- {sample['question']} :: {sample['gold_citations'][0]}")


if __name__ == "__main__":
    main()
