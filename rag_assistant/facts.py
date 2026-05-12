from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from psycopg.types.json import Json


COMPANY_ALIASES: dict[str, str] = {
    "tesla": "Tesla",
    "特斯拉": "Tesla",
    "catl": "CATL",
    "宁德时代": "CATL",
    "vestas": "Vestas",
    "longi": "LONGi",
    "隆基": "LONGi",
    "隆基绿能": "LONGi",
    "first solar": "First Solar",
    "sungrow": "Sungrow",
    "阳光电源": "Sungrow",
    "byd": "BYD",
    "比亚迪": "BYD",
}


METRIC_HINTS: list[tuple[str, str]] = [
    ("automotive revenue", "total automotive revenues"),
    ("total revenue", "total revenues"),
    ("revenue", "revenue"),
    ("营收", "revenue"),
    ("营业收入", "revenue"),
    ("net income", "net income"),
    ("净利润", "net income"),
    ("research and development", "research and development"),
    ("r&d", "research and development"),
    ("deliver", "vehicle deliveries"),
    ("交付", "vehicle deliveries"),
    ("shipment", "shipments"),
    ("出货", "shipments"),
]


@dataclass(slots=True)
class FactCandidate:
    company: str
    metric: str
    value: str
    unit: str | None
    period: str | None
    source_citation: str
    source_chunk_id: str | None
    confidence: float
    metadata: dict[str, Any]


def _infer_company(question: str, chunks: Sequence[dict[str, Any]]) -> str | None:
    lowered = question.lower()
    for alias, canonical in COMPANY_ALIASES.items():
        if alias in lowered or alias in question:
            return canonical
    for chunk in chunks:
        title = str(chunk.get("document_title") or "").lower()
        for alias, canonical in COMPANY_ALIASES.items():
            if alias in title:
                return canonical
    return None


def _infer_metric(question: str, answer: str) -> str | None:
    text = f"{question} {answer}".lower()
    for hint, metric in METRIC_HINTS:
        if hint in text:
            return metric
    return None


def _infer_period(question: str, answer: str) -> str | None:
    match = re.search(r"\b(20\d{2})\b", f"{question} {answer}")
    return match.group(1) if match else None


def _extract_value_and_unit(answer: str) -> tuple[str | None, str | None]:
    money = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)?", answer, re.IGNORECASE)
    if money:
        value = money.group(1)
        unit = money.group(2)
        if unit:
            unit = f"USD {unit.lower()}"
        else:
            unit = "USD"
        return value, unit
    number_with_unit = re.search(r"\b(?!20\d{2}\b)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|vehicles|mw|gw|mwh|gwh)\b", answer, re.IGNORECASE)
    if number_with_unit:
        value = number_with_unit.group(1)
        unit = number_with_unit.group(2).lower()
        return value, unit
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", answer)
    if percent:
        return percent.group(1), "%"
    return None, None


def extract_fact_candidates(
    question: str,
    answer: str,
    sources: Sequence[str],
    chunks: Sequence[dict[str, Any]],
) -> list[FactCandidate]:
    company = _infer_company(question, chunks)
    metric = _infer_metric(question, answer)
    period = _infer_period(question, answer)
    value, unit = _extract_value_and_unit(answer)
    if not company or not metric or not value or not sources or not chunks:
        return []

    source_index = 0
    citation_match = re.search(r"\[(\d+)\]", answer)
    if citation_match:
        source_index = max(0, int(citation_match.group(1)) - 1)
    source_index = min(source_index, len(sources) - 1, len(chunks) - 1)
    source = sources[source_index]
    chunk = chunks[source_index]

    return [
        FactCandidate(
            company=company,
            metric=metric,
            value=value,
            unit=unit,
            period=period,
            source_citation=source,
            source_chunk_id=chunk.get("chunk_id"),
            confidence=0.92 if "direct-extractive" in str(answer).lower() else 0.85,
            metadata={
                "question": question,
                "answer": answer,
                "source_rank": source_index + 1,
            },
        )
    ]


def persist_fact_candidates(db, candidates: Sequence[FactCandidate]) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    for candidate in candidates:
        row = db.fetch_one(
            """
            INSERT INTO research_facts (
                company, metric, value, unit, period, source_citation,
                source_chunk_id, confidence, metadata, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (company, metric, COALESCE(period, ''), value, source_citation)
            DO UPDATE SET
                unit = EXCLUDED.unit,
                source_chunk_id = COALESCE(research_facts.source_chunk_id, EXCLUDED.source_chunk_id),
                confidence = GREATEST(research_facts.confidence, EXCLUDED.confidence),
                metadata = research_facts.metadata || EXCLUDED.metadata,
                updated_at = now()
            RETURNING id::text, company, metric, value, unit, period, source_citation, review_status, confidence::float
            """,
            (
                candidate.company,
                candidate.metric,
                candidate.value,
                candidate.unit,
                candidate.period,
                candidate.source_citation,
                candidate.source_chunk_id,
                candidate.confidence,
                Json(candidate.metadata),
            ),
        )
        if row:
            saved.append(dict(row))
            _ensure_answer_review(db, row, candidate)
    return saved


def _ensure_answer_review(db, fact_row: dict[str, Any], candidate: FactCandidate) -> None:
    existing = db.fetch_one(
        """
        SELECT id
        FROM answer_reviews
        WHERE metadata->>'research_fact_id' = %s
          AND status IN ('pending', 'approved', 'needs_revision')
        LIMIT 1
        """,
        (fact_row["id"],),
    )
    if existing:
        return
    reviewed_fact = " ".join(
        part
        for part in [
            candidate.company,
            candidate.metric,
            candidate.period or "",
            candidate.value,
            candidate.unit or "",
        ]
        if part
    )
    db.execute(
        """
        INSERT INTO answer_reviews (
            status, reviewed_fact, source_citation, metadata
        )
        VALUES ('pending', %s, %s, %s)
        """,
        (
            reviewed_fact,
            candidate.source_citation,
            Json({"research_fact_id": fact_row["id"], **candidate.metadata}),
        ),
    )
