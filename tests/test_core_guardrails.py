from __future__ import annotations

import json
from pathlib import Path

from rag_assistant.core import ChunkRecord, RetrievalHit, RetrievalService
from scripts.ingest_energy_data import PdfBlock
from scripts import ingest_single_pdf
from scripts.ingest_single_pdf import parser_fingerprint
from rag_assistant.facts import extract_fact_candidates
from rag_assistant.jobs import JobWorker


def _hit(text: str, title: str = "Tesla 2023 Annual Report 10-K", page: int = 51) -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=1.0,
        chunk=ChunkRecord(
            id="chunk-1",
            document_title=title,
            chunk_level="paragraph",
            section_title="Overview",
            text=text,
            page_start=page,
            page_end=page,
            metadata={},
        ),
        matched_terms=[],
        reason="test",
    )


def test_direct_extracts_total_revenue_generic() -> None:
    # Generic, company-agnostic table extraction (no Tesla-specific branch).
    service = RetrievalService.__new__(RetrievalService)
    text = (
        "Consolidated Statements of Operations (in millions) "
        "Year Ended December 31, 2023 2022 2021 "
        "Total revenues 96,773 81,462 53,823 "
        "Net income attributable to common stockholders 14,997 12,556 5,519"
    )

    answer = service._direct_fact_answer(
        "What was the total revenue in 2023?",
        [_hit(text)],
    )

    assert answer is not None
    assert "96,773" in answer
    assert "total revenues" in answer.lower()
    # No fabricated currency/unit any more — only the extracted value.
    assert "$" not in answer
    assert "million" not in answer


def test_direct_fact_answers_in_question_language() -> None:
    service = RetrievalService.__new__(RetrievalService)
    text = "主要财务指标 2023 2022 营业收入 4,009 3,285 净利润 441 307"

    answer = service._direct_fact_answer("2023年营业收入是多少？", [_hit(text)])

    assert answer is not None
    assert "4,009" in answer
    assert "营业收入" in answer
    # Chinese question → Chinese answer (prompt rule 6).
    assert "The" not in answer


def test_number_grounding_allows_rounding_and_unit_scaling() -> None:
    service = RetrievalService.__new__(RetrievalService)
    hits = [_hit("Revenue was 50,000 million, with a margin of 48.2%.")]

    # 500 (亿) scales to 50,000 (百万); 48 rounds from 48.2 — both grounded.
    grounded, ungrounded = service._check_number_grounding(
        "营收约 500 亿，毛利率约 48%。", hits
    )

    assert grounded
    assert ungrounded == []


def test_number_grounding_flags_fabricated_number() -> None:
    service = RetrievalService.__new__(RetrievalService)
    hits = [_hit("Revenue was 50,000 million.")]

    grounded, ungrounded = service._check_number_grounding(
        "Revenue was 999 million.", hits
    )

    assert not grounded
    assert "999" in ungrounded


def test_sanitize_citations_removes_empty_and_out_of_range_refs() -> None:
    answer = "Evidence is in [1]至[] and invalid [9]. Another empty [][] remains."

    cleaned = RetrievalService._sanitize_citations(answer, source_count=2)

    assert "[]" not in cleaned
    assert "[9]" not in cleaned
    assert "[1]" in cleaned


def test_sanitize_preserves_plain_text_ranges() -> None:
    # Ordinary "至" between years must NOT be stripped.
    cleaned = RetrievalService._sanitize_citations("2020至2023年累计装机 [1]", source_count=3)

    assert "2020至2023年累计装机" in cleaned
    assert "[1]" in cleaned


def test_sanitize_trims_out_of_range_citation_range() -> None:
    cleaned = RetrievalService._sanitize_citations("见 [2]至[9] 资料", source_count=3)

    assert "[2]" in cleaned
    assert "[9]" not in cleaned
    assert "至" not in cleaned


def test_supercharger_case_does_not_expect_missing_station_count() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"
    data = json.loads(dataset_path.read_text())
    cases = [case for case in data["cases"] if case["id"] == "tesla_2023_008"]

    assert len(cases) == 1
    case = cases[0]
    assert "How many" not in case["question"]
    assert "5,265" not in case["expected_keywords"]
    assert "Supercharger" in case["expected_keywords"]


def test_cache_key_changes_when_strategy_changes(monkeypatch) -> None:
    service = RetrievalService.__new__(RetrievalService)
    service._corpus_fingerprint = "corpus-v1"

    monkeypatch.setenv("RAG_ENABLE_VECTOR", "0")
    first = service._retrieval_strategy_fingerprint(top_k=5)

    monkeypatch.setenv("RAG_ENABLE_VECTOR", "1")
    second = service._retrieval_strategy_fingerprint(top_k=5)

    assert first != second


def test_retrieval_result_cache_rehydrates_hits(monkeypatch) -> None:
    class FakeDb:
        def __init__(self) -> None:
            self.updated = False

        def fetch_one(self, sql, params=()):
            return {"chunk_ids": ["chunk-1"]}

        def execute(self, sql, params=()):
            self.updated = True

    service = RetrievalService.__new__(RetrievalService)
    service.db = FakeDb()
    service._retrieval_cache_key = lambda question, top_k: ("cache-key", "query-hash", "corpus", "strategy")
    monkeypatch.setenv("RAG_RETRIEVAL_RESULT_CACHE", "1")

    hits = service._fetch_retrieval_result_cache(
        "What was Tesla revenue?",
        1,
        [
            ChunkRecord(
                id="chunk-1",
                document_title="Tesla 2023 Annual Report 10-K",
                chunk_level="paragraph",
                section_title="Revenue",
                text="Total revenues were 96,773.",
                page_start=51,
                page_end=51,
                metadata={},
            )
        ],
    )

    assert hits is not None
    assert hits[0].rank == 1
    assert hits[0].chunk.id == "chunk-1"
    assert hits[0].reason == "retrieval_result_cache"
    assert service._last_retrieval_cache_stats["hit"] is True
    assert service.db.updated is True


def test_parser_fingerprint_is_stable() -> None:
    assert parser_fingerprint() == parser_fingerprint()
    assert len(parser_fingerprint()) == 64


def test_document_parse_cache_round_trip(monkeypatch) -> None:
    storage = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=()):
            self.sql = sql
            self.params = params
            if "INSERT INTO document_parse_cache" in sql:
                storage["row"] = {
                    "parse_method": params[3],
                    "ocr_used": params[4],
                    "ocr_reason": params[5],
                    "extracted_char_count": params[6],
                    "ocr_char_count": params[7],
                    "quality": {"quality_score": 0.9},
                    "blocks": [{"page_number": 1, "text": "hello"}],
                }

        def fetchone(self):
            if "row" not in storage:
                return None
            row = storage["row"]
            return (
                "cache-id",
                row["parse_method"],
                row["ocr_used"],
                row["ocr_reason"],
                row["extracted_char_count"],
                row["ocr_char_count"],
                row["quality"],
                row["blocks"],
            )

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

    monkeypatch.setattr(ingest_single_pdf.psycopg, "connect", lambda dsn: FakeConnection())

    ingest_single_pdf._write_parse_cache(
        "postgresql://example",
        "file-hash",
        "doc.pdf",
        [PdfBlock(page_number=1, text="hello")],
        "pdftotext",
        False,
        "text_sufficient",
        5,
        0,
        {"quality_score": 0.9},
    )
    loaded = ingest_single_pdf._load_parse_cache("postgresql://example", "file-hash")

    assert loaded is not None
    blocks, parse_method, ocr_used, ocr_reason, extracted_chars, ocr_chars, quality = loaded
    assert blocks == [PdfBlock(page_number=1, text="hello")]
    assert parse_method == "pdftotext"
    assert ocr_used is False
    assert ocr_reason == "text_sufficient"
    assert extracted_chars == 5
    assert ocr_chars == 0
    assert quality["quality_score"] == 0.9


def test_extract_fact_candidate_prefers_money_value_over_year() -> None:
    facts = extract_fact_candidates(
        "What was Tesla's automotive revenue in 2023?",
        "The total automotive revenues figure for 2023 was $82,419 million [1].",
        ["[1] Tesla 2023 Annual Report 10-K p.51"],
        [
            {
                "chunk_id": "17ceea57-40a1-41cd-ad29-a33ae1bae9c1",
                "document_title": "Tesla 2023 Annual Report 10-K",
            }
        ],
    )

    assert len(facts) == 1
    assert facts[0].company == "Tesla"
    assert facts[0].metric == "total automotive revenues"
    assert facts[0].period == "2023"
    assert facts[0].value == "82,419"
    assert facts[0].unit == "USD million"


def test_job_worker_runs_smoke_job() -> None:
    worker = JobWorker.__new__(JobWorker)
    result = worker._run_job(
        {
            "id": "job-1",
            "job_type": "smoke_test",
            "metadata": {"payload": {"hello": "world"}},
            "retries": 0,
            "max_retries": 1,
        }
    )

    assert result.ok is True
    assert result.payload["message"] == "worker ok"
    assert result.payload["payload"] == {"hello": "world"}
