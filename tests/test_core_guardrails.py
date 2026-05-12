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


def test_direct_extracts_total_automotive_revenues() -> None:
    service = RetrievalService.__new__(RetrievalService)
    text = (
        "Tesla, Inc. Consolidated Statements of Operations (in millions) "
        "Year Ended December 31, 2023 2022 2021 Revenues "
        "Automotive sales $ 78,509 $ 67,210 $ 44,125 "
        "Automotive regulatory credits 1,790 1,776 1,465 "
        "Automotive leasing 2,120 2,476 1,642 "
        "Total automotive revenues 82,419 71,462 47,232 "
        "Total revenues 96,773 81,462 53,823"
    )

    answer = service._direct_fact_answer(
        "What was Tesla's automotive revenue in 2023?",
        [_hit(text)],
    )

    assert answer is not None
    assert "82,419" in answer
    assert "Total automotive" in answer or "total automotive" in answer


def test_direct_extracts_delivered_consumer_vehicles() -> None:
    service = RetrievalService.__new__(RetrievalService)
    text = (
        "Overview and 2023 Highlights. In 2023, we produced 1,845,985 "
        "consumer vehicles and delivered 1,808,581 consumer vehicles."
    )

    answer = service._direct_fact_answer(
        "How many vehicles did Tesla deliver in 2023?",
        [_hit(text, page=34)],
    )

    assert answer is not None
    assert "1,808,581" in answer
    assert "473,382" not in answer


def test_sanitize_citations_removes_empty_and_out_of_range_refs() -> None:
    answer = "Evidence is in [1]至[] and invalid [9]. Another empty [][] remains."

    cleaned = RetrievalService._sanitize_citations(answer, source_count=2)

    assert "[]" not in cleaned
    assert "[9]" not in cleaned
    assert "[1]" in cleaned


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
