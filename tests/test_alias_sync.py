from __future__ import annotations

from rag_assistant.alias_sync import (
    apply_missing,
    build_expected,
    coverage_report,
    diff_missing,
)

SOURCES = [
    {"company": "CATL", "aliases": ["宁德时代"], "tags": ["annual_report", "battery", "storage"]},
    {"company": "Enphase", "tags": ["solar", "storage"]},  # no Chinese alias recorded
    {"company": "None", "tags": ["industry_report"]},  # manifest noise — ignored
]


def test_build_expected_derives_english_token_and_domain_terms() -> None:
    expected = build_expected(SOURCES)

    # English token always derivable.
    assert expected["document_hints"]["catl"] == ["catl"]
    assert expected["document_hints"]["enphase"] == ["enphase"]
    # Chinese alias maps to the token too.
    assert expected["document_hints"]["宁德时代"] == ["catl"]
    # Domain terms come from tags; document-type tags are dropped.
    assert expected["cn_alias_map"]["宁德时代"] == ["CATL", "battery", "storage"]


def test_build_expected_never_guesses_missing_chinese_alias() -> None:
    expected = build_expected(SOURCES)

    # Enphase has no recorded alias -> no cn_alias_map entry, flagged as TODO.
    assert "Enphase" in expected["todos"]
    assert not any(v == ["Enphase", "solar", "storage"] for v in expected["cn_alias_map"].values())
    # The "None" company is ignored entirely.
    assert "none" not in expected["document_hints"]


def test_diff_only_reports_missing_entries() -> None:
    expected = build_expected(SOURCES)
    current = {"document_hints": {"catl": ["catl"], "宁德时代": ["catl"]}, "cn_alias_map": {"宁德时代": ["CATL"]}}

    missing = diff_missing(expected, current)

    assert "enphase" in missing["document_hints"]
    assert "catl" not in missing["document_hints"]  # already present, not re-added
    assert "宁德时代" not in missing["cn_alias_map"]  # existing richer entry untouched


def test_coverage_report_splits_hard_and_soft_misses() -> None:
    current = {"document_hints": {"catl": ["catl"], "宁德时代": ["catl"]}, "cn_alias_map": {"宁德时代": ["CATL"]}}

    report = coverage_report(SOURCES, current)

    # Enphase token missing entirely -> hard miss.
    assert report["unregistered"] == ["Enphase"]
    # CATL registered with a Chinese alias -> no soft miss for it.
    assert "CATL" not in report["missing_aliases"]


def test_apply_missing_preserves_other_keys() -> None:
    current = {"_comment": "keep me", "document_hints": {}, "cn_alias_map": {}}
    missing = {"document_hints": {"enphase": ["enphase"]}, "cn_alias_map": {}}

    updated = apply_missing(current, missing)

    assert updated["_comment"] == "keep me"
    assert updated["document_hints"]["enphase"] == ["enphase"]
    # original dict not mutated in place
    assert current["document_hints"] == {}
