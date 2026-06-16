"""Derive query-alias entries from the corpus source manifest.

The corpus manifest (data/sources/energy_sources.json) is the source of truth
for which companies exist and what their human-recorded Chinese names are
(optional ``aliases`` field). This module turns that into the retrieval alias
maps stored in data/config/query_aliases.json.

Principle: automate the *copying*, never the *guessing*.
  - document_hints (English token) and domain terms (from tags) are derived
    deterministically.
  - Chinese aliases come only from the manifest's ``aliases`` field. A company
    without recorded aliases is reported as a TODO — never invented.

Used by scripts/sync_query_aliases.py (CLI) and the API startup self-check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_PATH = REPO_ROOT / "data" / "sources" / "energy_sources.json"
DEFAULT_ALIASES_PATH = REPO_ROOT / "data" / "config" / "query_aliases.json"

# Tags worth turning into retrieval expansion terms. Document-type tags such as
# "annual_report" / "investor_relations" are noise for search and are ignored.
TAG_TERMS: dict[str, list[str]] = {
    "battery": ["battery"],
    "storage": ["storage"],
    "solar": ["solar", "photovoltaic"],
    "inverter": ["inverter"],
    "ev": ["ev"],
    "wind": ["wind"],
    "renewables": ["renewables"],
}


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _token(name: str) -> str:
    return name.strip().lower()


def _domain_terms(tags: list[str] | None) -> list[str]:
    terms: list[str] = []
    for tag in tags or []:
        for term in TAG_TERMS.get(str(tag).lower(), []):
            if term not in terms:
                terms.append(term)
    return terms


def corpus_companies(sources: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Merge manifest records into one entry per company: aliases + tags."""
    companies: dict[str, dict[str, list[str]]] = {}
    for record in sources:
        name = record.get("company")
        if not name or str(name).lower() == "none":
            continue
        entry = companies.setdefault(name, {"aliases": [], "tags": []})
        for alias in record.get("aliases") or []:
            if alias not in entry["aliases"]:
                entry["aliases"].append(alias)
        for tag in record.get("tags") or []:
            if tag not in entry["tags"]:
                entry["tags"].append(tag)
    return companies


def build_expected(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """The alias entries the corpus *should* have, derived from the manifest."""
    companies = corpus_companies(sources)
    document_hints: dict[str, list[str]] = {}
    cn_alias_map: dict[str, list[str]] = {}
    todos: list[str] = []
    for name, info in companies.items():
        token = _token(name)
        terms = _domain_terms(info["tags"])
        document_hints[token] = [token]
        for alias in info["aliases"]:
            document_hints[_token(alias)] = [token]
        if info["aliases"]:
            for alias in info["aliases"]:
                cn_alias_map[alias] = [name, *terms]
        else:
            # No human-recorded Chinese name — leave it; do not guess.
            todos.append(name)
    return {
        "document_hints": document_hints,
        "cn_alias_map": cn_alias_map,
        "todos": todos,
    }


def diff_missing(expected: dict[str, Any], current: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Expected entries that are absent from the current aliases file."""
    missing: dict[str, dict[str, list[str]]] = {"document_hints": {}, "cn_alias_map": {}}
    for section in ("document_hints", "cn_alias_map"):
        current_section = current.get(section, {})
        for key, value in expected[section].items():
            if key not in current_section:
                missing[section][key] = value
    return missing


def coverage_report(sources: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, list[str]]:
    """For the startup self-check.

    unregistered    — English token missing from document_hints. Target-document
                      isolation cannot work for this company (hard problem).
    missing_aliases — registered, but no Chinese alias recorded. Chinese queries
                      may not expand correctly (soft problem).
    """
    companies = corpus_companies(sources)
    hints = current.get("document_hints", {})
    cn_keys = set(current.get("cn_alias_map", {}))
    unregistered: list[str] = []
    missing_aliases: list[str] = []
    for name, info in companies.items():
        if _token(name) not in hints:
            unregistered.append(name)
            continue
        if not any(alias in cn_keys for alias in info["aliases"]):
            missing_aliases.append(name)
    return {"unregistered": sorted(unregistered), "missing_aliases": sorted(missing_aliases)}


def apply_missing(current: dict[str, Any], missing: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Merge missing entries into a copy of the current aliases dict."""
    updated = json.loads(json.dumps(current))  # deep copy, preserves _comment
    for section in ("document_hints", "cn_alias_map"):
        updated.setdefault(section, {})
        for key, value in missing[section].items():
            updated[section][key] = value
    return updated
