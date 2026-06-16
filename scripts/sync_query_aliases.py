"""Sync data/config/query_aliases.json from the corpus source manifest.

Reads companies (and their human-recorded Chinese ``aliases``) from
data/sources/energy_sources.json and derives the retrieval alias entries.
English tokens and tag-based domain terms are generated deterministically;
Chinese aliases come only from the manifest — never guessed.

Default is dry-run (print the diff). Pass --apply to write the file.

    python -m scripts.sync_query_aliases            # show what's missing
    python -m scripts.sync_query_aliases --apply    # write missing entries
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json

from rag_assistant.alias_sync import (
    DEFAULT_ALIASES_PATH,
    DEFAULT_SOURCES_PATH,
    apply_missing,
    build_expected,
    coverage_report,
    diff_missing,
    load_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync query aliases from the corpus manifest.")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH))
    parser.add_argument("--aliases", default=str(DEFAULT_ALIASES_PATH))
    parser.add_argument("--apply", action="store_true", help="write missing entries to the aliases file")
    args = parser.parse_args()

    sources = load_json(args.sources)
    current = load_json(args.aliases)

    expected = build_expected(sources)
    missing = diff_missing(expected, current)
    report = coverage_report(sources, current)

    n_missing = len(missing["document_hints"]) + len(missing["cn_alias_map"])
    if n_missing == 0:
        print("✅ query_aliases.json is up to date — nothing to add.")
    else:
        print(f"Found {n_missing} alias entr{'y' if n_missing == 1 else 'ies'} to add:\n")
        for section in ("document_hints", "cn_alias_map"):
            for key, value in missing[section].items():
                print(f"  + {section}: {json.dumps(key, ensure_ascii=False)} -> {json.dumps(value, ensure_ascii=False)}")

    if expected["todos"]:
        print("\n⚠️  Companies with no recorded Chinese alias (NOT guessed — add an")
        print("    \"aliases\" field in energy_sources.json so Chinese queries match):")
        for name in sorted(expected["todos"]):
            print(f"    - {name}")

    if report["unregistered"]:
        print(f"\n❌ Unregistered companies (English token missing): {report['unregistered']}")

    if args.apply:
        if n_missing == 0:
            return
        updated = apply_missing(current, missing)
        Path(args.aliases).write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n💾 Wrote {n_missing} entr{'y' if n_missing == 1 else 'ies'} to {args.aliases}")
    elif n_missing > 0:
        print("\n(dry-run) re-run with --apply to write these entries.")


if __name__ == "__main__":
    main()
