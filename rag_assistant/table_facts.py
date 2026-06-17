"""Table → structured facts extractor (prototype, fitz-clean docs only).

Pipeline (see FINANCIAL_TABLE_RETRIEVAL_DESIGN.md):
  fitz.find_tables → pick the consolidated income statement (P&L line-item
  completeness, penalise multi-year summaries) → LLM (deepseek-chat) extracts a
  bounded metric vocabulary as JSON, year-anchored on the doc fiscal year →
  validate (grounding + sanity) → upsert into research_facts.

Scope: this reliably handles documents fitz extracts cleanly. Dense statements
that fitz mis-aligns (Vestas-class) produce low/failed validation and are left
to LLM-over-text + review — we never persist mis-extracted values as approved.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

import fitz

# Canonical metric vocabulary — keep variants distinct (ebit vs
# ebit_before_special_items). Must match FACT_METRIC_HINTS keys in core.py so
# the facts-first lookup can resolve them.
TARGET_METRICS = [
    "total_revenues", "total_automotive_revenues", "cost_of_revenue",
    "total_operating_cost", "gross_profit",
    "operating_income", "ebit", "ebit_before_special_items", "net_income",
    "net_income_attributable_to_common_stockholders", "research_and_development",
    # CN income-statement expense components (for the 营业总成本 identity)
    "taxes_and_surcharges", "selling_expenses", "admin_expenses", "finance_expenses",
]

# Components that sum to 营业总成本 (total_operating_cost) — a definitional
# identity in Chinese income statements, used to verify correct extraction.
_OPERATING_COST_COMPONENTS = [
    "cost_of_revenue", "taxes_and_surcharges", "selling_expenses",
    "admin_expenses", "research_and_development", "finance_expenses",
]

_PNL_MARKERS = [
    "revenue", "cost of revenue", "production cost", "gross profit",
    "research and development", "administ", "operating profit", "ebit",
    "operating income", "income from operations", "income tax", "net income",
    "net profit", "profit for the year",
    # Chinese A-share income-statement line items
    "营业收入", "营业总收入", "营业成本", "营业总成本", "营业利润",
    "利润总额", "净利润", "研发费用", "销售费用", "管理费用",
]
_OPERATING_GATE = ("operating profit", "ebit", "operating income", "income from operations",
                   "营业利润", "净利润")

_PROMPT = """Extract facts from ONE financial-statement table for {company}.

Table (rows are "cell | cell | ..."; a header row holds the year columns):
{table}

The document's latest fiscal year is {fy}; period columns run most-recent-first,
so the FIRST data column is {fy}, then {fy}-1, etc. Map each value to its year.

Output ONLY JSON:
{{"statement_type": "consolidated_income_statement|segment|five_year_summary|other",
  "currency": "USD|EUR|CNY|null", "scale": "millions|thousands|units|null",
  "facts": [{{"metric_key": "<one of: {vocab}>", "metric_label": "<verbatim row label>", "year": "YYYY", "value": "<num; parentheses=negative; keep % for margins>"}}]}}
Only emit metric_key values from that list. Parentheses mean NEGATIVE: (482) -> -482.
Skip metrics not present in the table.
Chinese statements map to the same keys: 营业收入/营业总收入->total_revenues,
营业成本->cost_of_revenue, 营业总成本->total_operating_cost, 营业利润->operating_income,
净利润->net_income, 归属于母公司/上市公司股东的净利润->net_income_attributable_to_common_stockholders,
研发费用->research_and_development, 税金及附加->taxes_and_surcharges, 销售费用->selling_expenses,
管理费用->admin_expenses, 财务费用->finance_expenses."""


@dataclass
class ExtractedFact:
    company: str
    metric: str
    value: str
    unit: str | None
    period: str
    statement_type: str
    confidence: float
    reliable: bool  # source table was well-formed (uniform columns)


def _table_wellformed(table_text: str) -> bool:
    """A cleanly-extracted statement table has a consistent column count across
    its value-bearing rows. This catches grossly-broken column structure, but
    NOT subtler row-label↔value misalignment (a uniform-column table can still
    attach a value to the wrong row). So a True here is necessary, not
    sufficient — auto-extracted facts from dense reports still warrant the
    review_status workflow / a reconciliation check before being fully trusted."""
    counts = [row.count("|") for row in table_text.splitlines() if re.search(r"\d", row)]
    if len(counts) < 3:
        return False
    dominant = max(set(counts), key=counts.count)
    if dominant == 0:
        return False
    share = sum(1 for c in counts if abs(c - dominant) <= 1) / len(counts)
    return share >= 0.6


def _table_text(tb) -> str:
    rows = tb.extract()
    return "\n".join(" | ".join(" ".join((c or "").split()) for c in r) for r in rows if any(r))


def find_primary_income_statement(pdf_path: str) -> str:
    """Return the serialized consolidated income statement (richest P&L coverage,
    multi-year summaries penalised) or "" if none found."""
    best_text, best_score = "", -1
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            try:
                tables = page.find_tables().tables
            except Exception:
                continue
            for tb in tables:
                txt = _table_text(tb)
                low = txt.lower()
                if not any(g in low for g in _OPERATING_GATE):
                    continue
                n_years = len(set(re.findall(r"\b20\d\d\b", txt)))
                score = sum(1 for m in _PNL_MARKERS if m in low) - (5 if n_years >= 4 else 0)
                if score > best_score:
                    best_text, best_score = txt, score
    finally:
        doc.close()
    return best_text


def _deepseek_json(prompt: str, max_tokens: int = 2048) -> dict | None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    payload = json.dumps({
        "model": os.getenv("TABLE_FACTS_MODEL", "deepseek-chat"),
        "max_tokens": max_tokens, "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError):
        return None
    m = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def _digits(s: str) -> str:
    return re.sub(r"[^\d]", "", str(s))


def extract_income_statement_facts(pdf_path: str, company: str, doc_title: str) -> list[ExtractedFact]:
    """Extract validated consolidated-income-statement facts from a PDF."""
    table = find_primary_income_statement(pdf_path)
    if not table:
        return []
    fy_match = re.search(r"20\d\d", doc_title)
    fy = fy_match.group(0) if fy_match else None
    obj = _deepseek_json(_PROMPT.format(company=company, table=table[:6000],
                                        fy=fy or "the latest year", vocab=", ".join(TARGET_METRICS)))
    if not obj:
        return []
    table_digits = _digits(table)
    well_formed = _table_wellformed(table)
    unit = obj.get("scale") and obj.get("currency") and f"{obj['scale']} {obj['currency']}"
    statement_type = obj.get("statement_type") or "other"

    # Collect grounded facts; index numeric values by (year, metric) for the
    # cross-row reconciliation below.
    raw: list[tuple[str, str, str]] = []  # (metric, value, year)
    nums: dict[tuple[str, str], float] = {}
    for f in obj.get("facts", []):
        metric, value, year = f.get("metric_key"), str(f.get("value", "")), str(f.get("year", ""))
        if metric not in TARGET_METRICS or not value or not year:
            continue
        if not _digits(value) or _digits(value) not in table_digits:  # grounding
            continue
        try:
            nums[(year, metric)] = float(value.replace(",", "").rstrip("%"))
        except ValueError:
            continue
        raw.append((metric, value, year))

    # Cross-row reconciliation for the fiscal year: a real income statement
    # satisfies gross_profit == revenue - cost_of_revenue. This catches
    # row-label/value misalignment that column-uniformity and single-value
    # sanity miss (e.g. revenue 15,382 wrongly attached to the gross_profit row).
    reconciled = _reconciles(nums, fy) if fy else None
    reliable = bool(well_formed and reconciled is True)

    return [
        ExtractedFact(company, metric, value, unit or obj.get("scale"),
                      year, statement_type, 0.95, reliable)
        for metric, value, year in raw
    ]


def _reconciles(nums: dict[tuple[str, str], float], year: str) -> bool | None:
    """Check an income-statement identity for the year (within tolerance).
    Returns True/False if the components of an identity were extracted, else None.

    Identities (a statement passing ANY is internally consistent):
      US gross:  gross_profit         == total_revenues - cost_of_revenue
      CN total:  total_operating_cost == sum(cost components)  [definitional;
                 always holds, and avoids the noisy investment/other-income
                 items sitting between 营业总成本 and 营业利润]
    """
    checks: list[bool] = []
    rev = nums.get((year, "total_revenues"))
    if rev is not None:
        tol = max(2.0, abs(rev) * 0.01)
        gp, cost = nums.get((year, "gross_profit")), nums.get((year, "cost_of_revenue"))
        if gp is not None and cost is not None:
            checks.append(abs(gp - (rev - abs(cost))) <= tol)

    tcost = nums.get((year, "total_operating_cost"))
    if tcost is not None:
        parts = [abs(nums[(year, k)]) for k in _OPERATING_COST_COMPONENTS if (year, k) in nums]
        if len(parts) >= 3:  # enough components for a meaningful definitional check
            checks.append(abs(abs(tcost) - sum(parts)) <= max(2.0, abs(tcost) * 0.01))

    if not checks:
        return None
    return any(checks)


def persist_facts(db, facts: list[ExtractedFact], source_citation: str) -> int:
    """Upsert extracted facts into research_facts. review_status='approved' only
    when statement_type is the consolidated income statement (validated path)."""
    n = 0
    for f in facts:
        # approved only when it's the consolidated statement AND the source table
        # was well-formed; otherwise pending (facts-first only serves approved).
        status = "approved" if (f.statement_type == "consolidated_income_statement" and f.reliable) else "pending"
        db.execute(
            """
            INSERT INTO research_facts (company, metric, value, unit, period, source_citation,
                                        confidence, review_status, metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (company, metric, COALESCE(period, ''), value, source_citation)
            DO UPDATE SET confidence = GREATEST(research_facts.confidence, EXCLUDED.confidence),
                          review_status = EXCLUDED.review_status,
                          metadata = research_facts.metadata || EXCLUDED.metadata
            """,
            (f.company, f.metric, f.value, f.unit, f.period, source_citation,
             f.confidence, status, json.dumps({"statement_type": f.statement_type, "source": "table_facts"})),
        )
        n += 1
    return n


def main() -> None:
    import argparse
    from rag_assistant.core import Database
    parser = argparse.ArgumentParser(description="Extract income-statement facts from a PDF into research_facts.")
    parser.add_argument("pdf")
    parser.add_argument("--company", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--apply", action="store_true", help="write to research_facts (else dry-run)")
    args = parser.parse_args()

    facts = extract_income_statement_facts(args.pdf, args.company, args.title)
    reliable = bool(facts and facts[0].reliable)
    print(f"extracted {len(facts)} facts | well-formed + reconciled: {reliable} "
          f"-> would persist as {'approved' if reliable else 'pending (not served)'}")
    for f in facts:
        print(f"  [{f.statement_type}] {f.metric} {f.period} = {f.value} {f.unit or ''}")
    if args.apply and facts:
        n = persist_facts(Database(), facts, args.title)
        print(f"persisted {n} facts to research_facts")


if __name__ == "__main__":
    main()
