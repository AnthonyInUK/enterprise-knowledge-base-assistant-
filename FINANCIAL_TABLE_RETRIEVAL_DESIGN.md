# Financial-table retrieval & extraction — design notes

How the RAG handles financial tables today, why precise numeric Q&A still fails
on some documents, and the target design. Written after diagnosing a class of
wrong-number answers (e.g. Vestas EBIT) that survived table-aware parsing.

## How retrieval works today

Hybrid: BM25 (lexical) + vector, **scored per chunk** ([`RetrievalService.retrieve`](rag_assistant/core.py)).
It only measures "how similar is this text to the query" — there is **no notion
of document structure**. A chunk is just a span of text.

A query like "EBIT / operating profit / 2023" matches *several* chunks at once —
the main income statement, the segment table, the five-year summary, cash-flow
notes — so they are **all retrieved**. The retriever has no idea which chunk is
the **consolidated income statement**, so it cannot prefer it. That capability —
"prefer the primary consolidated statement" — **we do not have**.

## How tables are handled today

`append_structured_tables` ([ingest_energy_data.py](scripts/ingest_energy_data.py)):
for each page, `fitz.find_tables()` serializes each row as `cell | cell | ...`
prefixed with `[TABLE]`.

This achieves **row adjacency** — a label stays glued to its row's numbers (this
fixed the scrambling seen on Tesla's clean single-table income statement). But it
does **not** capture **table semantics**:

1. **No table caption** — no "Consolidated Income Statement" identity marker.
2. **No column↔year binding** — which column is 2023 vs 2022 is lost.
3. **No separation of multiple tables on one page** — on a dense Vestas page,
   fitz merged the cash-flow adjustments table + a subsidiary list + the income
   statement into one `[TABLE]` blob with inconsistent `|` counts.

This is "table-semantic structured extraction" — **also not done**. What we do is
"row serialization".

## Why Vestas fails but Tesla passes

| | Tesla 10-K | Vestas annual report |
|---|---|---|
| Income-statement layout | single, ruled, regular | many tables per page, dense |
| fitz extraction | clean single table, columns↔years clear | tables merged, columns↔years lost |
| Same-label rows | mostly unique | "Operating profit (EBIT)" appears in main (482) / segment / five-year-summary (292) |
| Result | label glued to right value → correct | several (482)/292 retrieved, no table identity → model picks wrong one |

**In one line:** row serialization solves "single-table scrambling" but not
"many similar tables coexisting + no table identity + no column-year binding."
Tesla is the former (fixed); Vestas is the latter. Switching LLM (incl.
`deepseek-reasoner`) does **not** help — both flash and reasoner answer 292,
because the ambiguity is in the data presented, not the model.

## Two improvement directions

1. **Structure-tagged retrieval (medium):** tag each chunk with a
   `statement_type` (consolidated income statement / balance sheet / cash flow vs
   segment / summary / notes) at ingestion, and add a retrieval scoring term that
   up-weights primary statements and down-weights segment/summary/notes. Solves
   most Vestas-class cases. Needs a classifier + one scoring term + re-ingest.

2. **Table-semantic structured extraction (the end goal):** stop flattening
   tables to text. Extract them as structured objects (table title + column years
   + row labels + cell values), normalize numbers (parentheses→negative, commas,
   scale/unit), and store them queryable by `(company, metric, year)`.

## Decision: structured extraction is the project's main line

Precise financial Q&A is the core value, so (2) is the end state, not a
nice-to-have. The good news: **the store already exists** —
`research_facts (company, metric, value, unit, period, source_citation,
confidence, review_status, metadata)` ([sql/009_research_fact_graph.sql](sql/009_research_fact_graph.sql))
with `facts.py` persistence. Today it is fed *post-hoc from generated answers*;
the target redirects its feed to **structured financial tables at ingestion**.

### Target architecture

```
ingestion: fitz tables → classify statement_type → map column headers→years,
           rows→metrics, normalize cells (parens-negative, commas, scale)
           → upsert research_facts(company, metric, period, value, statement_type, source)
answer:    financial question → resolve (company, metric, year) against research_facts
           (statement_type lets consolidated win over segment/summary)
           → exact hit returns; else fall back to LLM-over-text
```

This is the **correct version of the `_direct_fact_answer`** we disabled: the
same "answer financial facts deterministically, bypass LLM drift" strategy, but
backed by clean structured data instead of fragile regex over flattened text.

### Roadmap

1. **Prototype (done):** hand-load a few income-statement facts into
   `research_facts`; add a facts-first lookup stage in `_compose_answer`
   (`_lookup_fact`, backend tag `facts-store`). Verified: Vestas EBIT → -482,
   margin → -3.1%, Tesla R&D → 3,969 — all previously wrong, now correct;
   non-financial questions correctly fall through to the LLM.
2. **Table→facts extractor (the heavy lift):** statement-type classification +
   column-year mapping + value normalization at ingestion.
3. **Full re-ingest** to populate `research_facts` (combine with the HTML
   noise-cleaning re-ingest to pay the re-embed cost once).
4. **Facts-first answering** for financial questions, LLM fallback otherwise.
5. Measure with the LLM-judge correctness eval ([scripts/eval_answer_correctness.py](scripts/eval_answer_correctness.py)).

## Findings from the extractor prototype (step 1+2 probe)

Built a throwaway extractor (fitz tables → deepseek-chat structured JSON →
grounding check) and ran it on Tesla and Vestas. What it established:

1. **LLM extraction works** — deepseek-chat (not v4-flash, which ignores the
   "only these metrics" constraint and overflows max_tokens), constrained to a
   bounded metric vocabulary, with grounding (value digits must appear in the
   table) produces clean structured facts.
2. **Column→year mapping is solvable** — the model mislabels years by one
   column unless anchored; passing the document fiscal year ("the leftmost data
   column is FY2023, then FY-1…") fixes it.
3. **Primary-statement selection is solvable-ish** — scan all tables, score by
   P&L line-item completeness, and penalise multi-year summaries (≥4 year
   columns). This picks the consolidated income statement over segment/five-year
   tables. **Tesla end-to-end is fully correct** (revenue 96,773 / automotive
   82,419 / R&D 3,969 / net income attributable 14,997, all FY2023).
4. **The real ceiling is upstream table-extraction quality.** On Vestas' dense,
   note-referenced consolidated income statement, `fitz.find_tables()` itself
   misaligns cells (revenue figures land on the gross-profit row;
   "Operating profit/(loss) (EBIT) | 61 | (444) (1,596)" is garbled). When the
   reconstructed grid is wrong, *every* derived value is unreliable — no amount
   of table selection, LLM choice, or prompting recovers it. The "-482" used
   earlier (and the golden label) likely came from such a mis-extracted /
   highlights table.

### Decision

- **facts-store is the main line.** Adopt it now for documents fitz extracts
  cleanly (most regular annual reports — Tesla-class).
- **fitz-defeating documents (Vestas-class) are out of scope until a stronger
  table parser is in.** A better extractor (camelot lattice / vision-LLM table
  parsing / commercial) is a separate infra item. Until then these tables are
  low-confidence → fall back to LLM-over-text + human review
  (`research_facts.review_status`); do **not** persist mis-extracted values as
  authoritative facts.
- **Guardrail at extraction time:** validate before persisting — grounding
  (value present in the table), plus sanity/articulation checks (e.g.
  gross_profit ≤ revenue, subtotal reconciliation). Failing rows → skip or mark
  `pending`, never `approved`.

### Status

- Facts-first lookup (`_lookup_fact`, Stage 0 of `_compose_answer`) is in place.
- Seeded facts: Tesla revenue/net-income/R&D (verified against a clean table,
  kept `approved`). The Vestas EBIT/-margin seeds were **removed** — their
  source table is fitz-garbled, so the values are not trustworthy (consistent
  with the guardrail above).
- The automatic table→facts extractor (step 2) is **not** built into ingestion
  yet; the prototype lives outside the repo. Next: build it for fitz-clean docs
  with the year-anchor + primary-statement-selection + guardrails proven here.
