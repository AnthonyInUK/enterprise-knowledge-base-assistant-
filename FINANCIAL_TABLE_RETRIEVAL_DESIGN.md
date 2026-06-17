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

### Status

- Prototype facts-first lookup is in `_compose_answer` (Stage 0). Facts are
  currently hand-seeded for Tesla/Vestas to validate end-to-end; the automatic
  table→facts extractor (step 2) is not built yet.
