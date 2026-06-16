"""LLM-as-judge answer-correctness eval.

Unlike eval_runner (token overlap) and evaluate_retrieval (chunk hit@k), this
measures the thing that actually matters: **is the generated answer correct?**

For each golden case it runs the full RAG answer, then asks a judge LLM to
compare that answer against the case's known facts (expected_keywords + notes)
and return a verdict: CORRECT / PARTIAL / WRONG, with a reason. Keyword presence
alone is not enough — the judge catches answers that mention the right number
while stating it wrongly ("96,773 was the 2022 figure", "revenue was not …").

    python -m scripts.eval_answer_correctness                 # all cases
    python -m scripts.eval_answer_correctness --limit 10      # quick sample
    python -m scripts.eval_answer_correctness --json          # machine output

Judge backend/model: QA_EVAL_JUDGE_BACKEND (default deepseek),
QA_EVAL_DEEPSEEK_JUDGE_MODEL (default DEEPSEEK_MODEL). The judge is grounded on
the known facts, so it scores correctness against ground truth, not taste.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import re
import time

from rag_assistant.core import RetrievalService

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"

JUDGE_PROMPT = """You are a strict grader for a retrieval-augmented QA system.

Question:
{question}

Known correct facts (ground truth — the answer MUST be consistent with these):
{facts}

System's answer:
{answer}

Decide whether the system's answer correctly answers the question and is
consistent with the known facts. A number appearing in the answer is NOT enough
— it must be stated as the correct answer to THIS question (right metric, right
year, right entity), not denied, not attributed to a different period.

Reply with ONLY a JSON object, no prose:
{{"verdict": "CORRECT" | "PARTIAL" | "WRONG", "reason": "<one short sentence>"}}
CORRECT = fully answers with the right fact. PARTIAL = right direction but
incomplete or imprecise. WRONG = missing, contradicted, or fabricated."""

SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}


def _judge(service: RetrievalService, question: str, facts: str, answer: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question, facts=facts, answer=answer)
    backend = os.getenv("QA_EVAL_JUDGE_BACKEND", "deepseek").lower()
    judge_model = os.getenv("QA_EVAL_DEEPSEEK_JUDGE_MODEL")
    prev_model = os.environ.get("DEEPSEEK_MODEL")
    if judge_model:
        os.environ["DEEPSEEK_MODEL"] = judge_model
    try:
        raw = service._call_claude_api(prompt) if backend == "claude" else service._call_deepseek_api(prompt)
    finally:
        if judge_model and prev_model is not None:
            os.environ["DEEPSEEK_MODEL"] = prev_model
    if not raw:
        return "WRONG", "judge call failed"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return "WRONG", f"unparseable judge output: {raw[:80]}"
    try:
        obj = json.loads(match.group(0))
        verdict = str(obj.get("verdict", "WRONG")).upper()
        return (verdict if verdict in SCORE else "WRONG"), str(obj.get("reason", ""))
    except ValueError:
        return "WRONG", f"unparseable judge output: {raw[:80]}"


def run(limit: int | None = None, top_k: int = 6) -> dict:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = data["cases"][:limit] if limit else data["cases"]
    service = RetrievalService()

    rows = []
    for case in cases:
        facts = "; ".join(case.get("expected_keywords") or []) or "(none provided)"
        if case.get("notes"):
            facts += f"  [note: {case['notes']}]"
        result = service.answer(case["question"], top_k=top_k)
        verdict, reason = _judge(service, case["question"], facts, result.answer)
        rows.append({
            "id": case.get("id"),
            "question": case["question"],
            "verdict": verdict,
            "score": SCORE[verdict],
            "reason": reason,
            "grounded": result.debug.get("answer_grounded"),
        })

    n = len(rows)
    correctness = sum(r["score"] for r in rows) / n if n else 0.0
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in SCORE}
    return {
        "n": n,
        "correctness": round(correctness, 4),
        "counts": counts,
        "answer_backend": os.getenv("RAG_LLM_BACKEND", "auto"),
        "judge": os.getenv("QA_EVAL_DEEPSEEK_JUDGE_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge answer-correctness eval.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    summary = run(limit=args.limit, top_k=args.top_k)
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"answer backend: {summary['answer_backend']} | judge: {summary['judge']}")
    print(f"cases: {summary['n']}  |  correctness: {summary['correctness']:.1%}")
    print(f"  CORRECT={summary['counts']['CORRECT']}  PARTIAL={summary['counts']['PARTIAL']}  WRONG={summary['counts']['WRONG']}")
    print(f"  ({summary['elapsed_s']}s)\n")
    for r in summary["rows"]:
        mark = {"CORRECT": "✅", "PARTIAL": "🟡", "WRONG": "❌"}[r["verdict"]]
        print(f"  {mark} {r['question'][:60]}")
        if r["verdict"] != "CORRECT":
            print(f"       → {r['reason']}")


if __name__ == "__main__":
    main()
