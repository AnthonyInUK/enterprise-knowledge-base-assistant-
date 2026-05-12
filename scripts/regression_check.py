"""
Run a lightweight retrieval regression gate.

This script is intentionally CI-friendly: it exits non-zero when key metrics
drop below thresholds.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.evaluate_retrieval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval regression checks.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "golden_dataset.json"))
    parser.add_argument("--strategy", default="full")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-hit-at-5", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.85)
    parser.add_argument("--min-page-hit-at-5", type=float, default=0.90)
    args = parser.parse_args()

    result = evaluate(
        dataset_path=Path(args.dataset),
        strategies=[args.strategy],
        top_k=args.top_k,
        with_answer_eval=False,
    )
    metrics = result["metrics"][args.strategy]
    failures: list[str] = []
    if metrics.hit_at_5 < args.min_hit_at_5:
        failures.append(f"Hit@5 {metrics.hit_at_5:.3f} < {args.min_hit_at_5:.3f}")
    if metrics.mrr < args.min_mrr:
        failures.append(f"MRR {metrics.mrr:.3f} < {args.min_mrr:.3f}")
    if metrics.page_hit_at_5 < args.min_page_hit_at_5:
        failures.append(f"P.Hit@5 {metrics.page_hit_at_5:.3f} < {args.min_page_hit_at_5:.3f}")

    print(
        "Regression metrics: "
        f"Hit@5={metrics.hit_at_5:.3f} "
        f"MRR={metrics.mrr:.3f} "
        f"P.Hit@5={metrics.page_hit_at_5:.3f}"
    )
    if failures:
        print("Regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Regression check passed.")


if __name__ == "__main__":
    main()
