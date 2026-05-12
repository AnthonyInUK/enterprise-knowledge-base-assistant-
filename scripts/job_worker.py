from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_assistant.jobs import JobWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PostgreSQL-backed async job worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Idle sleep seconds.")
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args()

    worker = JobWorker(worker_id=args.worker_id)
    while True:
        result = worker.run_once()
        if result:
            print(json.dumps(result, ensure_ascii=False))
        elif args.once:
            print(json.dumps({"status": "idle"}, ensure_ascii=False))
        if args.once:
            return
        if result is None:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
