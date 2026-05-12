from __future__ import annotations

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_assistant.core import RetrievalService


def main() -> None:
    service = RetrievalService()
    print(json.dumps(service.warmup(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
