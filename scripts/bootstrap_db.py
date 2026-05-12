from __future__ import annotations

import os
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
SCHEMA_PATH = ROOT / "sql" / "schema.sql"


def main() -> None:
    load_env_file(ROOT / ".env")
    if "DATABASE_URL" not in os.environ:
        load_env_file(ROOT / ".env.example")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is missing. Create .env from .env.example first.")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            cur.execute(
                """
                ALTER TABLE IF EXISTS chunks
                    ADD COLUMN IF NOT EXISTS parent_chunk_id uuid,
                    ADD COLUMN IF NOT EXISTS chunk_level text NOT NULL DEFAULT 'paragraph';

                CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id ON chunks(parent_chunk_id);

                ALTER TABLE IF EXISTS ingestion_jobs
                    ADD COLUMN IF NOT EXISTS parse_method text,
                    ADD COLUMN IF NOT EXISTS ocr_used boolean NOT NULL DEFAULT false,
                    ADD COLUMN IF NOT EXISTS ocr_reason text,
                    ADD COLUMN IF NOT EXISTS extracted_char_count int,
                    ADD COLUMN IF NOT EXISTS ocr_char_count int,
                    ADD COLUMN IF NOT EXISTS page_count int,
                    ADD COLUMN IF NOT EXISTS text_page_count int,
                    ADD COLUMN IF NOT EXISTS ocr_page_count int,
                    ADD COLUMN IF NOT EXISTS quality_score numeric(5,4),
                    ADD COLUMN IF NOT EXISTS needs_review boolean NOT NULL DEFAULT false;
                """
            )
        conn.commit()

    print(f"initialized schema from {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
