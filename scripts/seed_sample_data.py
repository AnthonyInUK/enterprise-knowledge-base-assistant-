from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
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
RAW_DOC_DIR = ROOT / "data" / "raw" / "seed_docs"


@dataclass
class SeedChunk:
    title: str
    text: str
    page_start: int = 1
    page_end: int = 1


def normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def split_markdown(content: str) -> list[SeedChunk]:
    lines = content.splitlines()
    sections: list[SeedChunk] = []
    current_title = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(
                SeedChunk(
                    title=current_title,
                    text=normalize_whitespace("\n".join(buffer)),
                )
            )

    for line in lines:
        if line.startswith("#"):
            flush()
            buffer.clear()
            current_title = line.lstrip("#").strip() or "Overview"
            continue
        buffer.append(line)

    flush()
    return [chunk for chunk in sections if chunk.text]


def embed_text(text: str, dimension: int = 1024) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dimension):
        byte = digest[index % len(digest)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    return values


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"


def main() -> None:
    load_env_file(ROOT / ".env")
    if "DATABASE_URL" not in os.environ:
        load_env_file(ROOT / ".env.example")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is missing. Create .env from .env.example first.")

    embedding_model = os.getenv("EMBEDDING_MODEL", "hash-bootstrap")
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

    markdown_files = sorted(RAW_DOC_DIR.glob("*.md"))
    if not markdown_files:
        raise SystemExit(f"no seed docs found in {RAW_DOC_DIR}")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for path in markdown_files:
                content = path.read_text(encoding="utf-8")
                doc_hash = file_hash(content)
                parsed_chunks = split_markdown(content)

                cur.execute(
                    """
                    INSERT INTO documents (source_type, source_url, title, file_name, file_hash, doc_type, language, status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (file_hash) DO UPDATE
                    SET updated_at = now()
                    RETURNING id
                    """,
                    (
                        "local_file",
                        str(path),
                        path.stem.replace("_", " "),
                        path.name,
                        doc_hash,
                        "report",
                        "zh",
                        "ready",
                        "{}",
                    ),
                )
                document_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO document_versions (document_id, version_no, raw_path, parsed_path, content_hash, published_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, now(), %s::jsonb)
                    ON CONFLICT (content_hash) DO UPDATE
                    SET ingested_at = now()
                    RETURNING id
                    """,
                    (document_id, 1, str(path), None, content_hash(content), "{}"),
                )
                document_version_id = cur.fetchone()[0]

                cur.execute(
                    "DELETE FROM chunks WHERE document_version_id = %s",
                    (document_version_id,),
                )

                for chunk_index, chunk in enumerate(parsed_chunks):
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            document_version_id,
                            chunk_index,
                            section_title,
                            page_start,
                            page_end,
                            char_start,
                            char_end,
                            token_count,
                            chunk_type,
                            text,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        RETURNING id
                        """,
                        (
                            document_version_id,
                            chunk_index,
                            chunk.title,
                            chunk.page_start,
                            chunk.page_end,
                            0,
                            len(chunk.text),
                            max(1, len(chunk.text.split())),
                            "paragraph",
                            chunk.text,
                            "{}",
                        ),
                    )
                    chunk_id = cur.fetchone()[0]
                    embedding = vector_literal(embed_text(chunk.text, embedding_dimension))
                    cur.execute(
                        """
                        INSERT INTO embeddings (
                            chunk_id,
                            embedding_model,
                            embedding_version,
                            dimension,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET embedding_model = EXCLUDED.embedding_model,
                            embedding_version = EXCLUDED.embedding_version,
                            dimension = EXCLUDED.dimension,
                            embedding = EXCLUDED.embedding,
                            created_at = now()
                        """,
                        (
                            chunk_id,
                            embedding_model,
                            "bootstrap-v1",
                            embedding_dimension,
                            embedding,
                        ),
                    )

                cur.execute(
                    """
                    INSERT INTO ingestion_jobs (document_id, job_type, status, started_at, finished_at, metadata)
                    VALUES (%s, %s, %s, now(), now(), %s::jsonb)
                    """,
                    (document_id, "seed_ingestion", "succeeded", "{}"),
                )

        conn.commit()

    print(f"seeded {len(markdown_files)} documents from {RAW_DOC_DIR}")


if __name__ == "__main__":
    main()
