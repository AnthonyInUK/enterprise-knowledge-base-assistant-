from __future__ import annotations

import json
import os
import re
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


KNOWN_ENTITY_TYPES = {
    "宁德时代": "company",
    "比亚迪": "company",
    "阳光电源": "company",
    "隆基绿能": "company",
    "Tesla": "company",
    "First Solar": "company",
    "Enphase": "company",
    "Vestas": "company",
    "Orsted": "company",
    "储能": "domain",
    "动力电池": "domain",
    "光伏": "domain",
    "风电": "domain",
    "逆变器": "product",
    "储能系统": "product",
    "光伏组件": "product",
    "电动车": "domain",
}

RELATION_HINTS = [
    ("包括", "includes"),
    ("覆盖", "covers"),
    ("属于", "belongs_to"),
    ("相关", "related_to"),
    ("布局", "invests_in"),
    ("生产", "produces"),
    ("研发", "develops"),
    ("专注", "focuses_on"),
    ("对标", "benchmarks"),
    ("合作", "partners_with"),
]


@dataclass
class MentionCandidate:
    entity_name: str
    entity_type: str
    start_offset: int
    end_offset: int


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def infer_entity_type(name: str) -> str:
    if name in KNOWN_ENTITY_TYPES:
        return KNOWN_ENTITY_TYPES[name]
    if re.search(r"(公司|集团|能源|电力|科技|电池|光伏|储能|风电)", name):
        return "company_or_domain"
    if re.search(r"(年报|季报|公告|报告|白皮书)", name):
        return "document"
    return "concept"


def extract_candidates(text: str) -> list[MentionCandidate]:
    candidates: list[MentionCandidate] = []
    for entity_name in sorted(KNOWN_ENTITY_TYPES, key=len, reverse=True):
        for match in re.finditer(re.escape(entity_name), text):
            candidates.append(
                MentionCandidate(
                    entity_name=entity_name,
                    entity_type=KNOWN_ENTITY_TYPES[entity_name],
                    start_offset=match.start(),
                    end_offset=match.end(),
                )
            )
    return candidates


def extract_relations(text: str) -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        for hint, relation_type in RELATION_HINTS:
            if hint not in line:
                continue
            if "：" in line:
                left, right = line.split("：", 1)
            elif ":" in line:
                left, right = line.split(":", 1)
            else:
                continue

            left = left.strip(" -*•")
            right = right.strip()
            if left and right:
                relations.append((left, relation_type, right))

    for line in lines:
        if "与" in line and "相关" in line:
            parts = re.split(r"[：:]", line, maxsplit=1)
            if len(parts) == 2:
                left = parts[0].strip(" -*•")
                right = parts[1].strip()
                relations.append((left, "related_to", right))

    return relations


def extract_cooccurrence_relations(text: str) -> list[tuple[str, str, str]]:
    candidates = extract_candidates(text)
    if not candidates:
        return []

    names_by_type: dict[str, list[str]] = {}
    for candidate in candidates:
        names_by_type.setdefault(candidate.entity_type, [])
        if candidate.entity_name not in names_by_type[candidate.entity_type]:
            names_by_type[candidate.entity_type].append(candidate.entity_name)

    companies = names_by_type.get("company", [])
    targets = (
        names_by_type.get("domain", [])
        + names_by_type.get("product", [])
        + names_by_type.get("company_or_domain", [])
    )

    relations: list[tuple[str, str, str]] = []
    for company in companies:
        for target in targets:
            if company != target:
                relations.append((company, "related_to", target))

    return relations


def jsonb_payload(value: object) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def get_or_create_entity(cur, name: str, entity_type: str, source_chunk_id: str | None) -> str:
    normalized_name = normalize_name(name)
    cur.execute(
        """
        INSERT INTO entities (name, entity_type, normalized_name, source_chunk_id, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (normalized_name, entity_type)
        DO UPDATE SET source_chunk_id = COALESCE(entities.source_chunk_id, EXCLUDED.source_chunk_id)
        RETURNING id
        """,
        (name, entity_type, normalized_name, source_chunk_id, "{}"),
    )
    return cur.fetchone()[0]


def main() -> None:
    load_env_file(ROOT / ".env")
    if "DATABASE_URL" not in os.environ:
        load_env_file(ROOT / ".env.example")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is missing. Create .env from .env.example first.")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.chunk_index, c.text, c.metadata, d.metadata
                FROM chunks c
                JOIN document_versions dv ON dv.id = c.document_version_id
                JOIN documents d ON d.id = dv.document_id
                ORDER BY c.created_at ASC
                """
            )
            chunks = cur.fetchall()

            for chunk_id, chunk_index, text, metadata, document_metadata in chunks:
                candidates = extract_candidates(text)
                if not candidates:
                    candidates = [
                        MentionCandidate(
                            entity_name="新能源行业",
                            entity_type="domain",
                            start_offset=0,
                            end_offset=4,
                        )
                    ]

                entity_ids: dict[tuple[str, str], str] = {}
                for candidate in candidates:
                    key = (normalize_name(candidate.entity_name), candidate.entity_type)
                    if key not in entity_ids:
                        entity_ids[key] = get_or_create_entity(
                            cur,
                            candidate.entity_name,
                            candidate.entity_type,
                            str(chunk_id),
                        )
                    cur.execute(
                        """
                        INSERT INTO entity_mentions (entity_id, chunk_id, start_offset, end_offset, metadata)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (entity_id, chunk_id, start_offset, end_offset) DO NOTHING
                        """,
                        (
                            entity_ids[key],
                            chunk_id,
                            candidate.start_offset,
                            candidate.end_offset,
                            "{}",
                        ),
                    )

                relations = extract_relations(text)
                relations.extend(extract_cooccurrence_relations(text))
                for head_name, relation_type, tail_name in relations:
                    head_type = infer_entity_type(head_name)
                    tail_type = infer_entity_type(tail_name)
                    head_id = get_or_create_entity(cur, head_name, head_type, str(chunk_id))
                    tail_id = get_or_create_entity(cur, tail_name, tail_type, str(chunk_id))
                    cur.execute(
                        """
                        INSERT INTO relations (
                            head_entity_id,
                            relation_type,
                            tail_entity_id,
                            source_chunk_id,
                            confidence,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (head_entity_id, relation_type, tail_entity_id, source_chunk_id)
                        DO UPDATE SET confidence = GREATEST(relations.confidence, EXCLUDED.confidence)
                        """,
                        (head_id, relation_type, tail_id, chunk_id, 0.7000, jsonb_payload(metadata)),
                    )

                if chunk_index == 0 and document_metadata:
                    company = document_metadata.get("company") or document_metadata.get("organization")
                    category = document_metadata.get("category")
                    region = document_metadata.get("region")

                    if company and category:
                        company_id = get_or_create_entity(cur, str(company), "company", str(chunk_id))
                        category_id = get_or_create_entity(cur, str(category), "category", str(chunk_id))
                        cur.execute(
                            """
                            INSERT INTO relations (
                                head_entity_id,
                                relation_type,
                                tail_entity_id,
                                source_chunk_id,
                                confidence,
                                metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (head_entity_id, relation_type, tail_entity_id, source_chunk_id)
                            DO NOTHING
                            """,
                            (company_id, "focuses_on", category_id, chunk_id, 0.9000, jsonb_payload(metadata)),
                        )

                    if company and region:
                        company_id = get_or_create_entity(cur, str(company), "company", str(chunk_id))
                        region_id = get_or_create_entity(cur, str(region), "region", str(chunk_id))
                        cur.execute(
                            """
                            INSERT INTO relations (
                                head_entity_id,
                                relation_type,
                                tail_entity_id,
                                source_chunk_id,
                                confidence,
                                metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (head_entity_id, relation_type, tail_entity_id, source_chunk_id)
                            DO NOTHING
                            """,
                            (company_id, "operates_in", region_id, chunk_id, 0.9000, jsonb_payload(metadata)),
                        )

        conn.commit()

    print(f"extracted graph data from {len(chunks)} chunks")


if __name__ == "__main__":
    main()
