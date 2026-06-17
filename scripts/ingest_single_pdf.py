"""
ingest_single_pdf.py — 直接把任意 PDF 注入数据库，无需修改 manifest

用法:
    python scripts/ingest_single_pdf.py <PDF路径> --title "Tesla 2023 10-K" --company tesla

示例:
    python scripts/ingest_single_pdf.py \
        "data/raw/UNITED STATES SECURITIES AND EXCHANGE COMMISSION.pdf" \
        --title "Tesla 2023 Annual Report 10-K" \
        --company tesla

注意: ingest 完之后还需要跑 embed_chunks.py 生成真实向量。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import psycopg
from dotenv import load_dotenv

# 复用 ingest_energy_data 里的所有解析逻辑
from scripts.ingest_energy_data import (
    ParagraphBlock,
    PdfBlock,
    clean_chunk_text,
    embed_text,
    extract_pdf_document,
    normalize_whitespace,
    sha256_text,
    split_blocks_into_sections,
    split_paragraph_text,
    vector_literal,
)

# Chunking parameters — tuned for financial PDFs (10-K, annual reports)
# max_words=400: ~400 words per chunk, ~2-3 paragraphs, good for RAG context
# overlap=50:    50-word overlap to avoid cutting mid-argument
CHUNK_MAX_WORDS = 400
CHUNK_OVERLAP = 50

load_dotenv()


def parser_fingerprint() -> str:
    payload = {
        "parser": "ingest_single_pdf.extract_pdf_document",
        "parser_version": "fitz-tables-clean-v2",  # bump when parsing logic changes
        "chunk_max_words": CHUNK_MAX_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_parse_cache(database_url: str, file_hash: str) -> tuple[list[PdfBlock], str, bool, str, int, int, dict] | None:
    if os.getenv("RAG_DOCUMENT_PARSE_CACHE", "1") != "1":
        return None
    try:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, parse_method, ocr_used, ocr_reason,
                       extracted_char_count, ocr_char_count, quality, blocks
                FROM document_parse_cache
                WHERE file_hash = %s AND parser_fingerprint = %s
                """,
                (file_hash, parser_fingerprint()),
            )
            row = cur.fetchone()
            if not row:
                return None
            cache_id, parse_method, ocr_used, ocr_reason, extracted_chars, ocr_chars, quality, blocks_json = row
            cur.execute(
                """
                UPDATE document_parse_cache
                SET hit_count = hit_count + 1, updated_at = now()
                WHERE id = %s
                """,
                (cache_id,),
            )
            conn.commit()
    except Exception:
        return None
    blocks = [
        PdfBlock(page_number=int(item["page_number"]), text=str(item["text"]))
        for item in blocks_json
    ]
    return blocks, str(parse_method), bool(ocr_used), str(ocr_reason or ""), int(extracted_chars or 0), int(ocr_chars or 0), dict(quality or {})


def _write_parse_cache(
    database_url: str,
    file_hash: str,
    file_name: str,
    blocks: list[PdfBlock],
    parse_method: str,
    ocr_used: bool,
    ocr_reason: str,
    extracted_chars: int,
    ocr_chars: int,
    quality: dict,
) -> None:
    if os.getenv("RAG_DOCUMENT_PARSE_CACHE", "1") != "1":
        return
    blocks_json = [
        {"page_number": block.page_number, "text": block.text}
        for block in blocks
    ]
    try:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_parse_cache (
                    file_hash, parser_fingerprint, file_name, parse_method,
                    ocr_used, ocr_reason, extracted_char_count, ocr_char_count,
                    quality, blocks, metadata, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, now())
                ON CONFLICT (file_hash, parser_fingerprint)
                DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    parse_method = EXCLUDED.parse_method,
                    ocr_used = EXCLUDED.ocr_used,
                    ocr_reason = EXCLUDED.ocr_reason,
                    extracted_char_count = EXCLUDED.extracted_char_count,
                    ocr_char_count = EXCLUDED.ocr_char_count,
                    quality = EXCLUDED.quality,
                    blocks = EXCLUDED.blocks,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    file_hash,
                    parser_fingerprint(),
                    file_name,
                    parse_method,
                    ocr_used,
                    ocr_reason,
                    extracted_chars,
                    ocr_chars,
                    json.dumps(quality, ensure_ascii=False),
                    json.dumps(blocks_json, ensure_ascii=False),
                    json.dumps({"source": "ingest_single_pdf"}, ensure_ascii=False),
                ),
            )
            conn.commit()
    except Exception:
        return


def ingest_pdf(
    pdf_path: Path,
    title: str,
    company: str = "",
    source_url: str = "",
    dry_run: bool = False,
) -> dict:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL 未设置，请检查 .env 文件")

    if not pdf_path.exists():
        raise SystemExit(f"文件不存在: {pdf_path}")

    print(f"[INFO] 解析 PDF: {pdf_path.name}")
    print(f"[INFO] 文档标题: {title}")

    file_hash = sha256_file(pdf_path)
    print(f"[INFO] 文件 SHA256: {file_hash[:16]}...")

    # 提取 PDF 内容。文件 hash + parser 配置不变时，直接复用解析缓存。
    cached_parse = _load_parse_cache(database_url, file_hash)
    if cached_parse:
        print("[INFO] 命中文档解析缓存，跳过 PDF/OCR 解析")
        blocks, parse_method, ocr_used, ocr_reason, extracted_chars, ocr_chars, quality = cached_parse
    else:
        blocks, parse_method, ocr_used, ocr_reason, extracted_chars, ocr_chars, quality = \
            extract_pdf_document(pdf_path)
        _write_parse_cache(
            database_url,
            file_hash,
            pdf_path.name,
            blocks,
            parse_method,
            ocr_used,
            ocr_reason,
            extracted_chars,
            ocr_chars,
            quality,
        )

    if not blocks:
        raise SystemExit("PDF 解析结果为空，可能是扫描件或加密文件")

    text = "\n\n".join(b.text for b in blocks if b.text.strip())
    content_hash = sha256_text(text)
    language = "en" if re.search(r"[A-Za-z]", text) and not re.search(r"[一-鿿]", text) else "zh"

    print(f"[INFO] 解析方式: {parse_method}  OCR: {ocr_used}")
    print(f"[INFO] 页数: {quality.get('page_count', 0)}  有效字符: {extracted_chars:,}")
    print(f"[INFO] 语言: {language}  质量分: {quality.get('quality_score', 0):.2f}")

    sections = split_blocks_into_sections(blocks)
    total_paragraphs = sum(len(s.paragraphs) for s in sections)
    print(f"[INFO] 切分出 {len(sections)} 个章节，{total_paragraphs} 个段落")

    if dry_run:
        print("[DRY-RUN] 跳过数据库写入")
        return {"sections": len(sections), "paragraphs": total_paragraphs}

    artifact_metadata = {
        "artifact_type": "pdf",
        "local_path": str(pdf_path.relative_to(ROOT) if pdf_path.is_relative_to(ROOT) else pdf_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "parse_method": parse_method,
        "ocr_used": ocr_used,
        "ocr_reason": ocr_reason,
        "extracted_char_count": extracted_chars,
        "ocr_char_count": ocr_chars,
        "page_quality": quality,
        "company": company,
        "source": "manual_ingest",
    }

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:

            # ── 1. upsert document ──────────────────────────────────────────
            cur.execute(
                """
                INSERT INTO documents (
                    source_type, source_url, title, file_name,
                    file_hash, doc_type, language, status, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (file_hash) DO UPDATE
                    SET title = EXCLUDED.title,
                        updated_at = now(),
                        metadata = EXCLUDED.metadata
                RETURNING id
                """,
                (
                    "manual_pdf", source_url or "", title, pdf_path.name,
                    file_hash, "annual_report", language, "ready",
                    json.dumps(artifact_metadata, ensure_ascii=False),
                ),
            )
            document_id = cur.fetchone()[0]
            print(f"[INFO] document_id = {document_id}")

            # ── 2. upsert document_version ──────────────────────────────────
            cur.execute(
                "SELECT id FROM document_versions WHERE content_hash = %s",
                (content_hash,),
            )
            existing = cur.fetchone()
            if existing:
                document_version_id = existing[0]
                print(f"[INFO] 文档内容未变，复用 document_version_id = {document_version_id}")
            else:
                cur.execute(
                    "SELECT COALESCE(MAX(version_no), 0) + 1 FROM document_versions WHERE document_id = %s",
                    (document_id,),
                )
                next_ver = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO document_versions (
                        document_id, version_no, raw_path, parsed_path,
                        content_hash, published_at, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, now(), %s::jsonb)
                    ON CONFLICT (content_hash) DO UPDATE SET ingested_at = now()
                    RETURNING id
                    """,
                    (
                        document_id, next_ver,
                        str(pdf_path.relative_to(ROOT) if pdf_path.is_relative_to(ROOT) else pdf_path),
                        None, content_hash,
                        json.dumps(artifact_metadata, ensure_ascii=False),
                    ),
                )
                document_version_id = cur.fetchone()[0]
                print(f"[INFO] 新建 document_version_id = {document_version_id}")

            # ── 3. 清空旧 chunks ────────────────────────────────────────────
            cur.execute("DELETE FROM chunks WHERE document_version_id = %s", (document_version_id,))

            # ── 4. 写入 chunks ──────────────────────────────────────────────
            chunk_index = 0
            total_chunks_written = 0

            for section_index, section in enumerate(sections):
                paragraph_chunks: list[ParagraphBlock] = []
                for para in section.paragraphs:
                    for split_text in split_paragraph_text(para.text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
                        paragraph_chunks.append(
                            ParagraphBlock(
                                text=split_text,
                                page_start=para.page_start,
                                page_end=para.page_end,
                            )
                        )

                chapter_text = clean_chunk_text(normalize_whitespace(
                    "\n\n".join(p.text for p in paragraph_chunks)
                ))
                if not chapter_text:
                    continue

                # chapter chunk
                cur.execute(
                    """
                    INSERT INTO chunks (
                        document_version_id, chunk_index, parent_chunk_id, chunk_level,
                        section_title, page_start, page_end, char_start, char_end,
                        token_count, chunk_type, text, metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING id
                    """,
                    (
                        document_version_id, chunk_index, None, "chapter",
                        section.title, section.page_start, section.page_end,
                        0, len(chapter_text), max(1, len(chapter_text.split())),
                        "pdf", chapter_text,
                        json.dumps({"chunk_level": "chapter", "section_index": section_index,
                                    "section_title": section.title}, ensure_ascii=False),
                    ),
                )
                chapter_id = cur.fetchone()[0]

                # bootstrap embedding for chapter
                cur.execute(
                    """
                    INSERT INTO embeddings (chunk_id, embedding_model, embedding_version, dimension, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE
                        SET embedding_model=EXCLUDED.embedding_model,
                            embedding=EXCLUDED.embedding, created_at=now()
                    """,
                    (chapter_id, "hash-bootstrap", "bootstrap-v1", 1024,
                     vector_literal(embed_text(chapter_text))),
                )
                chunk_index += 1
                total_chunks_written += 1

                # paragraph chunks
                for para_idx, para in enumerate(paragraph_chunks):
                    if not para.text.strip():
                        continue
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            document_version_id, chunk_index, parent_chunk_id, chunk_level,
                            section_title, page_start, page_end, char_start, char_end,
                            token_count, chunk_type, text, metadata
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        RETURNING id
                        """,
                        (
                            document_version_id, chunk_index, chapter_id, "paragraph",
                            section.title, para.page_start, para.page_end,
                            0, len(para.text), max(1, len(para.text.split())),
                            "pdf", para.text,
                            json.dumps({"chunk_level": "paragraph", "section_index": section_index,
                                        "paragraph_index": para_idx}, ensure_ascii=False),
                        ),
                    )
                    para_id = cur.fetchone()[0]
                    cur.execute(
                        """
                        INSERT INTO embeddings (chunk_id, embedding_model, embedding_version, dimension, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                            SET embedding_model=EXCLUDED.embedding_model,
                                embedding=EXCLUDED.embedding, created_at=now()
                        """,
                        (para_id, "hash-bootstrap", "bootstrap-v1", 1024,
                         vector_literal(embed_text(para.text))),
                    )
                    chunk_index += 1
                    total_chunks_written += 1

            # ── 5. ingestion_jobs 记录 ──────────────────────────────────────
            cur.execute(
                """
                INSERT INTO ingestion_jobs (
                    document_id, job_type, status, started_at, finished_at,
                    parse_method, ocr_used, ocr_reason,
                    extracted_char_count, ocr_char_count,
                    page_count, text_page_count, ocr_page_count,
                    quality_score, needs_review, metadata
                ) VALUES (%s,'manual_ingest','succeeded',now(),now(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                """,
                (
                    document_id, parse_method, ocr_used, ocr_reason,
                    extracted_chars, ocr_chars,
                    int(quality.get("page_count", 0)),
                    int(quality.get("text_page_count", 0)),
                    int(quality.get("ocr_page_count", 0)),
                    float(quality.get("quality_score", 0.0)),
                    bool(quality.get("needs_review", False)),
                    json.dumps(artifact_metadata, ensure_ascii=False),
                ),
            )

        conn.commit()

    print(f"[INFO] ✅ 写入完成！共 {total_chunks_written} 个 chunks")

    # ── 5. 自动抽取结构化财务事实（best-effort，不打断 ingestion）──────────
    # 从合并损益表抽 (company, metric, year) 事实写入 research_facts；只有
    # well-formed + 跨行勾稽通过的才 approved，否则 pending 待复核。需 DEEPSEEK
    # key，缺失则自动跳过。可用 RAG_EXTRACT_FACTS=0 关闭。
    facts_written = 0
    if os.getenv("RAG_EXTRACT_FACTS", "1") == "1":
        try:
            from rag_assistant.core import Database
            from rag_assistant.table_facts import extract_income_statement_facts, persist_facts
            facts = extract_income_statement_facts(str(pdf_path), company, title)
            if facts:
                facts_written = persist_facts(Database(), facts, title)
                approved = sum(1 for f in facts if f.reliable)
                print(f"[INFO] 📊 财务事实 {facts_written} 条入库（approved {approved} / pending {facts_written - approved}）")
        except Exception as exc:
            print(f"[WARN] 财务事实抽取跳过: {exc}")

    print(f"[INFO] 下一步：运行 embed_chunks.py 为新 chunks 生成真实向量")
    return {"document_id": str(document_id), "chunks": total_chunks_written, "facts": facts_written}


def main() -> None:
    parser = argparse.ArgumentParser(description="把单个 PDF 直接注入 RAG 数据库")
    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument("--title", required=True, help='文档标题，例如 "Tesla 2023 Annual Report 10-K"')
    parser.add_argument("--company", default="", help="公司标识符，例如 tesla / catl / byd")
    parser.add_argument("--source-url", default="", help="原始来源 URL（可选）")
    parser.add_argument("--dry-run", action="store_true", help="只解析不写入数据库")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path

    result = ingest_pdf(
        pdf_path=pdf_path,
        title=args.title,
        company=args.company,
        source_url=args.source_url,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
