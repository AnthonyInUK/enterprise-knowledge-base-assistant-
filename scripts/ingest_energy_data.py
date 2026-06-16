from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

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


MANIFEST_PATH = ROOT / "data" / "raw" / "energy" / "_collection_manifest.jsonl"


@dataclass
class Chunk:
    section_title: str
    text: str


@dataclass
class PdfBlock:
    page_number: int
    text: str


@dataclass
class ParagraphBlock:
    text: str
    page_start: int
    page_end: int


@dataclass
class Section:
    title: str
    paragraphs: list[ParagraphBlock]
    page_start: int
    page_end: int


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "p",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)
            self.parts.append(" ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonicalize_paragraph_for_dedupe(text: str) -> str:
    text = normalize_whitespace(text).casefold()
    text = re.sub(
        r"[\u2010-\u2015\-_/|·•*_=:;,.，。！？!?（）()\[\]{}<>]+", " ", text)
    text = re.sub(r"\b\d+(?:[.,:/-]\d+)*\b", " <num> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_table_block(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_lines: list[str] = []
    for line in lines:
        line = re.sub(r"\s{2,}", "\t", line)
        normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_text(text: str, dimension: int = 1024) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dimension):
        byte = digest[index % len(digest)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    return values


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"


def extract_html_text(raw_html: bytes) -> str:
    parser = TextExtractor()
    parser.feed(raw_html.decode("utf-8", errors="ignore"))
    return normalize_whitespace(parser.text())


def extract_pdf_pages_pdftotext(pdf_path: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split("\f")


def run_tesseract(image_path: Path) -> str:
    languages = ("chi_sim+eng", "eng")
    last_error: subprocess.CalledProcessError | None = None

    for language in languages:
        try:
            result = subprocess.run(
                [
                    "tesseract",
                    str(image_path),
                    "stdout",
                    "--psm",
                    "6",
                    "-l",
                    language,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return normalize_whitespace(result.stdout)
        except subprocess.CalledProcessError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return ""


def extract_pdf_pages_ocr(pdf_path: Path) -> list[str]:
    with TemporaryDirectory(prefix="energy_ocr_") as tmpdir:
        prefix = Path(tmpdir) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "200", str(pdf_path), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
        image_paths = sorted(Path(tmpdir).glob("page-*.png"))
        page_texts: list[str] = []
        for image_path in image_paths:
            page_text = run_tesseract(image_path)
            if page_text:
                page_texts.append(page_text)
        return page_texts


def split_page_text(page_text: str) -> list[str]:
    lines = [line.rstrip() for line in page_text.replace(
        "\r\n", "\n").replace("\r", "\n").splitlines()]
    return [line for line in lines if line.strip()]


def normalize_boilerplate_signature(line: str) -> str:
    line = normalize_whitespace(line).casefold()
    line = re.sub(r"page\s*\d+(?:\s*/\s*\d+)?", " page ", line)
    line = re.sub(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", " <date> ", line)
    line = re.sub(r"\d+(?:[.,:/-]\d+)*", " <num> ", line)
    line = re.sub(r"[|·•*_=-]+", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" .,:;|<>")


def is_static_noise_line(line: str) -> bool:
    text = normalize_whitespace(line)
    if not text:
        return True
    if re.fullmatch(r"[-–—_ ]+", text):
        return True
    if re.fullmatch(r"(page\s*)?\d+(\s*/\s*\d+)?", text.lower()):
        return True
    if re.fullmatch(r"\d+", text):
        return True
    patterns = (
        r"forward-looking statements?",
        r"safe harbor",
        r"all rights reserved",
        r"copyright",
        r"investor relations",
        r"annual report",
        r"interim report",
        r"website",
        r"www\.",
        r"http[s]?://",
        r"for more information",
        r"this document is for informational purposes",
        r"no representation or warranty",
        r"the company shall not be liable",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_table_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    multi_column_lines = sum(
        1 for line in lines if re.search(r"\S\s{2,}\S", line))
    if multi_column_lines < max(2, len(lines) // 2):
        return False
    if sum(1 for line in lines if re.search(r"[。！？.!?]", line)) > len(lines) // 2:
        return False
    return True


def page_edge_signature(lines: list[str]) -> str:
    normalized = [normalize_boilerplate_signature(line) for line in lines]
    normalized = [line for line in normalized if line]
    if not normalized:
        return ""
    return " | ".join(normalized[:4])


def clean_pdf_pages(raw_pages: list[str]) -> tuple[list[PdfBlock], dict[str, int]]:
    page_lines = [split_page_text(page_text) for page_text in raw_pages]
    candidate_counts: Counter[str] = Counter()
    for lines in page_lines:
        for edge in (lines[:2], lines[-2:]):
            signature = page_edge_signature(edge)
            if signature:
                candidate_counts[signature] += 1
        for line in lines[:3] + lines[-3:]:
            if is_static_noise_line(line):
                candidate_counts[normalize_boilerplate_signature(line)] += 1

    threshold = max(2, round(len(page_lines) * 0.4))
    repeated_noise = {signature for signature, count in candidate_counts.items(
    ) if signature and count >= threshold}

    cleaned_pages: list[PdfBlock] = []
    removed_lines = 0
    removed_pages = 0
    for page_number, lines in enumerate(page_lines, start=1):
        cleaned_lines: list[str] = []
        line_signatures = [
            normalize_boilerplate_signature(line) for line in lines]
        edge_signature = page_edge_signature(lines[:2] + lines[-2:])
        for index, line in enumerate(lines):
            signature = line_signatures[index]
            if not signature:
                removed_lines += 1
                continue
            is_edge_line = index < 3 or index >= max(0, len(lines) - 3)
            if is_static_noise_line(line):
                removed_lines += 1
                continue
            if is_edge_line and (signature in repeated_noise or (edge_signature and edge_signature in repeated_noise)):
                removed_lines += 1
                continue
            cleaned_lines.append(line)

        page_text = "\n".join(cleaned_lines).strip()
        if page_text:
            cleaned_pages.append(
                PdfBlock(page_number=page_number, text=page_text))
        else:
            removed_pages += 1

    stats = {
        "page_count": len(page_lines),
        "clean_page_count": len(cleaned_pages),
        "removed_noise_lines": removed_lines,
        "removed_noise_pages": removed_pages,
        "repeated_noise_signatures": len(repeated_noise),
    }
    return cleaned_pages, stats


def append_structured_tables(pdf_path: Path, blocks: list[PdfBlock]) -> int:
    """Append fitz-reconstructed tables to each page block.

    pdftotext -layout scrambles multi-column financial tables, detaching a
    label from its numbers (e.g. "Net income attributable to common
    stockholders" ends up next to the wrong figure). PyMuPDF's find_tables
    rebuilds the actual grid from line/char positions, so we append each table
    row as a single line — label glued to its values, never split across a
    chunk: "Research and development | 3,969 | 3,075 | 2,593".

    Prose still comes from pdftotext; this only adds structured table rows.
    Never raises — any failure leaves the block's text unchanged.
    """
    try:
        import fitz  # PyMuPDF, already a dependency
    except Exception:
        return 0
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return 0
    appended = 0
    try:
        for block in blocks:
            index = block.page_number - 1
            if index < 0 or index >= len(doc):
                continue
            try:
                tables = doc[index].find_tables()
            except Exception:
                continue
            serialized: list[str] = []
            for table in tables.tables:
                try:
                    rows = table.extract()
                except Exception:
                    continue
                lines = []
                for row in rows:
                    cells = [" ".join((cell or "").split()) for cell in row]
                    if any(cells):
                        lines.append(" | ".join(cells))
                if lines:
                    serialized.append("\n".join(lines))
            if serialized:
                block.text = block.text + "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(serialized)
                appended += len(serialized)
    finally:
        doc.close()
    return appended


def extract_pdf_document(pdf_path: Path) -> tuple[list[PdfBlock], str, bool, str, int, int, dict[str, int | float]]:
    pdftotext_pages = extract_pdf_pages_pdftotext(pdf_path)
    pdftotext_text = normalize_whitespace("\n\n".join(
        page for page in pdftotext_pages if page.strip()))
    compact = re.sub(r"\s+", "", pdftotext_text)
    text_extract_chars = len(compact)
    threshold_chars = max(
        600, int(0.4 * max(1, pdf_path.stat().st_size // 1024)))
    page_count = len(pdftotext_pages)
    text_density = text_extract_chars / max(1, page_count)
    allow_ocr = page_count <= 80 and text_extract_chars < max(
        1200, threshold_chars * 2)

    if text_extract_chars >= threshold_chars:
        cleaned_blocks, cleaning_stats = clean_pdf_pages(pdftotext_pages)
        table_count = append_structured_tables(pdf_path, cleaned_blocks)
        quality_score = min(1.0, text_density / 1200.0)
        return (
            cleaned_blocks,
            "pdftotext",
            False,
            "text_sufficient",
            text_extract_chars,
            0,
            {
                "page_count": page_count,
                "text_page_count": len(cleaned_blocks),
                "ocr_page_count": 0,
                "structured_tables": table_count,
                "quality_score": round(quality_score, 4),
                "needs_review": quality_score < 0.15,
                **cleaning_stats,
            },
        )

    ocr_pages = extract_pdf_pages_ocr(pdf_path)
    ocr_text = normalize_whitespace("\n\n".join(ocr_pages))
    if ocr_text:
        cleaned_blocks, cleaning_stats = clean_pdf_pages(ocr_pages)
        ocr_char_count = len(re.sub(r"\s+", "", ocr_text))
        quality_score = min(1.0, ocr_char_count / max(1, page_count * 900.0))
        return (
            cleaned_blocks,
            "ocr",
            True,
            "text_sparse",
            text_extract_chars,
            ocr_char_count,
            {
                "page_count": page_count,
                "text_page_count": 0,
                "ocr_page_count": len(cleaned_blocks),
                "quality_score": round(quality_score, 4),
                "needs_review": quality_score < 0.25,
                **cleaning_stats,
            },
        )

    cleaned_blocks, cleaning_stats = clean_pdf_pages(pdftotext_pages)
    table_count = append_structured_tables(pdf_path, cleaned_blocks)
    quality_score = min(1.0, text_density / 1200.0)
    return (
        cleaned_blocks,
        "pdftotext",
        False,
        "text_fallback",
        text_extract_chars,
        0,
        {
            "page_count": page_count,
            "text_page_count": len(cleaned_blocks),
            "ocr_page_count": 0,
            "structured_tables": table_count,
            "quality_score": round(quality_score, 4),
            "needs_review": True,
            **cleaning_stats,
        },
    )


def is_heading_paragraph(paragraph: str) -> bool:
    text = normalize_whitespace(paragraph)
    if not text:
        return False

    compact = re.sub(r"\s+", "", text)

    if len(text) > 120:
        return False

    cn_heading_terms = (
        "管理层讨论与分析",
        "管理层讨论",
        "董事会报告",
        "董事会工作报告",
        "财务报表",
        "财务报表附注",
        "财务状况",
        "经营情况",
        "经营业绩",
        "主要会计政策",
        "公司概况",
        "风险因素",
        "股东信息",
        "重大事项",
        "审计报告",
    )
    has_cn_heading = any(term in text for term in cn_heading_terms)
    if len(text) > 80 and not has_cn_heading:
        return False

    if re.search(r"[。！？.!?；;]", text):
        return False

    if re.match(r"^第[一二三四五六七八九十百0-9]+(章|节|部分|篇)", compact):
        return True

    if has_cn_heading:
        return True

    if compact.startswith("财务报告") and len(compact) <= 12:
        return True

    if text.endswith("报告") and len(text) <= 40:
        return True

    if re.match(r"^(第[一二三四五六七八九十0-9]+[章节部分篇条]?|[0-9]+(\.[0-9]+)*\s+)", text):
        return True

    if text.endswith(("：", ":")):
        return True

    if text.upper() == text and len(text) >= 3:
        return True

    if len(text) <= 24 and (" " not in text or len(text.split()) <= 4):
        return True

    return False


def clean_chunk_text(text: str, max_token_len: int = 120) -> str:
    """Drop retrieval-noise tokens and tame size-busting ones before chunking.

    Token-based (robust to escaped quotes in scraped HTML/JSON):
    - A whitespace-free token longer than max_token_len that looks like config
      (quotes, ":" pairs, or a URL) is embedded JSON / nav-blob noise scraped
      from HTML (e.g. Sungrow's index.html) — drop it; it is not prose and a
      single such token can be 1000+ chars.
    - Any other over-long run (e.g. a bare URL) is hard-wrapped so one token
      can't blow past the size bound and defeat word-count splitting.
    """
    kept: list[str] = []
    for token in text.split():
        if len(token) > max_token_len:
            looks_like_config = ('"' in token) or (":" in token) or ("http" in token)
            if looks_like_config:
                continue
            kept.extend(token[i:i + max_token_len] for i in range(0, len(token), max_token_len))
        else:
            kept.append(token)
    return " ".join(kept)


def split_paragraph_text(
    paragraph: str, max_words: int = 180, overlap: int = 35, max_chars: int = 2400
) -> list[str]:
    paragraph = clean_chunk_text(paragraph)
    words = paragraph.split()
    if not words:
        return []
    if len(words) <= max_words and len(paragraph) <= max_chars:
        return [paragraph]

    chunks: list[str] = []
    cursor = 0
    while cursor < len(words):
        window = words[cursor: cursor + max_words]
        while len(window) > 1 and len(" ".join(window)) > max_chars:
            window = window[:-1]
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if cursor + len(window) >= len(words):
            break
        # Keep overlap below the window so the cursor always makes real progress
        # (avoids hundreds of near-duplicate chunks when the window is small).
        step = max(1, len(window) - min(overlap, len(window) // 2))
        cursor += step

    return chunks


def parse_toc_line(line: str) -> tuple[str, int] | None:
    text = normalize_whitespace(line)
    if not text:
        return None
    if re.search(r"[。！？!?]", text):
        return None
    if len(text) > 160:
        return None
    if re.search(r"\.{2,}", text):
        text = re.sub(r"\.{2,}", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
    spaced = re.match(r"^(?P<title>.+?)\s+(?P<page>\d{1,4})$", text)
    if spaced:
        title = normalize_whitespace(spaced.group("title"))
        page = int(spaced.group("page"))
        if title and not title.isdigit():
            return title, page
    return None


def extract_toc_entries(blocks: list[PdfBlock]) -> list[tuple[str, int]]:
    if not blocks:
        return []
    max_scan_page = min(30, max(block.page_number for block in blocks))
    toc_pages: list[int] = []
    for block in blocks:
        if block.page_number > max_scan_page:
            break
        if "目录" in block.text:
            toc_pages.append(block.page_number)
    if not toc_pages:
        return []

    entries: list[tuple[str, int]] = []
    toc_page_set = set(toc_pages + [page + 1 for page in toc_pages])
    for block in blocks:
        if block.page_number not in toc_page_set:
            continue
        lines = split_page_text(block.text)
        index = 0
        while index < len(lines):
            line = lines[index]
            candidate = line
            if re.search(r"\.{2,}", line) and not re.search(r"\d{1,4}\s*$", line):
                if index + 1 < len(lines) and re.fullmatch(r"\d{1,4}", lines[index + 1].strip()):
                    candidate = f"{line} {lines[index + 1].strip()}"
                    index += 1
            parsed = parse_toc_line(candidate)
            if not parsed:
                index += 1
                continue
            title, page = parsed
            if page <= 0:
                index += 1
                continue
            entries.append((title, page))
            index += 1

    deduped: dict[int, str] = {}
    for title, page in entries:
        if page not in deduped:
            deduped[page] = title
    return sorted(((title, page) for page, title in deduped.items()), key=lambda item: item[1])


def split_blocks_into_sections(blocks: list[PdfBlock]) -> list[Section]:
    toc_entries = extract_toc_entries(blocks)
    use_toc = bool(toc_entries)
    max_page = max((block.page_number for block in blocks), default=1)
    toc_ranges: list[tuple[int, int, str]] = []
    if use_toc:
        for idx, (title, start_page) in enumerate(toc_entries):
            end_page = max_page
            if idx + 1 < len(toc_entries):
                end_page = max(start_page, toc_entries[idx + 1][1] - 1)
            toc_ranges.append((start_page, end_page, title))

    current_title = "Overview"
    sections: list[Section] = []
    current_paragraphs: list[ParagraphBlock] = []
    current_page_start = blocks[0].page_number if blocks else 1
    current_page_end = blocks[0].page_number if blocks else 1
    seen_paragraphs: set[str] = set()

    def toc_title_for_page(page_number: int) -> str | None:
        for start_page, end_page, title in toc_ranges:
            if start_page <= page_number <= end_page:
                return title
        return None

    def flush_section() -> None:
        if current_paragraphs:
            sections.append(
                Section(
                    title=current_title,
                    paragraphs=list(current_paragraphs),
                    page_start=current_page_start,
                    page_end=current_page_end,
                )
            )

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue

        paragraph_candidates = [part.strip() for part in re.split(
            r"\n\s*\n+", block_text) if part.strip()]
        if not paragraph_candidates:
            paragraph_candidates = [block_text]

        if use_toc:
            toc_title = toc_title_for_page(block.page_number)
            if toc_title and toc_title != current_title:
                flush_section()
                current_paragraphs.clear()
                current_title = toc_title
                current_page_start = block.page_number
                current_page_end = block.page_number
        else:
            heading_candidate = normalize_whitespace(paragraph_candidates[0])
            if is_heading_paragraph(heading_candidate):
                flush_section()
                current_paragraphs.clear()
                current_title = heading_candidate
                current_page_start = block.page_number
                current_page_end = block.page_number
                continue

        for paragraph in paragraph_candidates:
            paragraph_text = normalize_table_block(paragraph) if is_table_block(
                paragraph) else normalize_whitespace(paragraph)
            if not paragraph_text:
                continue
            fingerprint = sha256_text(
                canonicalize_paragraph_for_dedupe(paragraph_text))
            if fingerprint in seen_paragraphs:
                continue
            seen_paragraphs.add(fingerprint)
            current_paragraphs.append(
                ParagraphBlock(
                    text=paragraph_text,
                    page_start=block.page_number,
                    page_end=block.page_number,
                )
            )
            current_page_end = block.page_number
            if block.page_number < current_page_start:
                current_page_start = block.page_number

    flush_section()

    if not sections and blocks:
        merged_text = "\n\n".join(
            block.text for block in blocks if block.text.strip())
        if merged_text.strip():
            sections.append(
                Section(
                    title="Overview",
                    paragraphs=[
                        ParagraphBlock(
                            text=normalize_whitespace(merged_text),
                            page_start=blocks[0].page_number,
                            page_end=blocks[-1].page_number,
                        )
                    ],
                    page_start=blocks[0].page_number,
                    page_end=blocks[-1].page_number,
                )
            )

    return sections


def load_manifest() -> list[dict[str, object]]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"collection manifest not found: {MANIFEST_PATH}")

    rows: list[dict[str, object]] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_db_url() -> str:
    load_env_file(ROOT / ".env")
    if "DATABASE_URL" not in os.environ:
        load_env_file(ROOT / ".env.example")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is missing. Create .env from .env.example first.")
    return database_url


def main() -> None:
    database_url = ensure_db_url()
    manifest = load_manifest()

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for artifact in manifest:
                local_path = ROOT / str(artifact["local_path"])
                if not local_path.exists():
                    print(f"skip missing artifact: {local_path}")
                    continue

                artifact_type = str(artifact["artifact_type"])
                page_quality_stats: dict[str, int | float] = {}
                if artifact_type == "html":
                    raw_bytes = local_path.read_bytes()
                    text = extract_html_text(raw_bytes)
                    blocks = [PdfBlock(page_number=1, text=text)
                              ] if text else []
                    parse_method = "html_parser"
                    ocr_used = False
                    ocr_reason = "not_applicable"
                    extracted_char_count = len(re.sub(r"\s+", "", text))
                    ocr_char_count = 0
                    page_quality_stats = {"page_count": 1 if text else 0, "text_page_count": 1 if text else 0,
                                          "ocr_page_count": 0, "quality_score": 1.0 if text else 0.0, "needs_review": False}
                elif artifact_type == "pdf":
                    blocks, parse_method, ocr_used, ocr_reason, extracted_char_count, ocr_char_count, page_quality_stats = extract_pdf_document(
                        local_path)
                    text = "\n\n".join(
                        block.text for block in blocks if block.text.strip())
                else:
                    print(f"skip unsupported artifact type: {artifact_type}")
                    continue

                if not text or not blocks:
                    print(f"skip empty text: {local_path}")
                    continue

                content_hash = sha256_text(text)
                source_id = str(artifact["source_id"])
                title = str(artifact["title"])
                source_url = str(artifact["source_url"])
                sha256 = str(artifact["sha256"])
                downloaded_at = str(artifact["downloaded_at"])
                artifact_metadata = {
                    "source_id": source_id,
                    "artifact_type": artifact_type,
                    "local_path": str(local_path.relative_to(ROOT)),
                    "downloaded_at": downloaded_at,
                    "parse_method": parse_method,
                    "ocr_used": ocr_used,
                    "ocr_reason": ocr_reason,
                    "extracted_char_count": extracted_char_count,
                    "ocr_char_count": ocr_char_count,
                    "page_quality": page_quality_stats,
                    "company": artifact.get("company"),
                    "organization": artifact.get("organization"),
                    "region": artifact.get("region"),
                    "category": artifact.get("category"),
                    "tags": artifact.get("tags"),
                }

                cur.execute(
                    """
                    INSERT INTO documents (
                        source_type,
                        source_url,
                        title,
                        file_name,
                        file_hash,
                        doc_type,
                        language,
                        status,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (file_hash) DO UPDATE
                    SET updated_at = now(),
                        metadata = EXCLUDED.metadata
                    RETURNING id
                    """,
                    (
                        f"collected_{artifact_type}",
                        source_url,
                        title,
                        Path(local_path).name,
                        sha256,
                        "energy_source",
                        "en" if re.search(
                            r"[A-Za-z]", text) and not re.search(r"[\u4e00-\u9fff]", text) else "zh",
                        "ready",
                        json.dumps(artifact_metadata, ensure_ascii=False),
                    ),
                )
                document_id = cur.fetchone()[0]

                cur.execute(
                    """
                    SELECT id, version_no
                    FROM document_versions
                    WHERE content_hash = %s
                    """,
                    (content_hash,),
                )
                existing_version = cur.fetchone()

                if existing_version is not None:
                    document_version_id = existing_version[0]
                else:
                    cur.execute(
                        """
                        SELECT COALESCE(MAX(version_no), 0) + 1
                        FROM document_versions
                        WHERE document_id = %s
                        """,
                        (document_id,),
                    )
                    next_version_no = cur.fetchone()[0]

                    cur.execute(
                        """
                        INSERT INTO document_versions (
                            document_id,
                            version_no,
                            raw_path,
                            parsed_path,
                            content_hash,
                            published_at,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, now(), %s::jsonb)
                        ON CONFLICT (content_hash) DO UPDATE
                        SET ingested_at = now(),
                            metadata = EXCLUDED.metadata
                        RETURNING id
                        """,
                        (
                            document_id,
                            next_version_no,
                            str(local_path.relative_to(ROOT)),
                            None,
                            content_hash,
                            json.dumps(artifact_metadata, ensure_ascii=False),
                        ),
                    )
                    document_version_id = cur.fetchone()[0]

                cur.execute(
                    "DELETE FROM chunks WHERE document_version_id = %s",
                    (document_version_id,),
                )

                sections = split_blocks_into_sections(blocks)
                if not sections:
                    sections = [Section(title="Overview", paragraphs=[
                                        normalize_whitespace(text)], page_start=1, page_end=1)]

                chunk_index = 0
                for section_index, section in enumerate(sections):
                    paragraph_chunks: list[ParagraphBlock] = []
                    for paragraph in section.paragraphs:
                        split_texts = split_paragraph_text(paragraph.text)
                        if len(split_texts) == 1:
                            paragraph_chunks.append(paragraph)
                        else:
                            for split_text in split_texts:
                                paragraph_chunks.append(
                                    ParagraphBlock(
                                        text=split_text,
                                        page_start=paragraph.page_start,
                                        page_end=paragraph.page_end,
                                    )
                                )

                    chapter_text = clean_chunk_text(normalize_whitespace(
                        "\n\n".join(chunk.text for chunk in paragraph_chunks)
                        if paragraph_chunks
                        else "\n\n".join(paragraph.text for paragraph in section.paragraphs)
                    ))
                    if not chapter_text:
                        continue

                    chapter_metadata = json.dumps(
                        {
                            "source_id": source_id,
                            "artifact_type": artifact_type,
                            "chunk_level": "chapter",
                            "section_index": section_index,
                            "section_title": section.title,
                            "paragraph_count": len(paragraph_chunks) or len(section.paragraphs),
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                        },
                        ensure_ascii=False,
                    )
                    chapter_token_count = max(1, len(chapter_text.split()))
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            document_version_id,
                            chunk_index,
                            parent_chunk_id,
                            chunk_level,
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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        RETURNING id
                        """,
                        (
                            document_version_id,
                            chunk_index,
                            None,
                            "chapter",
                            section.title,
                            section.page_start,
                            section.page_end,
                            0,
                            len(chapter_text),
                            chapter_token_count,
                            artifact_type,
                            chapter_text,
                            chapter_metadata,
                        ),
                    )
                    chapter_chunk_id = cur.fetchone()[0]
                    chapter_embedding = vector_literal(
                        embed_text(chapter_text))
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
                            chapter_chunk_id,
                            "hash-bootstrap",
                            "bootstrap-v1",
                            1024,
                            chapter_embedding,
                        ),
                    )
                    chunk_index += 1

                    for paragraph_index, paragraph in enumerate(paragraph_chunks):
                        paragraph_metadata = json.dumps(
                            {
                                "source_id": source_id,
                                "artifact_type": artifact_type,
                                "chunk_level": "paragraph",
                                "section_index": section_index,
                                "section_title": section.title,
                                "paragraph_index": paragraph_index,
                                "parent_chunk_level": "chapter",
                                "page_start": paragraph.page_start,
                                "page_end": paragraph.page_end,
                            },
                            ensure_ascii=False,
                        )
                        paragraph_text = paragraph.text
                        paragraph_token_count = max(
                            1, len(paragraph_text.split()))
                        cur.execute(
                            """
                            INSERT INTO chunks (
                                document_version_id,
                                chunk_index,
                                parent_chunk_id,
                                chunk_level,
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
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            RETURNING id
                            """,
                            (
                                document_version_id,
                                chunk_index,
                                chapter_chunk_id,
                                "paragraph",
                                section.title,
                                paragraph.page_start,
                                paragraph.page_end,
                                0,
                                len(paragraph_text),
                                paragraph_token_count,
                                artifact_type,
                                paragraph_text,
                                paragraph_metadata,
                            ),
                        )
                        paragraph_chunk_id = cur.fetchone()[0]

                        embedding = vector_literal(embed_text(paragraph_text))
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
                                paragraph_chunk_id,
                                "hash-bootstrap",
                                "bootstrap-v1",
                                1024,
                                embedding,
                            ),
                        )
                        chunk_index += 1

                cur.execute(
                    """
                    INSERT INTO ingestion_jobs (
                        document_id,
                        job_type,
                        status,
                        started_at,
                        finished_at,
                        parse_method,
                        ocr_used,
                        ocr_reason,
                        extracted_char_count,
                        ocr_char_count,
                        page_count,
                        text_page_count,
                        ocr_page_count,
                        quality_score,
                        needs_review,
                        metadata
                    )
                    VALUES (%s, %s, %s, now(), now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        document_id,
                        "energy_ingestion",
                        "succeeded",
                        parse_method,
                        ocr_used,
                        ocr_reason,
                        extracted_char_count,
                        ocr_char_count,
                        int(page_quality_stats.get("page_count", 0)),
                        int(page_quality_stats.get("text_page_count", 0)),
                        int(page_quality_stats.get("ocr_page_count", 0)),
                        float(page_quality_stats.get("quality_score", 0.0)),
                        bool(page_quality_stats.get("needs_review", False)),
                        json.dumps(
                            {
                                "source_id": source_id,
                                "artifact_type": artifact_type,
                                "local_path": str(local_path.relative_to(ROOT)),
                                "parse_method": parse_method,
                                "ocr_used": ocr_used,
                                "ocr_reason": ocr_reason,
                                "extracted_char_count": extracted_char_count,
                                "ocr_char_count": ocr_char_count,
                                **page_quality_stats,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

        conn.commit()

    print(f"ingested {len(manifest)} collected artifacts")


if __name__ == "__main__":
    main()
