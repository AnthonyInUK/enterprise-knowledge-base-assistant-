from __future__ import annotations
from psycopg.rows import dict_row
from psycopg.types.json import Json
import psycopg

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


load_dotenv()


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
SENTENCE_RE = re.compile(r"(?<=[。！？!?\.])\s+")

EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

QUERY_NOISE_TOKENS = {
    "哪些",
    "什么",
    "怎么",
    "关于",
    "资料",
    "提到",
    "信息",
    "这份",
}


def _load_query_aliases() -> dict[str, dict[str, list[str]]]:
    """Load query-time alias maps from data/config/query_aliases.json.

    Path resolves relative to the repo root (this file's grandparent), which
    matches the layout shipped in the Docker image. Override with the env var
    RAG_ALIASES_PATH if needed.
    """
    path = os.getenv("RAG_ALIASES_PATH") or str(
        Path(__file__).resolve().parents[1] / "data" / "config" / "query_aliases.json"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to load query aliases from {path!r}: {exc}") from exc
    return {
        "cn_alias_map": {k: list(v) for k, v in data.get("cn_alias_map", {}).items()},
        "document_hints": {k: list(v) for k, v in data.get("document_hints", {}).items()},
    }


_QUERY_ALIASES = _load_query_aliases()
CN_ALIAS_MAP: dict[str, list[str]] = _QUERY_ALIASES["cn_alias_map"]

# Single source of truth for per-category retrieval config. Each category
# co-locates all of its term lists so adding/editing a category is a single
# edit here. The four module-level dicts below are read-only views derived
# from this — kept for backward compatibility with existing imports.
#   keywords — question-classification terms (bilingual)
#   rewrite  — query-expansion phrases injected into retrieval
#   rerank   — terms that earn a rerank bonus when present in a chunk
#   sections — section titles preferred for this category
CATEGORY_CONFIG: dict[str, dict[str, list[str]]] = {
    "financial": {
        "keywords": ["revenue", "income", "profit", "earnings", "margin", "financial", "收入", "营收", "利润", "财务", "经营结果"],
        "rewrite": [
            "revenue",
            "net sales",
            "gross profit",
            "operating income",
            "net income",
            "financial review",
            "results of operations",
            "management discussion",
        ],
        "rerank": ["revenue", "net sales", "gross profit", "operating income", "net income", "results of operations", "financial review"],
        "sections": [
            "financial review",
            "results of operations",
            "management discussion",
            "management discussion and analysis",
            "md&a",
            "mda",
            "financial statements",
            "财务报告",
            "管理层讨论与分析",
            "公司简介和主要财务指标",
            "主要财务指标",
            "财务报表",
            "财务报表附注",
            "经营业绩",
            "经营情况",
            "现金流量",
            "overview",
        ],
    },
    "capacity": {
        "keywords": ["capacity", "manufacturing", "production", "installed", "deployment", "gw", "mwh", "产能", "装机", "制造", "部署"],
        "rewrite": [
            "capacity",
            "manufacturing capacity",
            "production",
            "deployment",
            "installed capacity",
            "operations",
        ],
        "rerank": ["capacity", "manufacturing", "production", "installed", "deployment", "gw", "mwh"],
        "sections": ["capacity", "manufacturing", "production", "operations"],
    },
    "market": {
        "keywords": ["market", "region", "overseas", "international", "global", "country", "市场", "海外", "地区"],
        "rewrite": [
            "market",
            "region",
            "geographic",
            "international",
            "overseas",
            "segment",
        ],
        "rerank": ["market", "region", "geographic", "international", "overseas", "country", "segment"],
        "sections": ["market", "region", "geographic", "segment"],
    },
    "technology": {
        "keywords": ["technology", "product", "solution", "platform", "roadmap", "r&d", "innov", "技术", "产品", "解决方案", "技术路线"],
        "rewrite": [
            "technology",
            "product",
            "solution",
            "platform",
            "research and development",
            "roadmap",
        ],
        "rerank": ["technology", "product", "solution", "platform", "research", "development", "roadmap"],
        "sections": ["technology", "product", "solution", "research", "development"],
    },
    "project": {
        "keywords": ["project", "order", "contract", "agreement", "partner", "cooperation", "delivery", "项目", "订单", "合作"],
        "rewrite": [
            "project",
            "order",
            "contract",
            "agreement",
            "delivery",
            "pipeline",
            "partner",
        ],
        "rerank": ["project", "order", "contract", "agreement", "delivery", "pipeline", "partner"],
        "sections": ["project", "order", "contract", "delivery", "pipeline"],
    },
}

# Backward-compatible read-only views derived from CATEGORY_CONFIG.
QUESTION_CATEGORY_KEYWORDS: dict[str, list[str]] = {c: cfg["keywords"] for c, cfg in CATEGORY_CONFIG.items()}
QUERY_REWRITE_TERMS: dict[str, list[str]] = {c: cfg["rewrite"] for c, cfg in CATEGORY_CONFIG.items()}
RERANK_CATEGORY_TERMS: dict[str, list[str]] = {c: cfg["rerank"] for c, cfg in CATEGORY_CONFIG.items()}
PREFERRED_SECTION_TERMS: dict[str, list[str]] = {c: cfg["sections"] for c, cfg in CATEGORY_CONFIG.items()}

NOISY_SECTION_TERMS = [
    "notes to consolidated financial statements",
    "safe harbor",
    "forward-looking",
    "table of contents",
]

FINANCIAL_METRIC_TERMS = [
    "主要财务指标",
    "营业收入",
    "净利润",
    "归属于上市公司股东的净利润",
    "扣除非经常性损益",
    "经营活动产生的现金流量净额",
    "现金流量净额",
    "资产总额",
    "净资产",
    "每股收益",
    "加权平均净资产收益率",
    "revenue",
    "net profit",
    "operating income",
    "cash flow",
    "net income",
]

FINANCIAL_SECTION_TITLES = [
    "财务报告",
    "主要财务指标",
    "公司简介和主要财务指标",
    "财务报表",
    "财务报表附注",
]

FINANCIAL_METRIC_LABELS = [
    "营业收入",
    "归属于上市公司股东的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "经营活动产生的现金流量净额",
    "资产总额",
    "净资产",
    "每股收益",
    "加权平均净资产收益率",
]

# Facts-store lookup: map question phrasing to a canonical metric key. Ordered
# most-specific first (e.g. "ebit margin" before "ebit"). Prototype scope.
FACT_METRIC_HINTS: list[tuple[str, str]] = [
    ("ebit margin", "ebit_margin"),
    ("ebit", "ebit"),
    ("operating profit", "ebit"),
    ("research and development", "research_and_development"),
    ("r&d", "research_and_development"),
    ("研发", "research_and_development"),
    ("net income attributable", "net_income_attributable_to_common_stockholders"),
    ("归属于上市公司股东的净利润", "net_income_attributable_to_common_stockholders"),
    ("total revenue", "total_revenues"),
    ("营业收入", "total_revenues"),
]

# Version tag for the answer post-processing / direct-fact pipeline.
# Bump this whenever correction logic changes so cached answers invalidate.
ANSWER_PIPELINE_VERSION = "6"

# Company-agnostic metrics for the direct table-extraction path.
# Each entry maps generic question intent (bilingual triggers) to the row
# labels that may carry the value in a table, plus a display label per
# language. Add metrics here instead of hand-writing per-question if-branches.
DIRECT_FACT_METRICS: list[dict[str, Any]] = [
    {
        "triggers": ["total revenue", "total revenues", "营业收入", "总营收", "总收入"],
        "row_patterns": [r"Total revenues", r"Revenue", r"营业收入", r"营业总收入"],
        "label_en": "total revenues",
        "label_zh": "营业收入",
    },
    {
        "triggers": ["net income", "净利润"],
        "row_patterns": [
            r"Net income attributable to common stockholders",
            r"Net income",
            r"归属于上市公司股东的净利润",
            r"净利润",
        ],
        "label_en": "net income",
        "label_zh": "净利润",
    },
    {
        "triggers": ["research and development", "r&d", "研发", "研究与开发"],
        "row_patterns": [r"Research and development", r"研发费用", r"研发投入"],
        "label_en": "research and development",
        "label_zh": "研发投入",
    },
]

DOCUMENT_HINTS: dict[str, list[str]] = _QUERY_ALIASES["document_hints"]


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_title: str
    chunk_level: str
    section_title: str | None
    text: str
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any]
    parent_chunk_id: str | None = None

    @property
    def citation(self) -> str:
        pages = ""
        if self.page_start and self.page_end and self.page_start == self.page_end:
            pages = f"p.{self.page_start}"
        elif self.page_start and self.page_end:
            pages = f"p.{self.page_start}-{self.page_end}"
        elif self.page_start:
            pages = f"p.{self.page_start}"
        title = self.document_title.strip()
        return f"{title}{(' ' + pages) if pages else ''}"


@dataclass(slots=True)
class RetrievalHit:
    rank: int
    score: float
    chunk: ChunkRecord
    matched_terms: list[str]
    reason: str


@dataclass(slots=True)
class ComposedAnswer:
    answer: str
    backend: str
    used_llm: bool
    grounded: bool
    ungrounded_numbers: list[str]


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    prompt: str
    sources: list[str]
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: float
    used_llm: bool
    debug: dict[str, Any]


class Database:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv(
            "DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag_assistant")

    def connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def fetch_all(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params or ())
            return list(cur.fetchall())

    def fetch_one(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params or ())
            conn.commit()


class RetrievalService:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()
        self._chunks: list[ChunkRecord] | None = None
        self._token_cache: dict[str, Counter[str]] = {}
        self._idf_cache: dict[str, float] | None = None
        self._avg_chunk_len: float | None = None
        self._cross_encoder: Any | None = None
        self._cross_encoder_error: str | None = None
        self._embedding_model: Any | None = None
        self._embedding_error: str | None = None
        self._last_rerank_cache_stats: dict[str, Any] = {
            "enabled": False, "hits": 0, "misses": 0}
        self._last_vector_stats: dict[str, Any] = {
            "enabled": False, "count": 0}
        self._last_query_embedding_cache_stats: dict[str, Any] = {
            "enabled": False, "hit": False}
        self._last_retrieval_cache_stats: dict[str, Any] = {
            "enabled": False, "hit": False}
        self._last_answer_cache_stats: dict[str, Any] = {
            "enabled": False, "hit": False}
        self._corpus_fingerprint: str | None = None

    @staticmethod
    def _normalize_query(question: str) -> str:
        return " ".join(question.strip().lower().split())

    @classmethod
    def _query_hash(cls, question: str) -> str:
        return hashlib.sha256(cls._normalize_query(question).encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_json(payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_enabled(self, env_name: str, default: str = "1") -> bool:
        return os.getenv(env_name, default) == "1"

    def _ttl_clause(self, env_name: str) -> str:
        days = int(os.getenv(env_name, "7"))
        if days <= 0:
            return "NULL"
        return f"now() + interval '{days} days'"

    def _corpus_version_fingerprint(self) -> str:
        if self._corpus_fingerprint is not None:
            return self._corpus_fingerprint
        try:
            row = self.db.fetch_one(
                """
                SELECT
                    COUNT(*) AS chunk_count,
                    COALESCE(MAX(c.created_at), 'epoch'::timestamptz) AS latest_chunk_at,
                    COALESCE(MAX(dv.ingested_at), 'epoch'::timestamptz) AS latest_version_at,
                    COALESCE(MAX(e.created_at), 'epoch'::timestamptz) AS latest_embedding_at
                FROM chunks c
                JOIN document_versions dv ON dv.id = c.document_version_id
                LEFT JOIN embeddings e ON e.chunk_id = c.id
                """
            )
        except Exception:
            return "unknown-corpus"
        self._corpus_fingerprint = self._hash_json(row or {})
        return self._corpus_fingerprint

    def _retrieval_strategy_fingerprint(self, top_k: int) -> str:
        keys = [
            "RAG_ENABLE_VECTOR",
            "RAG_VECTOR_CANDIDATES",
            "RAG_ENABLE_RERANK",
            "RAG_RERANK_BACKEND",
            "RAG_RERANK_MODEL",
            "RAG_RERANK_CANDIDATES",
            "RAG_RERANK_RECALL_WEIGHT",
            "RAG_EMBED_MODEL",
            "RAG_CONTEXT_EXPANSION_MIN_K",
            "RAG_DOC_FALLBACK_MIN_K",
        ]
        return self._hash_json({"top_k": top_k, **{key: os.getenv(key, "") for key in keys}})

    def _answer_strategy_fingerprint(self, top_k: int) -> str:
        keys = [
            "RAG_LLM_BACKEND",
            "DEEPSEEK_MODEL",
            "CLAUDE_MODEL",
            "RAG_TEMPERATURE",
        ]
        return self._hash_json(
            {
                "retrieval": self._retrieval_strategy_fingerprint(top_k),
                # Bump when answer post-processing / direct-fact logic changes,
                # so previously cached answers don't outlive the change.
                "answer_pipeline_version": ANSWER_PIPELINE_VERSION,
                **{key: os.getenv(key, "") for key in keys},
            }
        )

    def load_chunks(self) -> list[ChunkRecord]:
        if self._chunks is not None:
            return self._chunks

        sql = """
            SELECT
                c.id::text AS id,
                COALESCE(d.title, '') AS document_title,
                COALESCE(c.chunk_level, 'paragraph') AS chunk_level,
                c.section_title,
                c.text,
                c.page_start,
                c.page_end,
                COALESCE(c.metadata, '{}'::jsonb) AS metadata,
                c.parent_chunk_id::text AS parent_chunk_id
            FROM chunks c
            JOIN document_versions dv ON dv.id = c.document_version_id
            JOIN documents d ON d.id = dv.document_id
            WHERE COALESCE(c.text, '') <> ''
            ORDER BY d.title, c.chunk_level, c.page_start NULLS LAST, c.id
        """
        rows = self.db.fetch_all(sql)
        self._chunks = [
            ChunkRecord(
                id=row["id"],
                document_title=row["document_title"],
                chunk_level=row["chunk_level"],
                section_title=row.get("section_title"),
                text=row["text"],
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                metadata=dict(row.get("metadata") or {}),
                parent_chunk_id=row.get("parent_chunk_id"),
            )
            for row in rows
        ]
        return self._chunks

    def _tokenize(self, text: str) -> Counter[str]:
        cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cached = self._token_cache.get(cache_key)
        if cached is not None:
            return cached
        tokens: list[str] = []
        for token in TOKEN_RE.findall(text):
            if token.isascii():
                tokens.append(token.lower())
                continue
            if len(token) <= 2:
                tokens.append(token)
                continue
            tokens.append(token)
            tokens.extend(token[i: i + 2] for i in range(len(token) - 1))
        counts = Counter(tokens)
        self._token_cache[cache_key] = counts
        return counts

    def _expand_query(self, question: str) -> list[str]:
        expanded = [question]
        for cn_term, aliases in CN_ALIAS_MAP.items():
            if cn_term in question:
                expanded.extend(aliases)
        expanded.extend(QUERY_REWRITE_TERMS.get(
            self._question_category(question), []))
        return expanded

    def _question_category(self, question: str) -> str:
        lowered = question.lower()
        scores = {
            category: sum(
                1 for keyword in keywords if keyword.lower() in lowered)
            for category, keywords in QUESTION_CATEGORY_KEYWORDS.items()
        }
        return max(scores, key=scores.get) if scores else "summary"

    def _target_document_terms(self, question: str) -> list[str]:
        lowered = question.lower()
        terms: list[str] = []
        for mention, doc_terms in DOCUMENT_HINTS.items():
            if mention.lower() in lowered or mention in question:
                terms.extend(doc_terms)
        return sorted(set(terms))

    def _matches_target_document(self, chunk: ChunkRecord, target_document_terms: Sequence[str]) -> bool:
        if not target_document_terms:
            return True
        lowered_title = chunk.document_title.lower()
        return any(term in lowered_title for term in target_document_terms)

    def _category_term_hits(self, chunk: ChunkRecord, question_category: str) -> int:
        searchable = " ".join(
            filter(None, [chunk.document_title,
                   chunk.section_title or "", chunk.text])
        ).lower()
        return sum(1 for term in RERANK_CATEGORY_TERMS.get(question_category, []) if term in searchable)

    def _build_query_tokens(self, question: str) -> Counter[str]:
        expanded = " ".join(self._expand_query(question))
        tokens = self._tokenize(expanded)
        return self._filter_query_tokens(tokens)

    def _build_financial_focus_tokens(self, question: str) -> Counter[str]:
        tokens = self._build_query_tokens(question)
        for term in FINANCIAL_METRIC_TERMS:
            tokens.update(self._tokenize(term))
        return self._filter_query_tokens(tokens)

    def _filter_query_tokens(self, tokens: Counter[str]) -> Counter[str]:
        return Counter(
            {
                token: count
                for token, count in tokens.items()
                if token not in EN_STOPWORDS and token not in QUERY_NOISE_TOKENS and not token.isdigit()
            }
        )

    def _term_idf(self) -> dict[str, float]:
        if self._idf_cache is not None:
            return self._idf_cache
        chunks = self.load_chunks()
        doc_freq: Counter[str] = Counter()
        total_len = 0
        for chunk in chunks:
            tokens = self._chunk_tokens(chunk)
            doc_freq.update(tokens.keys())
            total_len += sum(tokens.values())
        total_docs = max(1, len(chunks))
        self._avg_chunk_len = total_len / total_docs if total_docs else 1.0
        self._idf_cache = {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        return self._idf_cache

    def _chunk_tokens(self, chunk: ChunkRecord) -> Counter[str]:
        base = " ".join(
            filter(
                None,
                [
                    chunk.document_title,
                    chunk.section_title or "",
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False,
                               sort_keys=True),
                ],
            )
        )
        return self._tokenize(base)

    def _section_bonus(self, lowered_section: str, question_category: str) -> float:
        bonus = 0.0
        if any(term in lowered_section for term in PREFERRED_SECTION_TERMS.get(question_category, [])):
            bonus += 0.35
        if any(term in lowered_section for term in NOISY_SECTION_TERMS):
            bonus -= 0.05
        if question_category == "financial" and "overview" in lowered_section:
            bonus -= 0.25
        return bonus

    def _financial_metric_bonus(self, chunk: ChunkRecord) -> float:
        searchable = " ".join(
            filter(None, [chunk.document_title,
                   chunk.section_title or "", chunk.text])
        )
        hits = sum(1 for term in FINANCIAL_METRIC_TERMS if term in searchable)
        if hits <= 0:
            return 0.0
        return min(0.24, 0.03 * hits)

    def _score_chunk(
        self,
        question_tokens: Counter[str],
        chunk: ChunkRecord,
        question_category: str,
        target_document_terms: Sequence[str] = (),
    ) -> tuple[float, list[str], str]:
        chunk_tokens = self._chunk_tokens(chunk)
        if not chunk_tokens:
            return 0.0, [], "empty"

        idf = self._term_idf()
        matched_terms = sorted(set(question_tokens) & set(
            chunk_tokens), key=lambda token: idf.get(token, 0.0), reverse=True)
        avg_len = self._avg_chunk_len or 1.0
        chunk_len = max(1, sum(chunk_tokens.values()))
        k1 = 1.2
        b = 0.75
        weighted_overlap = 0.0
        weighted_query = 0.0
        for token, query_count in question_tokens.items():
            token_idf = idf.get(token, 0.0)
            weighted_query += token_idf * query_count
            term_freq = chunk_tokens.get(token, 0)
            if not term_freq:
                continue
            bm25_tf = (term_freq * (k1 + 1)) / (term_freq +
                                                k1 * (1 - b + b * chunk_len / avg_len))
            weighted_overlap += token_idf * bm25_tf
        coverage = weighted_overlap / max(0.01, weighted_query)
        title_hit = 0.0
        lowered_title = chunk.document_title.lower()
        lowered_section = (chunk.section_title or "").lower()
        if any(term in lowered_title for term in matched_terms):
            title_hit += 0.8
        if any(term in lowered_section for term in matched_terms):
            title_hit += 0.4

        section_bonus = self._section_bonus(lowered_section, question_category)
        level_bonus = 0.15 if chunk.chunk_level == "paragraph" else 0.05
        page_bonus = 0.0
        if chunk.page_start is not None:
            page_bonus = min(0.2, math.log1p(chunk.page_start) / 20.0)
            if question_category == "financial" and chunk.page_start <= 2:
                page_bonus -= 0.1

        score = coverage * 3.0 + title_hit + section_bonus + level_bonus + page_bonus
        if target_document_terms:
            if self._matches_target_document(chunk, target_document_terms):
                score += 0.3
            else:
                score -= 0.7
        if len(chunk.text) > 1200:
            score -= 0.05
        if len(chunk.text) < 80:
            score -= 0.1
        if question_category == "financial":
            score += self._financial_metric_bonus(chunk)

        reason_parts = []
        if matched_terms:
            reason_parts.append(f"matched={','.join(matched_terms[:6])}")
        if title_hit:
            reason_parts.append("title_boost")
        if target_document_terms and self._matches_target_document(chunk, target_document_terms):
            reason_parts.append("target_doc")
        if section_bonus:
            reason_parts.append(f"section_boost={section_bonus:.2f}")
        if question_category == "financial":
            metric_bonus = self._financial_metric_bonus(chunk)
            if metric_bonus:
                reason_parts.append(f"metric_boost={metric_bonus:.2f}")
        if chunk.chunk_level:
            reason_parts.append(f"level={chunk.chunk_level}")
        return score, matched_terms[:10], ";".join(reason_parts) or "lexical"

    def _focus_financial_section(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return hits
        focused = [
            hit for hit in hits
            if any(term in (hit.chunk.section_title or "") for term in FINANCIAL_SECTION_TITLES)
        ]
        if len(focused) < max(3, top_k // 2):
            return hits
        focused.sort(key=lambda hit: hit.score +
                     self._financial_metric_bonus(hit.chunk), reverse=True)
        remainder = [hit for hit in hits if hit not in focused]
        return focused + remainder

    def _financial_section_hits(
        self,
        question: str,
        chunks: Sequence[ChunkRecord],
        question_category: str,
        target_document_terms: Sequence[str],
        top_k: int,
    ) -> list[RetrievalHit]:
        if question_category != "financial":
            return []
        focus_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_level != "chapter"
            and any(term in (chunk.section_title or "") for term in FINANCIAL_SECTION_TITLES)
        ]
        if not focus_chunks:
            return []
        focus_tokens = self._build_financial_focus_tokens(question)
        focus_hits: list[RetrievalHit] = []
        for chunk in focus_chunks:
            score, matched_terms, reason = self._score_chunk(
                focus_tokens, chunk, question_category, target_document_terms
            )
            if score <= 0:
                continue
            focus_hits.append(
                RetrievalHit(
                    rank=0,
                    score=score,
                    chunk=chunk,
                    matched_terms=matched_terms,
                    reason=f"financial_focus;{reason}",
                )
            )
        focus_hits.sort(key=lambda hit: hit.score, reverse=True)
        focus_limit = max(top_k, int(os.getenv("RAG_FINANCIAL_FOCUS_K", "80")))
        return focus_hits[:focus_limit]

    def _add_document_category_fallbacks(
        self,
        paragraph_hits: list[RetrievalHit],
        chunks: Sequence[ChunkRecord],
        top_k: int,
        question_category: str,
        target_document_terms: Sequence[str],
    ) -> list[RetrievalHit]:
        if not target_document_terms or top_k < int(os.getenv("RAG_DOC_FALLBACK_MIN_K", "20")):
            return paragraph_hits

        by_id = {hit.chunk.id: hit for hit in paragraph_hits}
        limit = int(os.getenv("RAG_DOC_FALLBACK_LIMIT", "40"))
        fallback_candidates: list[tuple[int, ChunkRecord]] = []
        for chunk in chunks:
            if chunk.chunk_level == "chapter" or chunk.id in by_id:
                continue
            if not self._matches_target_document(chunk, target_document_terms):
                continue
            category_hits = self._category_term_hits(chunk, question_category)
            if category_hits <= 0:
                continue
            fallback_candidates.append((category_hits, chunk))

        fallback_candidates.sort(key=lambda item: (
            item[0], item[1].page_start or 0), reverse=True)
        base_score = paragraph_hits[min(
            len(paragraph_hits) - 1, max(0, top_k - 1))].score if paragraph_hits else 0.3
        expanded = list(paragraph_hits)
        for index, (category_hits, chunk) in enumerate(fallback_candidates[:limit]):
            hit = RetrievalHit(
                rank=0,
                score=base_score - 0.2 - index * 0.002 +
                min(0.2, 0.04 * category_hits),
                chunk=chunk,
                matched_terms=[],
                reason=f"doc_category_fallback;category_hits={category_hits};target_doc",
            )
            by_id[chunk.id] = hit
            expanded.append(hit)
        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded

    def _expand_recall_context(
        self,
        paragraph_hits: list[RetrievalHit],
        chunks: Sequence[ChunkRecord],
        top_k: int,
        question_category: str,
        target_document_terms: Sequence[str],
    ) -> list[RetrievalHit]:
        if top_k < int(os.getenv("RAG_CONTEXT_EXPANSION_MIN_K", "20")) or not paragraph_hits:
            return paragraph_hits

        by_id = {hit.chunk.id: hit for hit in paragraph_hits}
        seed_count = min(
            int(os.getenv("RAG_CONTEXT_EXPANSION_SEEDS", "12")), len(paragraph_hits))
        same_page_limit = int(
            os.getenv("RAG_CONTEXT_EXPANSION_PAGE_LIMIT", "4"))
        sibling_limit = int(
            os.getenv("RAG_CONTEXT_EXPANSION_SIBLING_LIMIT", "4"))

        chunks_by_parent: dict[str, list[ChunkRecord]] = defaultdict(list)
        chunks_by_doc_page: dict[tuple[str, int | None,
                                       int | None], list[ChunkRecord]] = defaultdict(list)
        for chunk in chunks:
            if chunk.chunk_level == "chapter":
                continue
            if target_document_terms and not self._matches_target_document(chunk, target_document_terms):
                continue
            if chunk.parent_chunk_id:
                chunks_by_parent[chunk.parent_chunk_id].append(chunk)
            chunks_by_doc_page[(chunk.document_title,
                                chunk.page_start, chunk.page_end)].append(chunk)

        expanded = list(paragraph_hits)
        for seed_index, seed in enumerate(paragraph_hits[:seed_count]):
            additions: list[tuple[str, ChunkRecord]] = []
            if seed.chunk.parent_chunk_id:
                siblings = chunks_by_parent.get(seed.chunk.parent_chunk_id, [])
                siblings = sorted(
                    siblings,
                    key=lambda chunk: self._category_term_hits(
                        chunk, question_category),
                    reverse=True,
                )[:sibling_limit]
                additions.extend(("parent_sibling", chunk)
                                 for chunk in siblings)

            same_page = chunks_by_doc_page.get(
                (seed.chunk.document_title, seed.chunk.page_start, seed.chunk.page_end), [])
            same_page = sorted(
                same_page,
                key=lambda chunk: self._category_term_hits(
                    chunk, question_category),
                reverse=True,
            )[:same_page_limit]
            additions.extend(("same_page", chunk) for chunk in same_page)

            for reason, chunk in additions:
                if chunk.id in by_id:
                    continue
                bonus = min(
                    0.18, 0.04 * self._category_term_hits(chunk, question_category))
                score = seed.score - 0.12 - seed_index * 0.01 + bonus
                hit = RetrievalHit(
                    rank=0,
                    score=score,
                    chunk=chunk,
                    matched_terms=[],
                    reason=f"context_expansion={reason};seed={seed.chunk.id}",
                )
                by_id[chunk.id] = hit
                expanded.append(hit)
        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded

    def _fetch_query_embedding_cache(self, question: str, model_name: str) -> list[float] | None:
        self._last_query_embedding_cache_stats = {
            "enabled": self._cache_enabled("RAG_QUERY_EMBEDDING_CACHE"),
            "hit": False,
            "model": model_name,
        }
        if not self._cache_enabled("RAG_QUERY_EMBEDDING_CACHE"):
            return None
        query_hash = self._query_hash(question)
        try:
            row = self.db.fetch_one(
                """
                SELECT embedding::text AS embedding
                FROM query_embedding_cache
                WHERE query_hash = %s
                  AND embedding_model = %s
                  AND embedding_version = %s
                """,
                (query_hash, model_name, os.getenv("RAG_EMBED_VERSION", "v1")),
            )
        except Exception as exc:
            self._last_query_embedding_cache_stats.update({"error": str(exc)})
            return None
        if not row:
            return None
        try:
            vector_text = str(row["embedding"]).strip().strip("[]")
            values = [float(item) for item in vector_text.split(",") if item]
        except Exception as exc:
            self._last_query_embedding_cache_stats.update({"error": f"parse failed: {exc}"})
            return None
        if len(values) != 1024:
            self._last_query_embedding_cache_stats.update({"error": f"dimension {len(values)}"})
            return None
        self._last_query_embedding_cache_stats.update({"hit": True})
        try:
            self.db.execute(
                """
                UPDATE query_embedding_cache
                SET hit_count = hit_count + 1, updated_at = now()
                WHERE query_hash = %s
                  AND embedding_model = %s
                  AND embedding_version = %s
                """,
                (query_hash, model_name, os.getenv("RAG_EMBED_VERSION", "v1")),
            )
        except Exception:
            pass
        return values

    def _write_query_embedding_cache(self, question: str, model_name: str, values: Sequence[float]) -> None:
        if not self._cache_enabled("RAG_QUERY_EMBEDDING_CACHE") or len(values) != 1024:
            return
        try:
            self.db.execute(
                """
                INSERT INTO query_embedding_cache (
                    query_hash, query_text, embedding_model, embedding_version,
                    dimension, embedding, metadata, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s, now())
                ON CONFLICT (query_hash, embedding_model, embedding_version)
                DO UPDATE SET
                    query_text = EXCLUDED.query_text,
                    embedding = EXCLUDED.embedding,
                    dimension = EXCLUDED.dimension,
                    updated_at = now()
                """,
                (
                    self._query_hash(question),
                    question,
                    model_name,
                    os.getenv("RAG_EMBED_VERSION", "v1"),
                    len(values),
                    self._vector_literal(values),
                    Json({"source": "query_embedding_cache"}),
                ),
            )
        except Exception as exc:
            self._last_query_embedding_cache_stats.update({"write_error": str(exc)})

    def _retrieval_cache_key(self, question: str, top_k: int) -> tuple[str, str, str, str]:
        query_hash = self._query_hash(question)
        corpus_fingerprint = self._corpus_version_fingerprint()
        strategy_fingerprint = self._retrieval_strategy_fingerprint(top_k)
        cache_key = self._hash_json(
            {
                "query_hash": query_hash,
                "corpus": corpus_fingerprint,
                "strategy": strategy_fingerprint,
                "top_k": top_k,
            }
        )
        return cache_key, query_hash, corpus_fingerprint, strategy_fingerprint

    def _fetch_retrieval_result_cache(self, question: str, top_k: int, chunks: Sequence[ChunkRecord]) -> list[RetrievalHit] | None:
        self._last_retrieval_cache_stats = {
            "enabled": self._cache_enabled("RAG_RETRIEVAL_RESULT_CACHE"),
            "hit": False,
        }
        if not self._cache_enabled("RAG_RETRIEVAL_RESULT_CACHE"):
            return None
        cache_key, _, _, _ = self._retrieval_cache_key(question, top_k)
        try:
            row = self.db.fetch_one(
                """
                SELECT chunk_ids
                FROM retrieval_result_cache
                WHERE cache_key = %s
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                (cache_key,),
            )
        except Exception as exc:
            self._last_retrieval_cache_stats.update({"error": str(exc)})
            return None
        if not row:
            return None
        chunk_ids = [str(item) for item in (row.get("chunk_ids") or [])]
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        hits: list[RetrievalHit] = []
        for rank, chunk_id in enumerate(chunk_ids[:top_k], start=1):
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                return None
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=0.0,
                    chunk=chunk,
                    matched_terms=[],
                    reason="retrieval_result_cache",
                )
            )
        if not hits:
            return None
        self._last_retrieval_cache_stats.update({"hit": True, "count": len(hits)})
        try:
            self.db.execute(
                """
                UPDATE retrieval_result_cache
                SET hit_count = hit_count + 1, updated_at = now()
                WHERE cache_key = %s
                """,
                (cache_key,),
            )
        except Exception:
            pass
        return hits

    def _write_retrieval_result_cache(self, question: str, top_k: int, hits: Sequence[RetrievalHit]) -> None:
        if not self._cache_enabled("RAG_RETRIEVAL_RESULT_CACHE") or not hits:
            return
        cache_key, query_hash, corpus_fingerprint, strategy_fingerprint = self._retrieval_cache_key(question, top_k)
        expires_sql = self._ttl_clause("RAG_RETRIEVAL_CACHE_TTL_DAYS")
        try:
            self.db.execute(
                f"""
                INSERT INTO retrieval_result_cache (
                    cache_key, query_hash, query_text, corpus_fingerprint,
                    strategy_fingerprint, top_k, chunk_ids, metadata, expires_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::uuid[], %s, {expires_sql}, now())
                ON CONFLICT (cache_key)
                DO UPDATE SET
                    chunk_ids = EXCLUDED.chunk_ids,
                    metadata = EXCLUDED.metadata,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = now()
                """,
                (
                    cache_key,
                    query_hash,
                    question,
                    corpus_fingerprint,
                    strategy_fingerprint,
                    top_k,
                    [hit.chunk.id for hit in hits],
                    Json({"source": "retrieval_result_cache"}),
                ),
            )
        except Exception as exc:
            self._last_retrieval_cache_stats.update({"write_error": str(exc)})

    def _vector_literal(self, values: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"

    def _load_embedding_model(self) -> Any | None:
        if self._embedding_model is not None:
            return self._embedding_model
        if self._embedding_error is not None:
            return None
        model_name = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._embedding_error = f"sentence-transformers unavailable: {exc}"
            return None
        try:
            device = os.getenv("RAG_EMBED_DEVICE") or None
            if device is None:
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        device = "mps"
                except Exception:
                    device = None
            kwargs: dict[str, Any] = {}
            if device:
                kwargs["device"] = device
            self._embedding_model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            self._embedding_error = f"failed to load {model_name}: {exc}"
            return None
        return self._embedding_model

    def _embed_query(self, question: str) -> list[float] | None:
        model_name = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
        cached = self._fetch_query_embedding_cache(question, model_name)
        if cached is not None:
            return cached
        model = self._load_embedding_model()
        if model is None:
            return None
        try:
            vector = model.encode(
                [question], normalize_embeddings=True, show_progress_bar=False)[0]
        except Exception as exc:
            self._embedding_error = f"query embedding failed: {exc}"
            return None
        values = [float(item) for item in vector.tolist()]
        if len(values) != 1024:
            self._embedding_error = f"query embedding dimension {len(values)} does not match vector(1024)"
            return None
        self._write_query_embedding_cache(question, model_name, values)
        return values

    def _vector_search(self, question: str, top_k: int) -> list[tuple[str, float]]:
        self._last_vector_stats = {"enabled": os.getenv(
            "RAG_ENABLE_VECTOR", "0") == "1", "count": 0}
        if os.getenv("RAG_ENABLE_VECTOR", "0") != "1":
            return []
        query_vector = self._embed_query(question)
        if query_vector is None:
            self._last_vector_stats = {"enabled": True,
                                       "count": 0, "error": self._embedding_error}
            return []
        model_name = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
        vector_literal = self._vector_literal(query_vector)
        try:
            rows = self.db.fetch_all(
                """
                SELECT e.chunk_id::text AS chunk_id, (e.embedding <=> %s::vector) AS distance
                FROM embeddings e
                WHERE e.embedding_model = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_literal, model_name, vector_literal, top_k),
            )
        except Exception as exc:
            self._embedding_error = f"vector search failed: {exc}"
            self._last_vector_stats = {"enabled": True,
                                       "count": 0, "error": self._embedding_error}
            return []
        hits = [(row["chunk_id"], float(row["distance"])) for row in rows]
        self._last_vector_stats = {"enabled": True,
                                   "count": len(hits), "model": model_name}
        return hits

    def _fuse_lexical_and_vector_hits(
        self,
        question: str,
        paragraph_hits: list[RetrievalHit],
        chunks: Sequence[ChunkRecord],
        question_category: str,
        target_document_terms: Sequence[str],
    ) -> list[RetrievalHit]:
        if os.getenv("RAG_ENABLE_VECTOR", "0") != "1" or not paragraph_hits:
            return paragraph_hits
        vector_k = int(os.getenv("RAG_VECTOR_CANDIDATES", "80"))
        lexical_k = int(os.getenv("RAG_LEXICAL_FUSION_CANDIDATES", "200"))
        rrf_k = int(os.getenv("RAG_RRF_K", "60"))
        vector_hits = self._vector_search(question, vector_k)
        if not vector_hits:
            return paragraph_hits

        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        hit_by_id = {hit.chunk.id: hit for hit in paragraph_hits}
        lexical_rank = {hit.chunk.id: rank for rank,
                        hit in enumerate(paragraph_hits[:lexical_k], start=1)}
        vector_rank = {chunk_id: rank for rank,
                       (chunk_id, _distance) in enumerate(vector_hits, start=1)}
        candidate_ids = set(lexical_rank) | set(vector_rank)

        fused: list[RetrievalHit] = []
        for chunk_id in candidate_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or chunk.chunk_level == "chapter":
                continue
            hit = hit_by_id.get(chunk_id)
            if hit is None:
                score, matched_terms, reason = self._score_chunk(
                    self._build_query_tokens(
                        question), chunk, question_category, target_document_terms
                )
                hit = RetrievalHit(rank=0, score=max(
                    score, 0.0), chunk=chunk, matched_terms=matched_terms, reason=reason)
            fused_score = 0.0
            reasons = [hit.reason]
            if chunk_id in lexical_rank:
                fused_score += 1.0 / (rrf_k + lexical_rank[chunk_id])
                reasons.append(f"lex_rank={lexical_rank[chunk_id]}")
            if chunk_id in vector_rank:
                fused_score += 1.0 / (rrf_k + vector_rank[chunk_id])
                reasons.append(f"vec_rank={vector_rank[chunk_id]}")
            fused.append(
                RetrievalHit(
                    rank=0,
                    score=fused_score,
                    chunk=chunk,
                    matched_terms=hit.matched_terms,
                    reason=";".join(reasons),
                )
            )
        fused.sort(key=lambda item: item.score, reverse=True)
        existing = {hit.chunk.id for hit in fused}
        fused.extend(
            hit for hit in paragraph_hits if hit.chunk.id not in existing)
        return fused

    def _load_cross_encoder(self) -> Any | None:
        if self._cross_encoder is not None:
            return self._cross_encoder
        if self._cross_encoder_error is not None:
            return None

        model_name = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            self._cross_encoder_error = f"sentence-transformers unavailable: {exc}"
            return None

        try:
            device = os.getenv("RAG_RERANK_DEVICE") or None
            local_only = os.getenv("RAG_RERANK_LOCAL_FILES_ONLY", "0") == "1"
            kwargs: dict[str, Any] = {"max_length": int(
                os.getenv("RAG_RERANK_MAX_LENGTH", "512"))}
            if device:
                kwargs["device"] = device
            if local_only:
                kwargs["local_files_only"] = True
            self._cross_encoder = CrossEncoder(model_name, **kwargs)
        except Exception as exc:
            self._cross_encoder_error = f"failed to load {model_name}: {exc}"
            return None
        return self._cross_encoder

    def _rerank_query_hash(self, question: str) -> str:
        normalized = " ".join(question.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _fetch_rerank_cache(self, query_hash: str, model_name: str, hits: Sequence[RetrievalHit]) -> dict[str, float]:
        if os.getenv("RAG_RERANK_CACHE", "1") != "1" or not hits:
            return {}
        try:
            rows = self.db.fetch_all(
                """
                SELECT chunk_id::text AS chunk_id, score
                FROM rerank_cache
                WHERE query_hash = %s
                  AND rerank_model = %s
                  AND chunk_id = ANY(%s::uuid[])
                """,
                (query_hash, model_name, [hit.chunk.id for hit in hits]),
            )
        except Exception as exc:
            self._cross_encoder_error = f"rerank cache read failed: {exc}"
            return {}
        return {row["chunk_id"]: float(row["score"]) for row in rows}

    def _write_rerank_cache(self, query_hash: str, model_name: str, scores: dict[str, float]) -> None:
        if os.getenv("RAG_RERANK_CACHE", "1") != "1" or not scores:
            return
        for chunk_id, score in scores.items():
            try:
                self.db.execute(
                    """
                    INSERT INTO rerank_cache (query_hash, chunk_id, rerank_model, score, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (query_hash, chunk_id, rerank_model)
                    DO UPDATE SET score = EXCLUDED.score, updated_at = now()
                    """,
                    (query_hash, chunk_id, model_name, score,
                     Json({"source": "bge_cross_encoder"})),
                )
            except Exception as exc:
                self._cross_encoder_error = f"rerank cache write failed: {exc}"
                return

    def _candidate_text(self, hit: RetrievalHit) -> str:
        text = " ".join(hit.chunk.text.split())
        section = hit.chunk.section_title or ""
        return f"{hit.chunk.document_title}\n{section}\n{text}"[: int(os.getenv("RAG_RERANK_TEXT_CHARS", "1800"))]

    def _rerank_with_cross_encoder(self, question: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit] | None:
        model_name = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
        query_hash = self._rerank_query_hash(question)
        cached_scores = self._fetch_rerank_cache(query_hash, model_name, hits)
        missing_hits = [
            hit for hit in hits if hit.chunk.id not in cached_scores]
        self._last_rerank_cache_stats = {
            "enabled": os.getenv("RAG_RERANK_CACHE", "1") == "1",
            "hits": len(cached_scores),
            "misses": len(missing_hits),
            "model": model_name,
        }

        new_scores: dict[str, float] = {}
        if missing_hits:
            model = self._load_cross_encoder()
            if model is None:
                return None
            pairs = [(question, self._candidate_text(hit))
                     for hit in missing_hits]
            try:
                raw_missing_scores = model.predict(
                    pairs, batch_size=int(os.getenv("RAG_RERANK_BATCH_SIZE", "8")))
            except Exception as exc:
                self._cross_encoder_error = f"cross-encoder predict failed: {exc}"
                return None
            new_scores = {hit.chunk.id: float(score) for hit, score in zip(
                missing_hits, raw_missing_scores)}
            self._write_rerank_cache(query_hash, model_name, new_scores)

        score_by_chunk_id = {**cached_scores, **new_scores}
        if len(score_by_chunk_id) < len(hits):
            return None

        bge_scores = [score_by_chunk_id[hit.chunk.id] for hit in hits]
        recall_scores = [hit.score for hit in hits]

        def normalize(values: list[float]) -> list[float]:
            if not values:
                return []
            low = min(values)
            high = max(values)
            if high <= low:
                return [0.5 for _ in values]
            return [(value - low) / (high - low) for value in values]

        bge_norm = normalize(bge_scores)
        recall_norm = normalize(recall_scores)
        recall_weight = float(os.getenv("RAG_RERANK_RECALL_WEIGHT", "0.5"))
        bge_weight = 1.0 - recall_weight

        reranked: list[RetrievalHit] = []
        for hit, raw_score, norm_score, norm_recall in zip(hits, bge_scores, bge_norm, recall_norm):
            score = bge_weight * norm_score + recall_weight * norm_recall
            reranked.append(
                RetrievalHit(
                    rank=hit.rank,
                    score=score,
                    chunk=hit.chunk,
                    matched_terms=hit.matched_terms,
                    reason=(
                        f"{hit.reason};bge_rerank={raw_score:.4f};"
                        f"bge_norm={norm_score:.4f};recall_norm={norm_recall:.4f}"
                    ),
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _rerank_hit(self, hit: RetrievalHit, question_category: str, expanded_query: str) -> RetrievalHit:
        text = " ".join(hit.chunk.text.split()).lower()
        section = (hit.chunk.section_title or "").lower()
        title = hit.chunk.document_title.lower()
        searchable = f"{title} {section} {text}"

        rerank_bonus = 0.0
        category_terms = RERANK_CATEGORY_TERMS.get(question_category, [])
        term_hits = [term for term in category_terms if term in searchable]
        if term_hits:
            rerank_bonus += min(0.45, 0.08 * len(term_hits))
        preferred_hits = [term for term in PREFERRED_SECTION_TERMS.get(
            question_category, []) if term in section]
        if preferred_hits:
            rerank_bonus += 0.35
        if any(term in section for term in NOISY_SECTION_TERMS):
            rerank_bonus -= 0.15
        if hit.chunk.page_start == 1 and any(term in section for term in ["annual report", "notes", "contents"]):
            rerank_bonus -= 0.12
        if re.search(r"\b(20\d{2}|\d+(?:\.\d+)?%|\$?\d+(?:,\d{3})*(?:\.\d+)?)\b", text):
            rerank_bonus += 0.08
        if len(text) < 120:
            rerank_bonus -= 0.08

        phrase_hits = 0
        for phrase in QUERY_REWRITE_TERMS.get(question_category, []):
            if " " in phrase and phrase in searchable:
                phrase_hits += 1
        if phrase_hits:
            rerank_bonus += min(0.25, 0.08 * phrase_hits)

        reason = hit.reason
        if rerank_bonus:
            reason = f"{reason};rerank={rerank_bonus:.2f}"
        return RetrievalHit(
            rank=hit.rank,
            score=hit.score + rerank_bonus,
            chunk=hit.chunk,
            matched_terms=hit.matched_terms,
            reason=reason,
        )

    def retrieve(self, question: str, top_k: int = 6) -> list[RetrievalHit]:
        chunks = self.load_chunks()
        cached_hits = self._fetch_retrieval_result_cache(question, top_k, chunks)
        if cached_hits is not None:
            self._last_vector_stats = {"enabled": os.getenv("RAG_ENABLE_VECTOR", "0") == "1", "count": 0, "cache_skipped": True}
            self._last_rerank_cache_stats = {"enabled": os.getenv("RAG_RERANK_CACHE", "1") == "1", "hits": 0, "misses": 0, "cache_skipped": True}
            return cached_hits
        query_tokens = self._build_query_tokens(question)

        chapter_hits: list[RetrievalHit] = []
        paragraph_hits: list[RetrievalHit] = []
        question_category = self._question_category(question)
        target_document_terms = self._target_document_terms(question)

        # ── Target-document isolation ──────────────────────────────────────
        # When a specific company is mentioned (e.g. "Tesla"), restrict BM25
        # and vector retrieval to that company's documents only.  Without this,
        # a 60 k-chunk CATL corpus drowns out a 10 k-chunk Tesla corpus because
        # shared financial terms ("revenue", "profit") get low IDF across the
        # full collection, making CATL chunks score higher despite the -0.7
        # cross-document penalty.
        search_chunks = chunks
        if target_document_terms:
            target_only = [
                c for c in chunks
                if self._matches_target_document(c, target_document_terms)
            ]
            if len(target_only) >= top_k * 2:
                search_chunks = target_only

        for chunk in search_chunks:
            score, matched_terms, reason = self._score_chunk(
                query_tokens, chunk, question_category, target_document_terms)
            if score <= 0:
                continue
            hit = RetrievalHit(rank=0, score=score, chunk=chunk,
                               matched_terms=matched_terms, reason=reason)
            if chunk.chunk_level == "chapter":
                chapter_hits.append(hit)
            else:
                paragraph_hits.append(hit)

        chapter_hits.sort(key=lambda item: item.score, reverse=True)
        seed_parents = {
            hit.chunk.id for hit in chapter_hits[: min(3, len(chapter_hits))]}
        if seed_parents:
            paragraph_hits = [
                hit
                for hit in paragraph_hits
                if hit.chunk.parent_chunk_id in seed_parents or hit.chunk.id in seed_parents
            ] + paragraph_hits

        paragraph_hits.sort(key=lambda item: item.score, reverse=True)
        paragraph_hits = self._add_document_category_fallbacks(
            paragraph_hits, search_chunks, top_k, question_category, target_document_terms)
        paragraph_hits = self._expand_recall_context(
            paragraph_hits, search_chunks, top_k, question_category, target_document_terms)
        paragraph_hits = self._fuse_lexical_and_vector_hits(
            question, paragraph_hits, search_chunks, question_category, target_document_terms)
        if question_category == "financial":
            focus_hits = self._financial_section_hits(
                question, search_chunks, question_category, target_document_terms, top_k
            )
            if focus_hits:
                focus_ids = {hit.chunk.id for hit in focus_hits}
                paragraph_hits = focus_hits + [
                    hit for hit in paragraph_hits if hit.chunk.id not in focus_ids
                ]
            paragraph_hits = self._focus_financial_section(
                paragraph_hits, top_k)
        rerank_enabled = os.getenv("RAG_ENABLE_RERANK", "0") == "1"
        if rerank_enabled:
            candidate_k = max(top_k, int(
                os.getenv("RAG_RERANK_CANDIDATES", "30")))
            rerank_backend = os.getenv(
                "RAG_RERANK_BACKEND", "heuristic").lower()
            candidates = paragraph_hits[:candidate_k]
            if rerank_backend in {"bge", "cross_encoder", "cross-encoder"}:
                reranked = self._rerank_with_cross_encoder(
                    question, candidates)
                if reranked is None and os.getenv("RAG_RERANK_FALLBACK", "heuristic") != "none":
                    expanded_query = " ".join(
                        self._expand_query(question)).lower()
                    reranked = [self._rerank_hit(
                        hit, question_category, expanded_query) for hit in candidates]
                    reranked.sort(key=lambda item: item.score, reverse=True)
                elif reranked is None:
                    reranked = candidates
            else:
                expanded_query = " ".join(self._expand_query(question)).lower()
                reranked = [self._rerank_hit(
                    hit, question_category, expanded_query) for hit in candidates]
                reranked.sort(key=lambda item: item.score, reverse=True)
            combined = reranked[:top_k]
        else:
            combined = paragraph_hits[:top_k]
        if len(combined) < top_k:
            extra = [hit for hit in chapter_hits if hit.chunk.id not in {
                h.chunk.id for h in combined}]
            combined.extend(extra[: top_k - len(combined)])

        # Hard document filter: if target_document_terms is set AND we have
        # enough matching hits, only return chunks from the target document.
        # This prevents high-scoring CATL chunks from drowning out Tesla/BYD etc.
        if target_document_terms:
            target_hits = [h for h in combined if self._matches_target_document(h.chunk, target_document_terms)]
            if len(target_hits) >= min(3, top_k):
                combined = target_hits

        deduped: list[RetrievalHit] = []
        seen: set[str] = set()
        for idx, hit in enumerate(combined, start=1):
            if hit.chunk.id in seen:
                continue
            hit.rank = idx
            deduped.append(hit)
            seen.add(hit.chunk.id)
        self._write_retrieval_result_cache(question, top_k, deduped)
        return deduped

    def build_prompt(
        self, question: str, hits: Sequence[RetrievalHit], financial_facts: str | None = None
    ) -> str:
        lines = [
            "You are a precise enterprise knowledge assistant specializing in renewable energy industry analysis.",
            "",
            "RULES:",
            "1. Answer ONLY from the provided source documents — never invent facts.",
            "2. Cite every factual claim inline with [1], [2], etc. matching the source list.",
            "3. Be concise and direct. Lead with the key number or fact, then context.",
            "4. For financial data always state: value, unit, year, and source citation.",
            "5. In financial tables, a number in parentheses means NEGATIVE, e.g. (482) = -482. "
            "Read the value on the SAME row as the requested metric label, and the column for the asked year.",
            "6. If the documents lack sufficient evidence, say exactly what is missing — do NOT guess.",
            "7. Use the same language as the question (中文问题→中文回答, English→English).",
            "",
            f"Question: {question}",
        ]
        if financial_facts:
            lines.extend([
                "",
                "Extracted financial metrics (use these as primary evidence):",
                financial_facts,
            ])
        lines.extend(["", "Source Documents:"])
        # Safety guard against pathological chunks (chapter-level aggregates can
        # be ~1MB; rare paragraph chunks reach ~25k) blowing up the prompt.
        # Default 3000 covers paragraph p99 (~2748 chars), so normal QA chunks
        # are sent whole while monster chunks stay bounded. (The old 700 cap
        # silently dropped any fact past char 700 of a chunk.)
        snippet_chars = int(os.getenv("RAG_PROMPT_SNIPPET_CHARS", "3000"))
        for idx, hit in enumerate(hits, start=1):
            snippet = " ".join(hit.chunk.text.split())
            snippet = snippet[:snippet_chars]
            citation = hit.chunk.citation
            lines.append(f"[{idx}] {citation}")
            lines.append(f"    {snippet}")
            lines.append("")
        lines.extend(["Answer (cite sources inline):"])
        return "\n".join(lines)

    def _pick_sentence(self, question_tokens: Counter[str], text: str) -> str:
        sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return " ".join(text.split())[:280]

        best_sentence = sentences[0]
        best_score = -1
        for sentence in sentences[:8]:
            sent_tokens = self._tokenize(sentence)
            overlap = sum(min(question_tokens[t], sent_tokens[t]) for t in set(
                question_tokens) & set(sent_tokens))
            score = overlap + min(1.0, len(sentence) / 180.0)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence

    def _extractive_answer(self, question: str, hits: Sequence[RetrievalHit]) -> str:
        question_tokens = self._build_query_tokens(question)
        lines = ["基于检索到的资料，"]
        for idx, hit in enumerate(hits[:3], start=1):
            sentence = self._pick_sentence(question_tokens, hit.chunk.text)
            if not sentence.endswith(("。", ".", "!", "?", "！", "？")):
                sentence += "。"
            lines.append(f"{idx}. {sentence} [{idx}]")
        if len(lines) == 1:
            lines.append("当前检索结果不足以直接回答，需要扩大检索范围。")
        return "\n".join(lines)

    def _extract_numbers(self, text: str) -> set[str]:
        if not text:
            return set()
        pattern = re.compile(r"(?<!\d)\d+(?:,\d{3})*(?:\.\d+)?%?(?!\d)")
        matches = pattern.findall(text)
        normalized: set[str] = set()
        for match in matches:
            token = match.replace(",", "")
            normalized.add(token)
        return normalized

    @staticmethod
    def _to_float(token: str) -> float | None:
        try:
            return float(token.replace(",", "").rstrip("%"))
        except ValueError:
            return None

    def _number_grounded_in_evidence(self, value: float, evidence: Sequence[float]) -> bool:
        """A number is considered grounded if it appears in evidence directly,
        after rounding, or after a common unit scaling (万/亿/million/billion)."""
        for ev in evidence:
            # Exact or rounding tolerance (handles 48 vs 48.2, 1% relative drift).
            if abs(value - ev) <= max(0.5, abs(ev) * 0.01):
                return True
            # Unit conversion: e.g. answer "500" (亿) vs evidence "50,000" (百万).
            for factor in (1e2, 1e3, 1e4, 1e6, 1e8):
                hi, lo = (value, ev) if value >= ev else (ev, value)
                if lo and abs(hi - lo * factor) <= max(0.5, hi * 0.01):
                    return True
        return False

    def _check_number_grounding(
        self, answer: str, hits: Sequence[RetrievalHit]
    ) -> tuple[bool, list[str]]:
        """Detect-only grounding check. Never mutates the answer — it reports
        which numbers are not backed by evidence so callers can log/flag them."""
        if not answer:
            return True, []
        pattern = re.compile(r"(?<!\d)\d+(?:,\d{3})*(?:\.\d+)?%?(?!\d)")
        answer_tokens = pattern.findall(answer)
        if not answer_tokens:
            return True, []
        evidence_text = " ".join(hit.chunk.text for hit in hits)
        evidence_floats = [
            f for f in (self._to_float(t) for t in pattern.findall(evidence_text))
            if f is not None
        ]
        if not evidence_floats:
            return True, []
        ungrounded: list[str] = []
        for token in answer_tokens:
            value = self._to_float(token)
            if value is None:
                continue
            if not self._number_grounded_in_evidence(value, evidence_floats):
                ungrounded.append(token)
        return (not ungrounded), ungrounded

    @staticmethod
    def _sanitize_citations(answer: str, source_count: int) -> str:
        if not answer:
            return answer
        cleaned = re.sub(r"\[\s*\]", "", answer)

        # Citation ranges like "[5]至[9]": keep only the in-range endpoints and
        # drop the connector only when an endpoint is dropped. This avoids
        # corrupting ordinary range text such as "2020至2023年".
        def range_repl(match: re.Match[str]) -> str:
            a, b = int(match.group(1)), int(match.group(3))
            a_ok = 1 <= a <= source_count
            b_ok = 1 <= b <= source_count
            if a_ok and b_ok:
                return match.group(0)
            if a_ok:
                return f"[{a}]"
            if b_ok:
                return f"[{b}]"
            return ""

        cleaned = re.sub(
            r"\[(\d+)\]\s*(至|到|through)\s*\[(\d+)\]",
            range_repl,
            cleaned,
            flags=re.IGNORECASE,
        )

        def repl(match: re.Match[str]) -> str:
            citation_number = int(match.group(1))
            if 1 <= citation_number <= source_count:
                return match.group(0)
            return ""

        cleaned = re.sub(r"\[(\d+)\]", repl, cleaned)
        cleaned = re.sub(r"\s+([,.;，。；])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _extract_financial_metrics(
        self, hits: Sequence[RetrievalHit], question: str
    ) -> tuple[dict[str, dict[str, str]], str | None]:
        if not hits:
            return {}, None
        text = "\n".join(hit.chunk.text for hit in hits)
        normalized = " ".join(text.split())
        unit_match = re.search(r"单位[:：]\s*([^\s，。;]+)", normalized)
        unit = unit_match.group(1) if unit_match else None
        metrics: dict[str, dict[str, str]] = {}
        number_pattern = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")
        for label in FINANCIAL_METRIC_LABELS:
            start = 0
            while True:
                idx = normalized.find(label, start)
                if idx == -1:
                    break
                window = normalized[idx + len(label): idx + len(label) + 160]
                numbers = number_pattern.findall(window)
                if numbers:
                    value = numbers[0]
                    yoy = next((num for num in numbers if num.endswith("%") and num != value), "")
                    metrics[label] = {"value": value}
                    if yoy:
                        metrics[label]["yoy"] = yoy
                    break
                start = idx + len(label)
        return metrics, unit

    def _render_financial_metrics(self, metrics: dict[str, dict[str, str]], unit: str | None) -> str:
        if not metrics:
            return ""
        lines = []
        for label in FINANCIAL_METRIC_LABELS:
            if label not in metrics:
                continue
            value = metrics[label].get("value", "")
            yoy = metrics[label].get("yoy", "")
            unit_suffix = f" {unit}" if unit and value and not value.endswith("%") else ""
            if yoy:
                lines.append(f"- {label}: {value}{unit_suffix} (YoY {yoy})")
            else:
                lines.append(f"- {label}: {value}{unit_suffix}")
        return "\n".join(lines)

    def _direct_fact_answer(self, question: str, hits: Sequence[RetrievalHit]) -> str | None:
        """Answer high-confidence numeric fact questions without LLM drift.

        This is intentionally generic: it extracts values from table-like rows
        by matching the requested metric label and target year instead of
        relying on company-specific benchmark cases.
        """
        q = question.lower()
        year_match = re.search(r"\b(20\d{2})\b", question)
        target_year = year_match.group(1) if year_match else ""

        def years_before(text: str, offset: int) -> list[str]:
            years = re.findall(r"\b20\d{2}\b", text[max(0, offset - 260):offset])
            ordered: list[str] = []
            for year in years:
                if year not in ordered:
                    ordered.append(year)
            return ordered[-4:]

        def row_value(label_patterns: Sequence[str]) -> tuple[str, int, str] | None:
            for idx, hit in enumerate(hits, start=1):
                text = " ".join(hit.chunk.text.split())
                for label_pattern in label_patterns:
                    match = re.search(label_pattern, text, flags=re.IGNORECASE)
                    if match:
                        label = match.group(0).strip()
                        window = text[match.end(): match.end() + 140]
                        values = re.findall(r"\$?\s*(\(?\d+(?:,\d{3})*(?:\.\d+)?%?\)?)", window)
                        values = [value for value in values if not re.fullmatch(r"20\d{2}", value)]
                        if not values:
                            continue
                        years = years_before(text, match.start())
                        value_index = years.index(target_year) if target_year in years else 0
                        if value_index >= len(values):
                            value_index = 0
                        return values[value_index], idx, label
            return None

        is_chinese = bool(re.search(r"[一-鿿]", question))

        for metric in DIRECT_FACT_METRICS:
            if not any(trigger in q for trigger in metric["triggers"]):
                continue
            found = row_value(metric["row_patterns"])
            if not found:
                continue
            value, idx, matched_label = found
            if is_chinese:
                period = target_year + "年" if target_year else "所询问期间"
                return f"{period}{metric['label_zh']}为 {value}（来源 [{idx}]，匹配行：{matched_label}）。"
            period = target_year or "the requested period"
            return f"The {metric['label_en']} for {period} was {value} [{idx}] (matched row: {matched_label})."

        return None

    def _call_claude_api(self, prompt: str) -> str | None:
        """Call Anthropic Claude API for answer generation.

        Requires ANTHROPIC_API_KEY in env. Falls back gracefully if unavailable.
        Set RAG_LLM_BACKEND=claude to force this backend, or leave it as the
        first-choice in the default fallback chain.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        temperature = float(os.getenv("RAG_TEMPERATURE", "0.2"))

        payload = json.dumps({
            "model": model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        try:
            text = data["content"][0]["text"]
            return str(text).strip() or None
        except (KeyError, IndexError, TypeError):
            return None

    def _call_deepseek_api(self, prompt: str) -> str | None:
        """Call DeepSeek's OpenAI-compatible chat API for answer generation.

        Requires DEEPSEEK_API_KEY in env. Falls back gracefully if unavailable.
        Set RAG_LLM_BACKEND=deepseek to force this backend, or leave it as part
        of the default fallback chain. Model via DEEPSEEK_MODEL (default
        deepseek-chat); the reasoner model is unnecessary for extractive QA.
        """
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        temperature = float(os.getenv("RAG_TEMPERATURE", "0.2"))

        payload = json.dumps({
            "model": model,
            "temperature": temperature,
            "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "1024")),
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = urllib.request.Request(
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        try:
            text = data["choices"][0]["message"]["content"]
            return str(text).strip() or None
        except (KeyError, IndexError, TypeError):
            return None

    def _call_llm(self, prompt: str) -> tuple[str | None, str]:
        """Try LLM backends in priority order; return (answer, backend_used)."""
        backend = os.getenv("RAG_LLM_BACKEND", "auto").lower()

        # "none" / "skip": retrieval-only mode for fast offline evaluation
        if backend in {"none", "skip"}:
            return None, "none"

        if backend == "claude":
            answer = self._call_claude_api(prompt)
            return answer, "claude"

        if backend == "deepseek":
            answer = self._call_deepseek_api(prompt)
            return answer, "deepseek"

        # auto: Claude first (zero-latency setup), then DeepSeek
        answer = self._call_claude_api(prompt)
        if answer:
            return answer, "claude"
        answer = self._call_deepseek_api(prompt)
        return answer, "deepseek"

    def _log_query(self, question: str, prompt: str, answer: str, hits: Sequence[RetrievalHit], latency_ms: float, used_llm: bool, grounded: bool = True, ungrounded_numbers: Sequence[str] | None = None) -> None:
        try:
            columns = self._table_columns("query_logs")
        except Exception:
            return
        if not columns:
            return
        payload: dict[str, Any] = {}
        for key, value in {
            "query_text": question,
            "rewritten_query": question,
            "retrieved_chunk_ids": [hit.chunk.id for hit in hits],
            "answer_text": answer,
            "latency_ms": int(latency_ms),
            "model_name": (os.getenv("DEEPSEEK_MODEL", "deepseek-chat") if os.getenv("DEEPSEEK_API_KEY") else os.getenv("CLAUDE_MODEL")) if used_llm else "extractive-baseline",
            "total_cost": 0,
            "metadata": {
                "used_llm": used_llm,
                "prompt": prompt,
                "source_count": len(hits),
                "answer_grounded": grounded,
                "ungrounded_numbers": list(ungrounded_numbers or []),
            },
            "created_at": datetime.now(timezone.utc),
        }.items():
            if key in columns:
                payload[key] = value
        if not payload:
            return
        cols = ", ".join(payload)
        placeholders = ", ".join(f"%({key})s" for key in payload)
        sql = f"INSERT INTO query_logs ({cols}) VALUES ({placeholders})"
        payload = {key: Json(value) if isinstance(
            value, dict) else value for key, value in payload.items()}
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, payload)
            conn.commit()

    def _table_columns(self, table_name: str) -> set[str]:
        rows = self.db.fetch_all(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (table_name,),
        )
        return {row["column_name"] for row in rows}

    def _answer_cache_key(self, question: str, top_k: int) -> tuple[str, str, str]:
        query_hash = self._query_hash(question)
        corpus_fingerprint = self._corpus_version_fingerprint()
        strategy_fingerprint = self._answer_strategy_fingerprint(top_k)
        cache_key = self._hash_json(
            {
                "query_hash": query_hash,
                "corpus": corpus_fingerprint,
                "strategy": strategy_fingerprint,
                "top_k": top_k,
            }
        )
        return cache_key, query_hash, corpus_fingerprint

    def _fetch_answer_cache(self, question: str, top_k: int) -> AnswerResult | None:
        started = time.perf_counter()
        self._last_answer_cache_stats = {
            "enabled": self._cache_enabled("RAG_ANSWER_CACHE"),
            "hit": False,
        }
        if not self._cache_enabled("RAG_ANSWER_CACHE"):
            return None
        cache_key, _, _ = self._answer_cache_key(question, top_k)
        try:
            row = self.db.fetch_one(
                """
                SELECT
                    answer_text, prompt, sources, retrieved_chunks, used_llm,
                    llm_backend, latency_ms, metadata
                FROM answer_cache
                WHERE cache_key = %s
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                (cache_key,),
            )
        except Exception as exc:
            self._last_answer_cache_stats.update({"error": str(exc)})
            return None
        if not row:
            return None
        retrieved_chunks = row.get("retrieved_chunks") or []
        if isinstance(retrieved_chunks, str):
            retrieved_chunks = json.loads(retrieved_chunks)
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        debug = dict(metadata.get("debug") or {})
        original_latency_ms = float(row.get("latency_ms") or 0.0)
        lookup_latency_ms = (time.perf_counter() - started) * 1000.0
        debug["answer_cache"] = {
            "enabled": True,
            "hit": True,
            "original_latency_ms": original_latency_ms,
            "lookup_latency_ms": lookup_latency_ms,
        }
        self._last_answer_cache_stats.update({"hit": True})
        try:
            self.db.execute(
                """
                UPDATE answer_cache
                SET hit_count = hit_count + 1, updated_at = now()
                WHERE cache_key = %s
                """,
                (cache_key,),
            )
        except Exception:
            pass
        return AnswerResult(
            question=question,
            answer=str(row["answer_text"]),
            prompt=str(row.get("prompt") or ""),
            sources=list(row.get("sources") or []),
            retrieved_chunks=list(retrieved_chunks or []),
            latency_ms=lookup_latency_ms,
            used_llm=bool(row.get("used_llm")),
            debug=debug,
        )

    def _write_answer_cache(self, result: AnswerResult, llm_backend: str) -> None:
        if not self._cache_enabled("RAG_ANSWER_CACHE") or not result.answer.strip():
            return
        cache_key, query_hash, corpus_fingerprint = self._answer_cache_key(result.question, result.debug.get("top_k", 6))
        expires_sql = self._ttl_clause("RAG_ANSWER_CACHE_TTL_DAYS")
        try:
            self.db.execute(
                f"""
                INSERT INTO answer_cache (
                    cache_key, query_hash, query_text, corpus_fingerprint,
                    answer_text, prompt, sources, retrieved_chunks, used_llm,
                    llm_backend, latency_ms, metadata, expires_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, {expires_sql}, now())
                ON CONFLICT (cache_key)
                DO UPDATE SET
                    answer_text = EXCLUDED.answer_text,
                    prompt = EXCLUDED.prompt,
                    sources = EXCLUDED.sources,
                    retrieved_chunks = EXCLUDED.retrieved_chunks,
                    used_llm = EXCLUDED.used_llm,
                    llm_backend = EXCLUDED.llm_backend,
                    latency_ms = EXCLUDED.latency_ms,
                    metadata = EXCLUDED.metadata,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = now()
                """,
                (
                    cache_key,
                    query_hash,
                    result.question,
                    corpus_fingerprint,
                    result.answer,
                    result.prompt,
                    result.sources,
                    Json(result.retrieved_chunks),
                    result.used_llm,
                    llm_backend,
                    result.latency_ms,
                    Json({"debug": result.debug, "source": "answer_cache"}),
                ),
            )
        except Exception as exc:
            self._last_answer_cache_stats.update({"write_error": str(exc)})

    def _lookup_fact(self, question: str) -> str | None:
        """Facts-first answering: resolve (company, metric, year) against the
        structured research_facts store. Exact and unambiguous — statement_type
        lets the consolidated statement win over segment/summary rows. Returns a
        formatted answer or None to fall through to the LLM. (Prototype scope.)
        """
        q = question.lower()
        company = next((toks[0] for name, toks in DOCUMENT_HINTS.items() if name in q), None)
        metric = next((key for hint, key in FACT_METRIC_HINTS if hint in q), None)
        if not company or not metric:
            return None
        year_match = re.search(r"\b(20\d{2})\b", question)
        year = year_match.group(1) if year_match else None
        try:
            rows = self.db.fetch_all(
                """
                SELECT value, unit, period, source_citation,
                       (metadata->>'statement_type') AS statement_type
                FROM research_facts
                WHERE company = %s AND metric = %s AND review_status = 'approved'
                  AND (%s::text IS NULL OR period = %s)
                ORDER BY (metadata->>'statement_type' = 'consolidated_income_statement') DESC,
                         confidence DESC
                LIMIT 1
                """,
                (company, metric, year, year),
            )
        except Exception:
            return None
        if not rows:
            return None
        row = rows[0]
        value = row["value"]
        unit = f" {row['unit']}" if row.get("unit") else ""
        period = row.get("period") or ""
        cite = row["source_citation"]
        if re.search(r"[一-鿿]", question):
            return f"{period}{('年' if period else '')}{value}{unit}（来源：{cite}）。"
        return f"For {period or 'the requested period'}, the value was {value}{unit} (source: {cite})."

    def _compose_answer(
        self,
        question: str,
        hits: Sequence[RetrievalHit],
        prompt: str,
        financial_facts: str,
    ) -> ComposedAnswer:
        """Single answer pipeline. Stages run in a fixed order so the three
        number-handling mechanisms can no longer interleave unpredictably:

          1. generate   — direct-fact → LLM → extractive (mutually exclusive)
          2. ground     — detect-only check, never mutates the generated text
          3. enrich     — if ungrounded, append grounded metrics (non-mutating)
          4. sanitize   — clean up citation markers

        Returns the text plus provenance/grounding metadata.
        """
        # ── Stage 0: facts-first (structured store) ───────────────────────
        # Exact (company, metric, year) lookup against research_facts; unlike the
        # regex extractor this is unambiguous and prefers the consolidated
        # statement, so it answers correctly where the LLM mis-picks among
        # similar rows (e.g. Vestas EBIT). Falls through when no fact matches.
        fact_answer = self._lookup_fact(question)
        if fact_answer:
            return ComposedAnswer(
                self._sanitize_citations(fact_answer, len(hits)),
                "facts-store", False, True, [],
            )

        # ── Stage 1: generate (exactly one source wins) ───────────────────
        # The direct-fact regex extractor is fragile: it grabs the first textual
        # match of a metric label, which can be prose ("R&D rose 36.56%") rather
        # than the value. With table-aware parsing the LLM reads clean rows and
        # extracts more reliably, so direct-fact is off by default.
        direct_answer = (
            self._direct_fact_answer(question, hits)
            if os.getenv("RAG_ENABLE_DIRECT_FACT", "0") == "1"
            else None
        )
        if direct_answer:
            answer, backend, used_llm = direct_answer, "direct-extractive", False
        else:
            llm_answer, backend = self._call_llm(prompt)
            if llm_answer and llm_answer.strip():
                answer, used_llm = llm_answer, True
            else:
                answer, backend, used_llm = self._extractive_answer(question, hits), "extractive", False

        # ── Stage 2: ground-check (detect only) ───────────────────────────
        # Direct-fact answers are extracted verbatim from evidence rows, so
        # they are grounded by construction.
        if backend == "direct-extractive":
            grounded, ungrounded_numbers = True, []
        else:
            grounded, ungrounded_numbers = self._check_number_grounding(answer, hits)

        # ── Stage 3: enrich (never mutate the generated numbers) ───────────
        if not grounded and financial_facts:
            answer = answer.rstrip() + "\n\n指标摘录（来自证据）：\n" + financial_facts

        # ── Stage 4: sanitize citations ───────────────────────────────────
        answer = self._sanitize_citations(answer, len(hits))
        return ComposedAnswer(answer, backend, used_llm, grounded, ungrounded_numbers)

    def answer(self, question: str, top_k: int = 6) -> AnswerResult:
        started = time.perf_counter()
        cached_answer = self._fetch_answer_cache(question, top_k)
        if cached_answer is not None:
            return cached_answer
        hits = self.retrieve(question, top_k=top_k)
        metrics, unit = self._extract_financial_metrics(hits, question)
        financial_facts = self._render_financial_metrics(metrics, unit)
        prompt = self.build_prompt(question, hits, financial_facts or None)
        composed = self._compose_answer(question, hits, prompt, financial_facts)
        answer = composed.answer
        llm_backend = composed.backend
        used_llm = composed.used_llm
        grounded = composed.grounded
        ungrounded_numbers = composed.ungrounded_numbers
        sources = [f"[{idx}] {hit.chunk.citation}" for idx,
                   hit in enumerate(hits, start=1)]
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_query(question, prompt, answer, hits, latency_ms, used_llm, grounded, ungrounded_numbers)
        result = AnswerResult(
            question=question,
            answer=answer,
            prompt=prompt,
            sources=sources,
            retrieved_chunks=[
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 4),
                    "chunk_id": hit.chunk.id,
                    "document_title": hit.chunk.document_title,
                    "chunk_level": hit.chunk.chunk_level,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                    "section_title": hit.chunk.section_title,
                    "text": hit.chunk.text,
                    "matched_terms": hit.matched_terms,
                    "reason": hit.reason,
                }
                for hit in hits
            ],
            latency_ms=latency_ms,
            used_llm=used_llm,
            debug={
                "top_k": top_k,
                "answer_grounded": grounded,
                "ungrounded_numbers": ungrounded_numbers,
                "llm_backend": llm_backend,
                "llm_backend_config": os.getenv("RAG_LLM_BACKEND", "auto"),
                "claude_model": os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
                "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "query_terms": list(self._build_query_tokens(question).keys())[:40],
                "retrieval_count": len(hits),
                "question_category": self._question_category(question),
                "target_document_terms": self._target_document_terms(question),
                "context_expansion_min_k": int(os.getenv("RAG_CONTEXT_EXPANSION_MIN_K", "20")),
                "doc_fallback_min_k": int(os.getenv("RAG_DOC_FALLBACK_MIN_K", "20")),
                "rerank_enabled": os.getenv("RAG_ENABLE_RERANK", "0") == "1",
                "rerank_backend": os.getenv("RAG_RERANK_BACKEND", "heuristic"),
                "rerank_model": os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base"),
                "rerank_error": self._cross_encoder_error,
                "rerank_cache": self._last_rerank_cache_stats,
                "query_embedding_cache": self._last_query_embedding_cache_stats,
                "retrieval_result_cache": self._last_retrieval_cache_stats,
                "answer_cache": self._last_answer_cache_stats,
                "vector_search": self._last_vector_stats,
                "embedding_error": self._embedding_error,
                "rerank_candidates": int(os.getenv("RAG_RERANK_CANDIDATES", "30")),
            },
        )
        self._write_answer_cache(result, llm_backend)
        return result

    def warmup(self) -> dict[str, Any]:
        """Load expensive local state before the first user-facing query."""
        started = time.perf_counter()
        chunks = self.load_chunks()
        self._term_idf()
        warmed: dict[str, Any] = {
            "chunks": len(chunks),
            "idf_terms": len(self._idf_cache or {}),
            "embedding_model": False,
            "rerank_model": False,
        }
        if os.getenv("RAG_ENABLE_VECTOR", "0") == "1" and os.getenv("RAG_WARMUP_EMBED_MODEL", "1") == "1":
            warmed["embedding_model"] = self._load_embedding_model() is not None
            if self._embedding_error:
                warmed["embedding_error"] = self._embedding_error
        if os.getenv("RAG_ENABLE_RERANK", "0") == "1" and os.getenv("RAG_WARMUP_RERANK_MODEL", "1") == "1":
            warmed["rerank_model"] = self._load_cross_encoder() is not None
            if self._cross_encoder_error:
                warmed["rerank_error"] = self._cross_encoder_error
        warmed["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return warmed


def format_answer(result: AnswerResult) -> str:
    lines = [
        f"Question: {result.question}",
        "",
        "Answer:",
        result.answer,
        "",
        "Sources:",
    ]
    lines.extend(result.sources or ["(none)"])
    lines.extend(
        [
            "",
            "Retrieved chunks:",
        ]
    )
    for hit in result.retrieved_chunks:
        matched_terms = ", ".join(hit.get('matched_terms', [])[:6]) or "-"
        lines.append(
            f"- #{hit['rank']} score={hit['score']} {hit['document_title']} "
            f"{hit['chunk_level']} p.{hit['page_start'] or '?'}-{hit['page_end'] or '?'} :: "
            f"{matched_terms}"
        )
    lines.extend(
        [
            "",
            f"Latency: {result.latency_ms:.1f} ms",
            f"LLM used: {result.used_llm} (backend: {result.debug.get('llm_backend', '?')})",
            "",
            "Debug:",
            json.dumps(result.debug, ensure_ascii=False, indent=2),
        ]
    )
    return "\n".join(lines)
