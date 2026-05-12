"""
evaluate_retrieval.py — RAG 检索策略对比评测

对比 4 种检索策略的标准信息检索指标：
  1. bm25    — 纯 BM25 关键词检索（无向量）
  2. vector  — 纯向量 ANN 检索（pgvector cosine，无 BM25）
  3. hybrid  — BM25 + 向量 RRF 融合，无重排序
  4. full    — BM25 + 向量 + BGE 重排序（生产配置）

评测指标：
  Hit@1 / Hit@3 / Hit@5   — 正确文档出现在 Top-K 的比例
  MRR@10                   — 平均倒数排名（Mean Reciprocal Rank）
  NDCG@5                   — 归一化折损累积增益
  mean_latency_ms          — 平均检索延迟（不含 LLM）

用法：
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --dataset data/golden_dataset.json --top-k 5
    python scripts/evaluate_retrieval.py --strategies bm25,vector,hybrid,full
    python scripts/evaluate_retrieval.py --with-answer-eval   # 额外跑 LLM 答案质量（较慢）
    python scripts/evaluate_retrieval.py --filter-category financial
    python scripts/evaluate_retrieval.py --filter-lang en
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from rag_assistant.core import Database, RetrievalService, ChunkRecord, RetrievalHit

# ─── ANSI 颜色 ───────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
DIM    = "\033[2m"

DEFAULT_DATASET = ROOT / "data" / "golden_dataset.json"
REPORTS_DIR = ROOT / "eval_reports"


# ─── 策略配置 ─────────────────────────────────────────────────────────────────
STRATEGY_CONFIGS: dict[str, dict[str, str]] = {
    "bm25": {
        "RAG_ENABLE_VECTOR": "0",
        "RAG_ENABLE_RERANK": "0",
        "RAG_LLM_BACKEND": "none",
        "label": "BM25-only      (TF-IDF 关键词)",
    },
    "vector": {
        # vector-only 通过直接 SQL 实现，不走 BM25 pipeline
        "RAG_LLM_BACKEND": "none",
        "label": "Vector-only    (pgvector ANN)",
    },
    "hybrid": {
        "RAG_ENABLE_VECTOR": "1",
        "RAG_ENABLE_RERANK": "0",
        "RAG_LLM_BACKEND": "none",
        "label": "Hybrid         (BM25 + Vector, no rerank)",
    },
    "full": {
        "RAG_ENABLE_VECTOR": "1",
        "RAG_ENABLE_RERANK": "1",
        "RAG_LLM_BACKEND": "none",
        "label": "Hybrid+Rerank  (生产配置 ★)",
    },
}


# ─── 数据结构 ─────────────────────────────────────────────────────────────────
@dataclass
class EvalCase:
    id: str
    question: str
    gold_doc: str           # 必须与 DB 中 document_title 完全匹配（子串即可）
    gold_page_range: list[int] | None   # [start, end] 或 null
    expected_keywords: list[str]
    category: str
    difficulty: str
    language: str
    verified: bool = False
    notes: str = ""


@dataclass
class CaseResult:
    case_id: str
    question: str
    gold_doc: str
    strategy: str
    doc_rank: int | None        # 正确文档首次出现的排名（1-indexed），None 表示未命中
    page_rank: int | None       # 正确页面首次出现的排名，None 表示未命中
    retrieved_docs: list[str] = field(default_factory=list)  # 检索到的文档标题列表（去重）
    latency_ms: float = 0.0
    keyword_hit: bool = False   # 检索文本中是否包含期望关键词


@dataclass
class StrategyMetrics:
    strategy: str
    label: str
    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    page_hit_at_5: float
    mean_latency_ms: float
    keyword_hit_rate: float


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def _doc_matches(chunk_title: str, gold_doc: str) -> bool:
    """检查 chunk 所属文档是否是目标文档（大小写不敏感子串匹配）。"""
    return gold_doc.lower() in chunk_title.lower()


def _page_matches(chunk: dict, gold_page_range: list[int] | None) -> bool:
    """检查 chunk 的页码是否落在 gold_page_range 内。"""
    if gold_page_range is None:
        return True  # 不限页码，只要文档对就算命中
    p_start = chunk.get("page_start")
    p_end = chunk.get("page_end") or p_start
    if p_start is None:
        return False
    gold_start, gold_end = gold_page_range
    # 允许 ±5 页的容差
    return p_start <= gold_end + 5 and (p_end or p_start) >= gold_start - 5


def _find_rank(
    hits: Sequence[dict],
    gold_doc: str,
    gold_page_range: list[int] | None,
    match_page: bool = False,
) -> int | None:
    """返回第一个命中的排名（1-indexed），未命中返回 None。"""
    for rank, chunk in enumerate(hits, 1):
        title = chunk.get("document_title", "")
        if _doc_matches(title, gold_doc):
            if not match_page:
                return rank
            if _page_matches(chunk, gold_page_range):
                return rank
    return None


def _keyword_hit(hits: Sequence[dict], keywords: list[str]) -> bool:
    """检索结果的文本中是否包含任意一个期望关键词。"""
    all_text = " ".join(chunk.get("text", "") for chunk in hits[:5]).lower()
    return any(kw.lower() in all_text for kw in keywords)


def _ndcg_at_k(rank: int | None, k: int = 5) -> float:
    """计算单条结果的 NDCG@K（binary relevance）。"""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)   # 归一化系数：ideal DCG = 1/log2(2) = 1.0


def _compute_metrics(results: list[CaseResult], label: str, strategy: str) -> StrategyMetrics:
    n = len(results)
    if n == 0:
        raise ValueError("No results to compute metrics from.")

    def _avg(vals):
        return sum(vals) / n

    return StrategyMetrics(
        strategy=strategy,
        label=label,
        n=n,
        hit_at_1=_avg([1 if r.doc_rank == 1 else 0 for r in results]),
        hit_at_3=_avg([1 if r.doc_rank is not None and r.doc_rank <= 3 else 0 for r in results]),
        hit_at_5=_avg([1 if r.doc_rank is not None and r.doc_rank <= 5 else 0 for r in results]),
        mrr=_avg([1.0 / r.doc_rank if r.doc_rank is not None else 0.0 for r in results]),
        ndcg_at_5=_avg([_ndcg_at_k(r.doc_rank, k=5) for r in results]),
        page_hit_at_5=_avg([1 if r.page_rank is not None and r.page_rank <= 5 else 0 for r in results]),
        mean_latency_ms=_avg([r.latency_ms for r in results]),
        keyword_hit_rate=_avg([1 if r.keyword_hit else 0 for r in results]),
    )


# ─── 向量检索（直接 SQL，绕过 BM25） ─────────────────────────────────────────
def _embed_query(question: str, model_name: str = "BAAI/bge-m3") -> list[float]:
    """用 sentence-transformers 生成查询向量。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    model = SentenceTransformer(model_name, device=device)
    vec = model.encode(question, normalize_embeddings=True)
    return [float(v) for v in vec.tolist()]


_embed_model_cache: dict[str, Any] = {}


def _get_embed_model(model_name: str = "BAAI/bge-m3") -> Any:
    """懒加载 embedding 模型（每次进程只加载一次）。"""
    if model_name not in _embed_model_cache:
        print(f"  {DIM}[加载向量模型 {model_name}...]{RESET}", flush=True)
        try:
            from sentence_transformers import SentenceTransformer
            device = "cpu"
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass
            _embed_model_cache[model_name] = SentenceTransformer(model_name, device=device)
        except ImportError:
            raise RuntimeError("sentence-transformers not installed")
    return _embed_model_cache[model_name]


def _vector_only_retrieve(
    question: str,
    db: Database,
    top_k: int = 5,
    model_name: str = "BAAI/bge-m3",
) -> tuple[list[dict], float]:
    """
    纯向量检索：直接对 embeddings 表做 cosine 相似度排序，返回 top-k chunks。
    返回 (chunks_list, latency_ms)
    """
    t0 = time.perf_counter()
    model = _get_embed_model(model_name)
    vec = model.encode(question, normalize_embeddings=True)
    vec_str = "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"

    rows = db.fetch_all(
        """
        SELECT
            c.id::text        AS chunk_id,
            c.text,
            c.page_start,
            c.page_end,
            c.section_title,
            c.chunk_level,
            d.title           AS document_title,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        JOIN documents d ON d.id = dv.document_id
        WHERE e.embedding_model = %s
          AND c.chunk_level = 'paragraph'
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_str, model_name, vec_str, top_k),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return [dict(r) for r in rows], latency_ms


# ─── 检索策略运行器 ──────────────────────────────────────────────────────────
def _run_bm25_or_hybrid(
    service: RetrievalService,
    question: str,
    top_k: int,
    env_overrides: dict[str, str],
) -> tuple[list[dict], float]:
    """
    通过临时修改 env 变量，用 BM25-only 或 Hybrid 策略调用 service.answer()。
    RAG_LLM_BACKEND=none 确保跳过 LLM，只测检索部分。
    返回 (retrieved_chunks_dicts, retrieval_latency_ms)
    """
    # 强制跳过 LLM
    full_overrides = {**env_overrides, "RAG_LLM_BACKEND": "none"}
    original = {k: os.environ.get(k) for k in full_overrides}
    try:
        os.environ.update(full_overrides)
        result = service.answer(question, top_k=top_k)
        # latency_ms 含少量 extractive fallback 开销，仍以检索为主
        return list(result.retrieved_chunks), result.latency_ms
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_strategy(
    strategy: str,
    service: RetrievalService,
    db: Database,
    question: str,
    top_k: int,
) -> tuple[list[dict], float]:
    """分发到对应的检索策略。返回 (chunk_dicts, latency_ms)。"""
    cfg = STRATEGY_CONFIGS[strategy]

    if strategy == "vector":
        model_name = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
        return _vector_only_retrieve(question, db, top_k=top_k, model_name=model_name)

    env_overrides = {k: v for k, v in cfg.items() if k != "label"}
    return _run_bm25_or_hybrid(service, question, top_k, env_overrides)


# ─── 答案质量评测（可选，需要 LLM） ─────────────────────────────────────────
def _answer_quality_eval(
    service: RetrievalService,
    cases: list[EvalCase],
    top_k: int,
    sample_size: int = 10,
) -> dict[str, float]:
    """
    对前 sample_size 条 case 跑完整 answer()（含 LLM 生成），
    评估答案的关键词命中率。
    """
    print(f"\n{BOLD}[答案质量评测 — {sample_size} 条样本]{RESET}")
    os.environ.pop("RAG_LLM_BACKEND", None)  # 恢复真实 LLM

    keyword_hits = 0
    total = 0
    for case in cases[:sample_size]:
        print(f"  Q: {case.question[:60]}...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = service.answer(case.question, top_k=top_k)
        latency = (time.perf_counter() - t0) * 1000
        answer_lower = result.answer.lower()
        hit = any(kw.lower() in answer_lower for kw in case.expected_keywords)
        keyword_hits += int(hit)
        total += 1
        status = f"{GREEN}✓{RESET}" if hit else f"{RED}✗{RESET}"
        print(f"{status} ({latency:.0f}ms)")

    rate = keyword_hits / total if total > 0 else 0.0
    print(f"\n  关键词命中率: {BOLD}{rate:.1%}{RESET} ({keyword_hits}/{total})")
    return {"keyword_hit_rate": rate, "sample_size": total}


# ─── 主评测循环 ──────────────────────────────────────────────────────────────
def evaluate(
    dataset_path: Path,
    strategies: list[str],
    top_k: int = 5,
    filter_category: str | None = None,
    filter_lang: str | None = None,
    with_answer_eval: bool = False,
    answer_eval_samples: int = 10,
) -> dict[str, Any]:
    # 加载 golden dataset
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    cases: list[EvalCase] = []
    for item in data["cases"]:
        if filter_category and item["category"] != filter_category:
            continue
        if filter_lang and item["language"] != filter_lang:
            continue
        cases.append(EvalCase(
            id=item["id"],
            question=item["question"],
            gold_doc=item["gold_doc"],
            gold_page_range=item.get("gold_page_range"),
            expected_keywords=item.get("expected_keywords", []),
            category=item["category"],
            difficulty=item["difficulty"],
            language=item["language"],
            verified=item.get("verified", False),
            notes=item.get("notes", ""),
        ))

    if not cases:
        print("❌ 没有匹配的测试案例，请检查过滤条件")
        return {}

    print(f"\n{BOLD}📋 评测数据集: {dataset_path.name}{RESET}")
    print(f"   案例总数: {len(cases)}  |  Top-K: {top_k}  |  策略: {', '.join(strategies)}")
    if filter_category:
        print(f"   过滤类别: {filter_category}")
    if filter_lang:
        print(f"   过滤语言: {filter_lang}")

    # 连接数据库
    db = Database()
    service = RetrievalService(db)

    all_strategy_results: dict[str, list[CaseResult]] = {}
    all_strategy_metrics: dict[str, StrategyMetrics] = {}

    for strategy in strategies:
        label = STRATEGY_CONFIGS[strategy]["label"]
        print(f"\n{BOLD}{CYAN}▶ 策略: {label}{RESET}")

        results: list[CaseResult] = []

        for i, case in enumerate(cases, 1):
            print(f"  [{i:2d}/{len(cases)}] {case.question[:55]}...", end=" ", flush=True)

            try:
                hits, latency_ms = run_strategy(strategy, service, db, case.question, top_k)
            except Exception as e:
                print(f"{RED}ERROR: {e}{RESET}")
                results.append(CaseResult(
                    case_id=case.id, question=case.question,
                    gold_doc=case.gold_doc, strategy=strategy,
                    doc_rank=None, page_rank=None,
                    latency_ms=0.0,
                ))
                continue

            doc_rank = _find_rank(hits, case.gold_doc, case.gold_page_range, match_page=False)
            page_rank = _find_rank(hits, case.gold_doc, case.gold_page_range, match_page=True)
            kw_hit = _keyword_hit(hits, case.expected_keywords)

            retrieved_docs = []
            seen = set()
            for h in hits:
                t = h.get("document_title", "")
                if t and t not in seen:
                    seen.add(t)
                    retrieved_docs.append(t)

            results.append(CaseResult(
                case_id=case.id,
                question=case.question,
                gold_doc=case.gold_doc,
                strategy=strategy,
                doc_rank=doc_rank,
                page_rank=page_rank,
                retrieved_docs=retrieved_docs,
                latency_ms=latency_ms,
                keyword_hit=kw_hit,
            ))

            # 打印单条结果
            rank_str = f"rank={doc_rank}" if doc_rank else "miss "
            color = GREEN if doc_rank and doc_rank <= 3 else (YELLOW if doc_rank else RED)
            print(f"{color}{rank_str}{RESET} {DIM}({latency_ms:.0f}ms){RESET}")

        metrics = _compute_metrics(results, label, strategy)
        all_strategy_results[strategy] = results
        all_strategy_metrics[strategy] = metrics

    # ── 答案质量评测（可选）─────────────────────────────────────────────────
    answer_eval_result = None
    if with_answer_eval:
        answer_eval_result = _answer_quality_eval(
            service, cases, top_k, sample_size=answer_eval_samples
        )

    return {
        "dataset": str(dataset_path),
        "top_k": top_k,
        "n_cases": len(cases),
        "strategies": strategies,
        "metrics": all_strategy_metrics,
        "case_results": all_strategy_results,
        "answer_eval": answer_eval_result,
    }


# ─── 输出对比表格 ─────────────────────────────────────────────────────────────
def print_comparison_table(metrics: dict[str, StrategyMetrics]) -> None:
    print(f"\n{'='*90}")
    print(f"{BOLD}  策略对比结果{RESET}")
    print(f"{'='*90}")
    header = f"  {'策略':<35} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'NDCG@5':>7} {'P.Hit@5':>8} {'KW%':>5} {'延迟ms':>7}"
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'-'*86}")

    ordered = sorted(metrics.values(), key=lambda m: m.hit_at_5, reverse=True)
    medal = ["🥇", "🥈", "🥉", "  "]

    for rank_idx, m in enumerate(ordered):
        icon = medal[min(rank_idx, 3)]
        is_best = rank_idx == 0
        color = BOLD if is_best else ""
        line = (
            f"  {icon} {m.label:<33} "
            f"{m.hit_at_1:>6.1%} "
            f"{m.hit_at_3:>6.1%} "
            f"{m.hit_at_5:>6.1%} "
            f"{m.mrr:>6.3f} "
            f"{m.ndcg_at_5:>7.3f} "
            f"{m.page_hit_at_5:>8.1%} "
            f"{m.keyword_hit_rate:>5.1%} "
            f"{m.mean_latency_ms:>7.0f}"
        )
        print(f"{color}{line}{RESET}")

    print(f"{'='*90}")
    print(f"\n  指标说明:")
    print(f"  {DIM}Hit@K    — 正确文档出现在 Top-K 的比例（越高越好）{RESET}")
    print(f"  {DIM}MRR      — 平均倒数排名，衡量正确结果的排名质量（最大=1.0）{RESET}")
    print(f"  {DIM}NDCG@5   — 归一化折损累积增益，综合考虑相关性和排名位置{RESET}")
    print(f"  {DIM}P.Hit@5  — 同时满足文档和页码匹配的命中率（更严格）{RESET}")
    print(f"  {DIM}KW%      — 检索结果文本中包含期望关键词的比例{RESET}")


def print_failure_analysis(
    case_results: dict[str, list[CaseResult]],
    cases: list[EvalCase],
) -> None:
    """打印 full 策略的失败案例分析。"""
    full_results = case_results.get("full", [])
    if not full_results:
        return

    failures = [r for r in full_results if r.doc_rank is None or r.doc_rank > 5]
    if not failures:
        print(f"\n{GREEN}✅ 生产策略（Hybrid+Rerank）无 Top-5 失败案例！{RESET}")
        return

    print(f"\n{BOLD}⚠️  失败案例分析（Hybrid+Rerank，共 {len(failures)} 条）:{RESET}")
    case_map = {c.id: c for c in cases}
    for r in failures[:8]:
        case = case_map.get(r.case_id)
        diff = f" [{case.difficulty}]" if case else ""
        print(f"  {RED}✗{RESET} [{r.case_id}]{diff} {r.question[:65]}")
        print(f"    期望文档: {r.gold_doc}")
        print(f"    实际召回: {r.retrieved_docs[:3]}")


# ─── 报告保存 ─────────────────────────────────────────────────────────────────
def save_report(result: dict[str, Any], metrics: dict[str, StrategyMetrics]) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"retrieval_eval_{ts}.json"

    serializable_metrics = {}
    for k, m in metrics.items():
        serializable_metrics[k] = {
            "strategy": m.strategy,
            "label": m.label,
            "n": m.n,
            "hit_at_1": round(m.hit_at_1, 4),
            "hit_at_3": round(m.hit_at_3, 4),
            "hit_at_5": round(m.hit_at_5, 4),
            "mrr": round(m.mrr, 4),
            "ndcg_at_5": round(m.ndcg_at_5, 4),
            "page_hit_at_5": round(m.page_hit_at_5, 4),
            "mean_latency_ms": round(m.mean_latency_ms, 1),
            "keyword_hit_rate": round(m.keyword_hit_rate, 4),
        }

    # 每条 case 结果
    case_details = {}
    for strategy, results in result.get("case_results", {}).items():
        case_details[strategy] = [
            {
                "case_id": r.case_id,
                "question": r.question,
                "gold_doc": r.gold_doc,
                "doc_rank": r.doc_rank,
                "page_rank": r.page_rank,
                "retrieved_docs": r.retrieved_docs[:5],
                "latency_ms": round(r.latency_ms, 1),
                "keyword_hit": r.keyword_hit,
            }
            for r in results
        ]

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": result.get("dataset"),
        "top_k": result.get("top_k"),
        "n_cases": result.get("n_cases"),
        "metrics": serializable_metrics,
        "case_details": case_details,
        "answer_eval": result.get("answer_eval"),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


# ─── 摘要建议 ─────────────────────────────────────────────────────────────────
def print_summary(metrics: dict[str, StrategyMetrics]) -> None:
    full = metrics.get("full")
    bm25 = metrics.get("bm25")
    vector = metrics.get("vector")

    if not full:
        return

    print(f"\n{BOLD}📈 关键发现:{RESET}")

    if bm25 and vector:
        bm25_gain = (full.hit_at_5 - bm25.hit_at_5) * 100
        vec_gain = (full.hit_at_5 - vector.hit_at_5) * 100
        print(f"  • Hybrid+Rerank 相比纯 BM25 Hit@5 提升 {bm25_gain:+.1f}%")
        print(f"  • Hybrid+Rerank 相比纯向量 Hit@5 提升 {vec_gain:+.1f}%")

    if full.hit_at_5 >= 0.85:
        grade = f"{GREEN}优秀 🌟{RESET}"
    elif full.hit_at_5 >= 0.70:
        grade = f"{YELLOW}良好 ✅{RESET}"
    else:
        grade = f"{RED}需改进 ⚠️{RESET}"

    print(f"  • 生产配置总体评级: {grade}  (Hit@5={full.hit_at_5:.1%}, MRR={full.mrr:.3f})")

    print(f"\n{DIM}可在 README 中引用：{RESET}")
    print(f'  "在 {full.n} 条 golden dataset 测试中，混合检索+重排序方案的 Hit@5 达到 '
          f'{full.hit_at_5:.0%}，MRR 为 {full.mrr:.2f}，优于纯 BM25（{bm25.hit_at_5:.0%}）'
          f'和纯向量检索（{vector.hit_at_5:.0%}）"' if bm25 and vector else "")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比 BM25 / Vector / Hybrid / Hybrid+Rerank 四种检索策略"
    )
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET),
        help=f"golden dataset JSON 路径（默认：{DEFAULT_DATASET}）"
    )
    parser.add_argument(
        "--strategies", default="bm25,vector,hybrid,full",
        help="逗号分隔的策略名，可选：bm25,vector,hybrid,full"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="每次检索返回的 chunk 数量（默认：5）"
    )
    parser.add_argument(
        "--filter-category", default=None,
        help="只跑特定类别的 case（financial / operational / technical / strategic）"
    )
    parser.add_argument(
        "--filter-lang", default=None,
        help="只跑特定语言的 case（en / zh）"
    )
    parser.add_argument(
        "--with-answer-eval", action="store_true",
        help="同时评测 LLM 答案质量（会启动 Ollama，较慢）"
    )
    parser.add_argument(
        "--answer-samples", type=int, default=10,
        help="答案质量评测的样本数（默认：10）"
    )
    parser.add_argument(
        "--no-vector", action="store_true",
        help="跳过 vector-only 策略（避免加载 embedding 模型）"
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    if args.no_vector and "vector" in strategies:
        strategies.remove("vector")
        print(f"{YELLOW}[跳过 vector-only 策略]{RESET}")

    # 验证策略名
    invalid = [s for s in strategies if s not in STRATEGY_CONFIGS]
    if invalid:
        print(f"❌ 未知策略: {invalid}. 可选: {list(STRATEGY_CONFIGS.keys())}")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ 找不到 dataset 文件: {dataset_path}")
        sys.exit(1)

    print(f"\n{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  RAG 检索策略对比评测{RESET}")
    print(f"{BOLD}{'='*90}{RESET}")

    result = evaluate(
        dataset_path=dataset_path,
        strategies=strategies,
        top_k=args.top_k,
        filter_category=args.filter_category,
        filter_lang=args.filter_lang,
        with_answer_eval=args.with_answer_eval,
        answer_eval_samples=args.answer_samples,
    )

    if not result:
        sys.exit(1)

    # 打印对比表格
    metrics = result["metrics"]
    print_comparison_table(metrics)

    # 加载 case 列表用于失败分析
    with open(dataset_path, encoding="utf-8") as f:
        raw_cases = [EvalCase(**{
            k: v for k, v in item.items()
            if k in EvalCase.__dataclass_fields__
        }) for item in json.load(f)["cases"]]

    print_failure_analysis(result["case_results"], raw_cases)
    print_summary(metrics)

    # 保存报告
    report_path = save_report(result, metrics)
    print(f"\n{DIM}📁 详细报告已保存: {report_path}{RESET}")
    print(f"\n{BOLD}评测完成！{RESET}\n")


if __name__ == "__main__":
    main()
