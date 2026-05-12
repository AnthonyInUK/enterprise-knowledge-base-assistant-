"""
benchmark_models.py — Ollama 多模型对比测试

对候选模型跑同一批测试问题，输出答案质量、引用率、延迟排名。

用法:
    # 先 pull 想测试的模型（见下方命令），然后：
    python scripts/benchmark_models.py

    # 只测指定模型
    python scripts/benchmark_models.py --models qwen2.5:7b-instruct,mistral:7b-instruct

    # 调整测试问题数量（越少越快）
    python scripts/benchmark_models.py --questions 3
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ─── 候选模型 ────────────────────────────────────────────────────────────────
# 按推荐优先级排列（适合中英混合企业文档场景）
CANDIDATE_MODELS = [
    "qwen2.5:7b-instruct",      # 当前配置，中文最强
    "qwen2.5:14b-instruct",     # 质量更高，需要 ~10GB RAM
    "qwen2.5:3b-instruct",      # 最快，适合资源受限
    "mistral:7b-instruct",      # 英文强，中文一般
    "llama3.1:8b",              # Meta 官方，英文好
    "deepseek-r1:7b",           # 推理型，适合复杂问题
    "gemma2:9b",                # Google，中英平衡
]

# ─── 测试问题集 ───────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "特斯拉2023年的总营收是多少？",
    "宁德时代的主要产品线有哪些？",
    "What is LONGi's solar panel shipment volume in 2023?",
    "阳光电源的储能业务收入占比是多少？",
    "BYD's revenue breakdown by segment in 2023?",
    "隆基绿能2023年净利润同比变化情况？",
    "What are the key risks facing First Solar?",
]

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model: str
    question: str
    answer: str
    latency_ms: float
    has_citation: bool       # 有 [1] [2] 等引用
    answer_length: int
    error: str = ""


@dataclass
class ModelSummary:
    model: str
    avg_latency_ms: float
    citation_rate: float     # 有引用的答案比例
    avg_answer_len: float
    error_rate: float
    results: list[ModelResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """综合评分 (越高越好):
           引用率 × 60 + 速度分 × 20 + 长度分 × 20
        """
        citation_score = self.citation_rate * 60
        # 速度分: 2000ms以内满分，超过5000ms得0分
        speed_score = max(0, min(20, 20 * (1 - (self.avg_latency_ms - 2000) / 3000)))
        # 长度分: 150-500字之间最好
        len_score = min(20, self.avg_answer_len / 25) if self.avg_answer_len < 500 else max(0, 20 - (self.avg_answer_len - 500) / 100)
        return citation_score + speed_score + len_score


# ─── Ollama 调用 ─────────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context_snippets: list[str]) -> str:
    lines = [
        "You are a precise enterprise knowledge assistant specializing in renewable energy.",
        "Answer ONLY from the provided sources. Cite every fact with [1], [2], etc.",
        "Use the same language as the question.",
        "",
        f"Question: {question}",
        "",
        "Sources:",
    ]
    for i, snippet in enumerate(context_snippets, 1):
        lines.append(f"[{i}] {snippet[:400]}")
    lines.append("\nAnswer (cite sources inline):")
    return "\n".join(lines)


def get_dummy_context(question: str) -> list[str]:
    """
    真实场景应调用 RetrievalService.retrieve()。
    这里用占位上下文，专注测试 LLM 的回答能力和引用遵从性。
    """
    return [
        "Tesla 2023 Annual Report: Total revenues were $96.77 billion, an increase of 19% YoY. "
        "Automotive revenues: $82.42B. Energy generation and storage: $6.04B. Services: $8.32B.",
        "Tesla Q4 2023: Net income $7.93B. Operating margin 8.2%. Deliveries: 1.81M vehicles total in 2023.",
        "宁德时代2023年报：公司营业收入4009亿元，同比下降9.66%。动力电池系统营收2946亿元，储能电池系统营收599亿元。",
    ]


def call_ollama(model: str, prompt: str, base_url: str, timeout: int = 90) -> tuple[str, float]:
    """Returns (answer_text, latency_ms). Raises on error."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    latency_ms = (time.perf_counter() - t0) * 1000
    return str(data.get("response", "")).strip(), latency_ms


def is_model_available(model: str, base_url: str) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        model_short = model.split(":")[0]
        return any(model_short in n for n in names)
    except Exception:
        return False


def has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text))


# ─── 测试逻辑 ─────────────────────────────────────────────────────────────────

def benchmark_model(
    model: str,
    questions: list[str],
    base_url: str,
) -> ModelSummary:
    results: list[ModelResult] = []
    errors = 0

    for q in questions:
        context = get_dummy_context(q)
        prompt = build_rag_prompt(q, context)
        try:
            answer, latency = call_ollama(model, prompt, base_url)
            results.append(ModelResult(
                model=model,
                question=q,
                answer=answer,
                latency_ms=latency,
                has_citation=has_citation(answer),
                answer_length=len(answer),
            ))
        except Exception as exc:
            errors += 1
            results.append(ModelResult(
                model=model,
                question=q,
                answer="",
                latency_ms=0,
                has_citation=False,
                answer_length=0,
                error=str(exc),
            ))

    valid = [r for r in results if not r.error]
    return ModelSummary(
        model=model,
        avg_latency_ms=sum(r.latency_ms for r in valid) / max(1, len(valid)),
        citation_rate=sum(1 for r in valid if r.has_citation) / max(1, len(valid)),
        avg_answer_len=sum(r.answer_length for r in valid) / max(1, len(valid)),
        error_rate=errors / max(1, len(results)),
        results=results,
    )


# ─── 输出 ─────────────────────────────────────────────────────────────────────

def print_header() -> None:
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  Ollama 模型对比基准测试{RESET}")
    print(f"{BOLD}{'='*72}{RESET}\n")


def print_model_progress(model: str, idx: int, total: int, q_idx: int, q_total: int) -> None:
    bar_len = 20
    filled = int(bar_len * q_idx / max(1, q_total))
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] 模型 {idx}/{total}: {model} — 问题 {q_idx}/{q_total}", end="", flush=True)


def color_score(score: float) -> str:
    if score >= 70:
        return f"{GREEN}{score:.1f}{RESET}"
    elif score >= 50:
        return f"{YELLOW}{score:.1f}{RESET}"
    else:
        return f"{RED}{score:.1f}{RESET}"


def print_results_table(summaries: list[ModelSummary]) -> None:
    summaries_sorted = sorted(summaries, key=lambda s: s.score, reverse=True)

    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  排名结果{RESET}")
    print(f"{BOLD}{'─'*72}{RESET}")
    print(f"  {'排名':<4} {'模型':<28} {'综合分':<8} {'引用率':<8} {'延迟(ms)':<10} {'答案长':<8} {'错误率'}")
    print(f"  {'─'*4} {'─'*28} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*6}")

    for rank, s in enumerate(summaries_sorted, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"  {rank}."))
        cite_color = GREEN if s.citation_rate >= 0.8 else (YELLOW if s.citation_rate >= 0.5 else RED)
        lat_color = GREEN if s.avg_latency_ms < 2000 else (YELLOW if s.avg_latency_ms < 4000 else RED)
        err_color = GREEN if s.error_rate == 0 else RED
        print(
            f"  {medal:<5} {s.model:<28} "
            f"{color_score(s.score):<17} "
            f"{cite_color}{s.citation_rate:.0%}{RESET:<14} "
            f"{lat_color}{s.avg_latency_ms:.0f}{RESET:<18} "
            f"{s.avg_answer_len:.0f}字{'':<5} "
            f"{err_color}{s.error_rate:.0%}{RESET}"
        )

    # 推荐
    best = summaries_sorted[0]
    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  推荐模型: {GREEN}{best.model}{RESET}")
    print(f"  综合分 {best.score:.1f} | 引用率 {best.citation_rate:.0%} | 平均延迟 {best.avg_latency_ms:.0f}ms")
    print(f"\n  在 .env 中设置:")
    print(f"  {CYAN}OLLAMA_MODEL={best.model}{RESET}")


def print_answer_samples(summaries: list[ModelSummary], n_samples: int = 2) -> None:
    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  答案样本（第一个测试问题）{RESET}")
    print(f"{BOLD}{'─'*72}{RESET}")
    for s in sorted(summaries, key=lambda x: x.score, reverse=True):
        if not s.results:
            continue
        r = s.results[0]
        print(f"\n  {BOLD}[{s.model}]{RESET}  延迟: {r.latency_ms:.0f}ms  引用: {'✓' if r.has_citation else '✗'}")
        preview = r.answer[:300].replace("\n", " ")
        print(f"  {DIM}{preview}{'...' if len(r.answer) > 300 else ''}{RESET}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple Ollama models for RAG answer quality")
    parser.add_argument(
        "--models",
        default="",
        help="逗号分隔的模型列表。不填则测试所有已下载模型。",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=5,
        help="测试问题数量 (1-7, 越少越快)，默认5",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="显示各模型的答案样本",
    )
    args = parser.parse_args()

    base_url = args.ollama_url
    questions = TEST_QUESTIONS[:max(1, min(7, args.questions))]

    print_header()

    # 确定要测试的模型
    if args.models:
        models_to_test = [m.strip() for m in args.models.split(",")]
    else:
        # 自动检测已下载的模型
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            downloaded = [m["name"] for m in data.get("models", [])]
        except Exception:
            print(f"  {RED}✗ 无法连接 Ollama ({base_url}){RESET}")
            print(f"    请先运行: ollama serve")
            sys.exit(1)

        if not downloaded:
            print(f"  {YELLOW}⚠ 没有已下载的模型{RESET}")
            print(f"\n  推荐先 pull 这几个（从小到大，根据你的显存选择）:\n")
            for m in CANDIDATE_MODELS:
                print(f"    ollama pull {m}")
            sys.exit(0)

        print(f"  已下载模型: {downloaded}")
        # 过滤出候选列表中的，加上所有已下载的
        known = {m.split(":")[0] for m in CANDIDATE_MODELS}
        models_to_test = downloaded  # test all downloaded ones

    print(f"  测试模型数: {len(models_to_test)}")
    print(f"  测试问题数: {len(questions)}")
    print(f"  Ollama URL: {base_url}\n")

    summaries: list[ModelSummary] = []
    for idx, model in enumerate(models_to_test, 1):
        if not is_model_available(model, base_url):
            print(f"\n  {YELLOW}⚠ 跳过 {model}（未下载，运行: ollama pull {model}）{RESET}")
            continue

        print(f"\n  测试中: {BOLD}{model}{RESET}")
        results_for_model: list[ModelResult] = []
        errors = 0

        for q_idx, question in enumerate(questions, 1):
            print_model_progress(model, idx, len(models_to_test), q_idx, len(questions))
            context = get_dummy_context(question)
            prompt = build_rag_prompt(question, context)
            try:
                answer, latency = call_ollama(model, prompt, base_url, timeout=120)
                results_for_model.append(ModelResult(
                    model=model,
                    question=question,
                    answer=answer,
                    latency_ms=latency,
                    has_citation=has_citation(answer),
                    answer_length=len(answer),
                ))
            except Exception as exc:
                errors += 1
                results_for_model.append(ModelResult(
                    model=model, question=question, answer="",
                    latency_ms=0, has_citation=False, answer_length=0, error=str(exc),
                ))

        print()  # newline after progress bar
        valid = [r for r in results_for_model if not r.error]
        summary = ModelSummary(
            model=model,
            avg_latency_ms=sum(r.latency_ms for r in valid) / max(1, len(valid)),
            citation_rate=sum(1 for r in valid if r.has_citation) / max(1, len(valid)),
            avg_answer_len=sum(r.answer_length for r in valid) / max(1, len(valid)),
            error_rate=errors / max(1, len(results_for_model)),
            results=results_for_model,
        )
        summaries.append(summary)
        print(f"    引用率: {summary.citation_rate:.0%}  延迟: {summary.avg_latency_ms:.0f}ms  综合分: {summary.score:.1f}")

    if not summaries:
        print(f"\n  {RED}没有可测试的模型。{RESET}")
        print(f"  请先 pull: ollama pull qwen2.5:7b-instruct")
        return

    print_results_table(summaries)
    if args.show_samples:
        print_answer_samples(summaries)

    # 写结果到文件
    output_path = ROOT / "scripts" / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "model": s.model,
                "score": round(s.score, 2),
                "citation_rate": round(s.citation_rate, 3),
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "avg_answer_len": round(s.avg_answer_len, 1),
                "error_rate": round(s.error_rate, 3),
            }
            for s in sorted(summaries, key=lambda x: x.score, reverse=True)
        ], f, ensure_ascii=False, indent=2)
    print(f"\n  {CYAN}详细结果已保存: scripts/benchmark_results.json{RESET}\n")


if __name__ == "__main__":
    main()
