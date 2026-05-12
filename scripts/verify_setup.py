"""
verify_setup.py — Phase 1 + Phase 2 快速验证脚本

检查项目:
  1. 数据库连接 + embedding覆盖率
  2. Ollama 连通性 + 模型是否已拉取
  3. Claude API key 是否有效
  4. 端到端查询测试 (quick smoke test)

用法:
    cd enterprise-knowledge-base-assistant
    python scripts/verify_setup.py          # 全量检查
    python scripts/verify_setup.py --quick  # 跳过端到端查询
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
import urllib.request
import urllib.error
import time

from dotenv import load_dotenv

load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

SMOKE_QUERY = "特斯拉2023年营收是多少？"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {CYAN}→{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─'*55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*55}{RESET}")


# ─────────────────────────────────────────────────────────
# 1. Database + Embedding coverage
# ─────────────────────────────────────────────────────────

def check_database() -> bool:
    section("1. 数据库 + Embedding 覆盖率")
    try:
        from rag_assistant.core import Database
        db = Database()

        # total chunks
        row = db.fetch_one("SELECT COUNT(*) AS n FROM chunks")
        total = int(row["n"]) if row else 0

        # real embeddings (BAAI/bge-m3)
        model = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
        row2 = db.fetch_one(
            "SELECT COUNT(*) AS n FROM embeddings WHERE embedding_model = %s",
            (model,),
        )
        real = int(row2["n"]) if row2 else 0

        # placeholder / hash-bootstrap
        row3 = db.fetch_one(
            "SELECT COUNT(*) AS n FROM embeddings WHERE embedding_model = 'hash-bootstrap'",
        )
        placeholder = int(row3["n"]) if row3 else 0

        coverage = real / total * 100 if total else 0

        ok(f"数据库连接正常")
        print(f"     总 chunks:        {total:,}")
        print(f"     真实 embeddings:  {real:,}  ({coverage:.1f}%)")
        print(f"     占位 embeddings:  {placeholder:,}  (hash-bootstrap)")

        pending = total - real
        if coverage >= 99:
            ok(f"Embedding 覆盖率 {coverage:.1f}% — Phase 1 完成！")
        elif coverage >= 50:
            warn(f"覆盖率 {coverage:.1f}%，仍有 {pending:,} 个chunk待处理")
            info(f"运行: python scripts/embed_chunks.py --batch-size 32")
        else:
            fail(f"覆盖率仅 {coverage:.1f}%，需要运行embedding生成")
            info(f"运行: python scripts/embed_chunks.py --batch-size 32")
            info(f"预计耗时: ~{pending / 180 / 60:.0f} 分钟 (Apple Silicon MPS)")

        return True
    except Exception as exc:
        fail(f"数据库连接失败: {exc}")
        info("检查 Docker 是否运行: docker ps | grep postgres")
        info("或检查 .env 中 DATABASE_URL 配置")
        return False


# ─────────────────────────────────────────────────────────
# 2. Ollama
# ─────────────────────────────────────────────────────────

def check_ollama() -> bool:
    section("2. Ollama 本地 LLM")
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    # ping /api/tags
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        available_models = [m["name"] for m in data.get("models", [])]
        ok(f"Ollama 服务正在运行 ({base_url})")
        print(f"     已下载模型: {available_models or '(无)'}")

        # check target model
        model_short = model.split(":")[0]
        matched = [m for m in available_models if model_short in m]
        if matched:
            ok(f"目标模型 {model} 已就绪")
            return True
        else:
            warn(f"目标模型 {model} 未找到")
            info(f"拉取命令: ollama pull {model}")
            return False
    except (urllib.error.URLError, OSError):
        fail(f"Ollama 未运行 ({base_url})")
        info("启动命令: ollama serve")
        info(f"拉取模型: ollama pull {model}")
        return False


# ─────────────────────────────────────────────────────────
# 3. Claude API
# ─────────────────────────────────────────────────────────

def check_claude_api() -> bool:
    section("3. Claude API")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    if not api_key:
        warn("ANTHROPIC_API_KEY 未设置")
        info("在 .env 中添加: ANTHROPIC_API_KEY=sk-ant-xxxxx")
        info("(设置后无需 Ollama，可立即使用 Claude 生成答案)")
        return False

    # test with minimal request
    payload = json.dumps({
        "model": claude_model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode("utf-8")
    req = urllib.request.Request(
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
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        ok(f"Claude API 连通  (模型: {claude_model})")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            fail("ANTHROPIC_API_KEY 无效 (401 Unauthorized)")
        elif exc.code == 429:
            warn("Claude API 速率限制，但 key 有效")
            return True
        else:
            fail(f"Claude API 错误: HTTP {exc.code}")
        return False
    except Exception as exc:
        fail(f"Claude API 连接失败: {exc}")
        return False


# ─────────────────────────────────────────────────────────
# 4. End-to-end smoke test
# ─────────────────────────────────────────────────────────

def check_e2e(skip: bool = False) -> None:
    section("4. 端到端查询测试")
    if skip:
        info("跳过 (--quick 模式)")
        return

    try:
        from rag_assistant.core import RetrievalService
        svc = RetrievalService()

        t0 = time.perf_counter()
        result = svc.answer(SMOKE_QUERY, top_k=4)
        elapsed = (time.perf_counter() - t0) * 1000

        ok(f"查询成功  ({elapsed:.0f} ms)")
        print(f"     问题: {SMOKE_QUERY}")
        print(f"     LLM:  {result.debug.get('llm_backend', '?')}")
        print(f"     答案预览: {result.answer[:200].strip()}...")
        print(f"     检索hits: {result.debug.get('retrieval_count', 0)}")
        print(f"     向量检索: {result.debug.get('vector_search', {})}")
    except Exception as exc:
        fail(f"端到端查询失败: {exc}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────

def print_summary(db_ok: bool, ollama_ok: bool, claude_ok: bool) -> None:
    section("总结 & 推荐下一步")

    llm_ready = ollama_ok or claude_ok

    if db_ok and llm_ready:
        ok("系统就绪！可以运行完整的 RAG 查询")
    else:
        warn("系统未完全就绪，请按下方建议操作")

    print()
    if not db_ok:
        print(f"  {RED}[CRITICAL]{RESET} Docker/Postgres未运行")
        print(f"            docker compose up -d")
    if not claude_ok and not ollama_ok:
        print(f"  {YELLOW}[LLM]{RESET} 请选择至少一个LLM后端:")
        print(f"        选项A (推荐): .env 添加 ANTHROPIC_API_KEY=sk-ant-xxxx")
        print(f"        选项B:        ollama serve && ollama pull qwen2.5:7b-instruct")

    backend = os.getenv("RAG_LLM_BACKEND", "auto")
    embed_model = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
    vector_enabled = os.getenv("RAG_ENABLE_VECTOR", "0")
    rerank_enabled = os.getenv("RAG_ENABLE_RERANK", "0")

    print(f"\n  {CYAN}当前配置{RESET}")
    print(f"    RAG_LLM_BACKEND   = {backend}")
    print(f"    RAG_EMBED_MODEL   = {embed_model}")
    print(f"    RAG_ENABLE_VECTOR = {vector_enabled}")
    print(f"    RAG_ENABLE_RERANK = {rerank_enabled}")
    print()
    print(f"  {CYAN}测试查询{RESET}")
    print(f"    python scripts/answer_question.py '{SMOKE_QUERY}'")
    print(f"    python scripts/answer_question.py --json '{SMOKE_QUERY}'")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 1 + Phase 2 setup")
    parser.add_argument("--quick", action="store_true", help="跳过端到端查询测试")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}  RAG System Setup Verification{RESET}")
    print(f"{BOLD}{'='*55}{RESET}")

    db_ok = check_database()
    ollama_ok = check_ollama()
    claude_ok = check_claude_api()
    check_e2e(skip=args.quick)
    print_summary(db_ok, ollama_ok, claude_ok)


if __name__ == "__main__":
    main()
