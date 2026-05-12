#!/usr/bin/env python3
"""
实时监控 embedding 进度的脚本
在另一个终端运行: python scripts/monitor_embedding_progress.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
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


def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_rate(rate: float) -> str:
    """格式化速率"""
    return f"{rate:.0f} chunks/s"


def main() -> None:
    load_env_file(ROOT / ".env")
    if "DATABASE_URL" not in os.environ:
        load_env_file(ROOT / ".env.example")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is missing")

    model_name = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")

    print(f"🔍 监控 Embedding 进度 (Model: {model_name})")
    print("=" * 80)

    start_time = time.time()
    last_count = 0
    last_check = start_time
    last_rate = 0.0

    try:
        while True:
            with psycopg.connect(database_url) as conn:
                with conn.cursor() as cur:
                    # 总chunks数
                    cur.execute("SELECT COUNT(*) FROM chunks")
                    total_chunks = cur.fetchone()[0]

                    # 已嵌入chunks数
                    cur.execute(
                        """
                        SELECT COUNT(*) FROM chunks c
                        WHERE EXISTS (
                            SELECT 1 FROM embeddings e
                            WHERE e.chunk_id = c.id
                            AND e.embedding_model = %s
                        )
                        """,
                        (model_name,),
                    )
                    embedded_chunks = cur.fetchone()[0]

                    # 待嵌入chunks数
                    pending = total_chunks - embedded_chunks

                    # 计算速率
                    now = time.time()
                    elapsed_since_last = now - last_check
                    if elapsed_since_last >= 5:  # 每5秒更新一次速率
                        new_count = embedded_chunks
                        chunks_since_last = new_count - last_count
                        last_rate = chunks_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
                        last_count = new_count
                        last_check = now

                    # 计算剩余时间
                    elapsed_total = now - start_time
                    if last_rate > 0 and pending > 0:
                        remaining_seconds = pending / last_rate
                        remaining_str = format_duration(remaining_seconds)
                        eta_time = datetime.now() + timedelta(seconds=remaining_seconds)
                        eta_str = eta_time.strftime("%H:%M:%S")
                    else:
                        remaining_str = "计算中..."
                        eta_str = "--:--:--"

                    # 进度条
                    progress_pct = (
                        (embedded_chunks / total_chunks * 100)
                        if total_chunks > 0
                        else 0
                    )
                    bar_length = 40
                    filled = int(bar_length * embedded_chunks / max(1, total_chunks))
                    bar = "█" * filled + "░" * (bar_length - filled)

                    # 输出进度信息
                    print(
                        f"\r[{bar}] {progress_pct:5.1f}% | "
                        f"已完成: {embedded_chunks:6d}/{total_chunks:6d} | "
                        f"待处理: {pending:6d} | "
                        f"速率: {format_rate(last_rate)} | "
                        f"耗时: {format_duration(elapsed_total)} | "
                        f"预计完成: {eta_str}",
                        end="",
                        flush=True,
                    )

                    # 检查是否完成
                    if pending == 0:
                        print("\n")
                        print("=" * 80)
                        print(f"✅ 完成！总耗时: {format_duration(elapsed_total)}")
                        print(f"   总向量数: {embedded_chunks:,}")
                        print(f"   平均速率: {format_rate(embedded_chunks / elapsed_total)}")
                        break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 80)
        print("⏸️  监控已停止")
        print(f"   已完成: {embedded_chunks:,} chunks")
        print(f"   耗时: {format_duration(elapsed_total)}")


if __name__ == "__main__":
    main()
