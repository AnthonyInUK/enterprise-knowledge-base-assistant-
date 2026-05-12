from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import os
import time
from typing import Any

from rag_assistant.core import Database

DEFAULT_MODEL = "BAAI/bge-m3"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _detect_device() -> str:
    """Auto-detect best available device: MPS (Apple Silicon) > CUDA > CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_model(model_name: str, device: str | None = None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required. Install with: python3 -m pip install -e '.[rerank]'") from exc
    resolved_device = device or _detect_device()
    print(f"[INFO] 使用设备: {resolved_device}")
    return SentenceTransformer(model_name, device=resolved_device)


def _format_progress_bar(completed: int, total: int, elapsed: float) -> str:
    """格式化进度条，显示进度、速率和预计完成时间"""
    if total == 0:
        return ""

    percent = completed / total
    bar_length = 40
    filled = int(bar_length * percent)
    bar = "█" * filled + "░" * (bar_length - filled)

    # 计算速率和预计时间
    if elapsed > 0:
        rate = completed / elapsed
        remaining = total - completed
        remaining_time = remaining / rate if rate > 0 else 0
    else:
        rate = 0
        remaining_time = 0

    # 格式化输出
    percent_str = f"{percent*100:.1f}%"
    rate_str = f"{rate:.0f} chunks/s"
    elapsed_str = f"{int(elapsed)}s"

    if remaining_time > 0:
        remaining_str = f"ETA: {int(remaining_time)}s"
    else:
        remaining_str = "完成"

    return f"[{bar}] {percent_str} | 已完成: {completed}/{total} | 速率: {rate_str} | 耗时: {elapsed_str} | {remaining_str}"


def embed_chunks(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    limit: int | None = None,
    force: bool = False,
    verbose: bool = True,
    device: str | None = None,
    sleep_between_batches: float = 0.0,
) -> dict[str, Any]:
    """
    生成chunk的embedding并存储到PostgreSQL

    Args:
        model_name: 使用的embedding模型名称
        batch_size: 批处理大小（内存紧张时用4-8）
        limit: 最多处理的chunk数（None表示全部）
        force: 是否强制重新生成已有的embedding
        verbose: 是否输出进度信息
        device: 强制指定设备 (cpu/cuda/mps)，None时自动检测
        sleep_between_batches: 每批次后休眠秒数，防止系统过热/卡死

    Returns:
        包含统计信息的字典
    """
    db = Database()

    # 构建查询条件
    where = ""
    params: list[Any] = []
    if force:
        where = "WHERE COALESCE(c.text, '') <> ''"
    else:
        where = """
        WHERE COALESCE(c.text, '') <> ''
          AND NOT EXISTS (
              SELECT 1 FROM embeddings e
              WHERE e.chunk_id = c.id AND e.embedding_model = %s
          )
        """
        params.append(model_name)

    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT %s"
        params.append(limit)

    # 查询需要embedding的chunks
    if verbose:
        print(f"[INFO] 查询需要embedding的chunks...")

    rows = db.fetch_all(
        f"""
        SELECT
            c.id::text AS chunk_id,
            COALESCE(d.title, '') AS document_title,
            COALESCE(c.section_title, '') AS section_title,
            c.text
        FROM chunks c
        JOIN document_versions dv ON dv.id = c.document_version_id
        JOIN documents d ON d.id = dv.document_id
        {where}
        ORDER BY c.created_at, c.id
        {limit_sql}
        """,
        tuple(params),
    )

    if not rows:
        return {"model": model_name, "selected": 0, "embedded": 0, "dimension": None}

    total_chunks = len(rows)
    if verbose:
        print(f"[INFO] 需要处理 {total_chunks} 个chunks")
        print(f"[INFO] 加载embedding模型 {model_name}...")

    model = _load_model(model_name, device=device)

    if verbose:
        print(f"[INFO] 开始生成embedding，批大小={batch_size}")

    embedded = 0
    skipped = 0
    dimension: int | None = None
    start_time = time.perf_counter()

    for batch_idx, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]

        # 准备文本
        texts = []
        for row in batch:
            body = " ".join(row["text"].split())[:1800]
            texts.append("\n".join([row["document_title"], row["section_title"], body]))

        # 生成embedding（批次级错误恢复：跳过失败批次而非崩溃整个任务）
        try:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            if verbose:
                print(f"\n[WARN] 批次 {batch_idx} 编码失败，跳过 {len(batch)} 个chunk: {exc}")
            skipped += len(batch)
            continue

        # 存储到数据库（单条失败不影响其他）
        for row, vector in zip(batch, vectors):
            values = [float(item) for item in vector.tolist()]
            dimension = len(values)

            if dimension != 1024:
                if verbose:
                    print(f"\n[WARN] chunk {row['chunk_id']} 维度 {dimension} != 1024，跳过")
                skipped += 1
                continue

            try:
                db.execute(
                    """
                    INSERT INTO embeddings (chunk_id, embedding_model, embedding_version, dimension, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_version = EXCLUDED.embedding_version,
                        dimension = EXCLUDED.dimension,
                        embedding = EXCLUDED.embedding,
                        created_at = now()
                    """,
                    (row["chunk_id"], model_name, "v1", dimension, _vector_literal(values)),
                )
                embedded += 1
            except Exception as exc:
                if verbose:
                    print(f"\n[WARN] 写入 chunk {row['chunk_id']} 失败: {exc}")
                skipped += 1

        # 批次间休眠（防止系统卡死）
        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

        # 输出进度
        if verbose:
            elapsed = time.perf_counter() - start_time
            progress = _format_progress_bar(embedded, total_chunks, elapsed)
            print(f"\r{progress}", end="", flush=True)

    elapsed_total = time.perf_counter() - start_time
    if verbose:
        print()  # 换行
        print(f"[INFO] ✅ 完成！耗时: {elapsed_total:.1f}s")
        print(f"[INFO] 总共生成: {embedded} 个embedding，跳过: {skipped}")
        print(f"[INFO] 平均速率: {embedded/elapsed_total:.1f} chunks/s" if elapsed_total > 0 else "")

    return {
        "model": model_name,
        "selected": len(rows),
        "embedded": embedded,
        "skipped": skipped,
        "dimension": dimension,
        "elapsed_seconds": elapsed_total,
        "rate_chunks_per_second": embedded / elapsed_total if elapsed_total > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks with a real sentence-transformers model and store pgvector rows."
    )
    parser.add_argument("--model", default=os.getenv("RAG_EMBED_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed chunks even if this model already exists.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="强制指定设备 (cpu/cuda/mps)。不填则自动检测 (Apple Silicon 自动选 mps)。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每个批次后休眠秒数（内存紧张时用 0.5~2.0，给系统喘息空间）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    result = embed_chunks(
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        verbose=not args.quiet,
        device=args.device,
        sleep_between_batches=args.sleep,
    )

    # 输出最终统计
    print()
    print("=" * 60)
    print("Embedding 生成完成")
    print("=" * 60)
    print(f"模型: {result['model']}")
    print(f"处理chunk数: {result['selected']}")
    print(f"成功生成: {result['embedded']}")
    print(f"跳过(错误): {result.get('skipped', 0)}")
    print(f"向量维度: {result['dimension']}")
    if result.get('elapsed_seconds') is not None:
        print(f"总耗时: {result['elapsed_seconds']:.2f}s")
        print(f"平均速率: {result['rate_chunks_per_second']:.1f} chunks/s")
    else:
        print("总耗时: N/A (没有需要处理的chunks)")
    print("=" * 60)


if __name__ == "__main__":
    main()
