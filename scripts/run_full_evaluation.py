#!/usr/bin/env python3
"""
完整系统评估脚本
运行所有质量指标测试，生成可对比的报告
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_assistant.core import Database
from rag_assistant.eval_runner import run_eval
from rag_assistant.tune_rerank_candidates import tune_candidates

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "eval_reports"
REPORTS_DIR.mkdir(exist_ok=True)


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


def check_eval_set_exists(eval_set_name: str) -> bool:
    """检查评估集是否存在"""
    db = Database()
    result = db.fetch_one("SELECT id FROM eval_sets WHERE name = %s", (eval_set_name,))
    return result is not None


def print_section(title: str) -> None:
    """打印分段标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """打印评估指标"""
    print(f"📊 {label}")
    print("-" * 40)

    # 显示主要指标
    metrics_to_show = [
        ("检索质量 (Retrieval)", "retrieval_mean", 0.65),
        ("答案质量 (Answer)", "answer_mean", 0.30),
        ("引用准确 (Citation)", "citation_mean", 0.50),
        ("通过率 (Pass Rate)", "pass_rate", 0.70),
    ]

    for name, key, target in metrics_to_show:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                # 判断是否达到目标
                status = "✅" if value >= target else "⚠️"
                print(f"  {status} {name:15} {value:.4f} (目标: {target:.2f})")
            else:
                print(f"  📌 {name:15} {value}")

    if "count" in metrics:
        print(f"  📝 测试样本数: {metrics['count']}")


def generate_report(baseline: dict[str, Any], timestamp: str) -> str:
    """生成评估报告"""
    report = {
        "timestamp": timestamp,
        "baseline": baseline,
        "analysis": analyze_results(baseline),
        "recommendations": get_recommendations(baseline),
    }

    report_path = REPORTS_DIR / f"eval_report_{timestamp.replace(':', '-')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return str(report_path)


def analyze_results(baseline: dict[str, Any]) -> dict[str, Any]:
    """分析评估结果，识别瓶颈"""
    analysis = {
        "bottlenecks": [],
        "strengths": [],
        "overall_status": "unknown",
    }

    retrieval = baseline.get("retrieval_mean", 0)
    answer = baseline.get("answer_mean", 0)
    citation = baseline.get("citation_mean", 0)
    pass_rate = baseline.get("pass_rate", 0)

    # 识别瓶颈
    if retrieval < 0.65:
        analysis["bottlenecks"].append({
            "component": "检索质量",
            "score": retrieval,
            "issue": "检索到的chunks可能不相关或排序不好",
            "suggestions": [
                "运行 tune_rerank_candidates.py 优化重排序参数",
                "考虑使用Deepseek进行重排序",
                "检查embedding模型是否合适",
            ],
        })
    else:
        analysis["strengths"].append("检索质量良好")

    if answer < 0.30:
        analysis["bottlenecks"].append({
            "component": "答案质量",
            "score": answer,
            "issue": "LLM生成的答案与标准答案差异大",
            "suggestions": [
                "尝试更强的LLM模型（如Deepseek）",
                "优化提示词工程",
                "检查上下文长度和质量",
            ],
        })
    else:
        analysis["strengths"].append("答案质量良好")

    if citation < 0.50:
        analysis["bottlenecks"].append({
            "component": "引用准确性",
            "score": citation,
            "issue": "答案中的引用源不准确",
            "suggestions": [
                "检查源文档元数据是否准确",
                "优化引用提取逻辑",
                "手动检查几个失败案例",
            ],
        })
    else:
        analysis["strengths"].append("引用准确性良好")

    # 整体评价
    if pass_rate >= 0.80:
        analysis["overall_status"] = "优秀 🌟"
    elif pass_rate >= 0.70:
        analysis["overall_status"] = "良好 ✅"
    elif pass_rate >= 0.60:
        analysis["overall_status"] = "可接受 ⚠️ 建议优化"
    else:
        analysis["overall_status"] = "需要改进 🔴"

    return analysis


def get_recommendations(baseline: dict[str, Any]) -> list[str]:
    """根据结果给出建议"""
    recommendations = []
    pass_rate = baseline.get("pass_rate", 0)
    retrieval = baseline.get("retrieval_mean", 0)
    answer = baseline.get("answer_mean", 0)

    if pass_rate >= 0.75:
        recommendations.append("✅ 当前系统性能优异，建议直接上线或保持现状")
    elif pass_rate >= 0.70:
        recommendations.append("⚙️ 性能可接受，可进行小范围优化")
        if retrieval < 0.65:
            recommendations.append("  → 优先优化检索参数")
    else:
        if retrieval < 0.65 and answer < 0.30:
            recommendations.append("🔧 建议采用混合方案：")
            recommendations.append("  1. 先优化检索参数（tune_rerank_candidates）")
            recommendations.append("  2. 如果检索仍低于0.65，使用Deepseek重排序")
            recommendations.append("  3. 如果答案质量低于0.30，考虑升级LLM")
        elif retrieval < 0.65:
            recommendations.append("🔄 建议使用Deepseek进行重排序")
        else:
            recommendations.append("🤖 建议尝试更强的LLM模型")

    return recommendations


def main() -> None:
    load_env_file(ROOT / ".env")
    if "DATABASE_URL" not in os.environ:
        load_env_file(ROOT / ".env.example")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print_section("🚀 完整RAG系统评估")
    print(f"开始时间: {timestamp}")
    print(f"评估集: energy_starter_v1")

    # 检查评估集是否存在
    if not check_eval_set_exists("energy_starter_v1"):
        print("❌ 评估集 'energy_starter_v1' 不存在")
        print("请先运行: python scripts/eval_seed.py")
        sys.exit(1)

    # 运行基线评估
    print_section("Step 1: 基线评估（当前模型）")
    print("运行中... 这可能需要几分钟\n")

    try:
        baseline = run_eval(eval_set_name="energy_starter_v1", top_k=6)
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        sys.exit(1)

    print_metrics("当前模型性能", baseline)

    # 优化检索参数（可选）
    print_section("Step 2: 优化检索参数（可选）")
    tune_rerank = input("是否运行重排序参数调优？(y/n): ").strip().lower() == 'y'

    if tune_rerank:
        print("调优中... 这将运行多次评估\n")
        try:
            tune_result = tune_candidates(
                eval_set_name="energy_starter_v1",
                candidates=[50, 75, 100, 120],
                top_k=6,
                backend="bge",
                recall_weight=0.7,
            )

            print("🔍 重排序参数调优结果:")
            print("-" * 40)
            for row in tune_result["results"]:
                print(
                    f"  K={row['candidate_k']:3d}: "
                    f"retrieval={row['retrieval_mean']:.4f} "
                    f"pass={row['pass_rate']:.4f} "
                    f"time={row['elapsed_per_case_ms']:.1f}ms"
                )
        except Exception as e:
            print(f"⚠️ 参数调优失败: {e}")

    # 生成报告
    print_section("Step 3: 生成评估报告")
    report_path = generate_report(baseline, timestamp_file)
    print(f"✅ 报告已保存: {report_path}")

    # 显示分析结果
    print_section("📈 结果分析")
    analysis = analyze_results(baseline)

    print(f"📊 整体评价: {analysis['overall_status']}\n")

    if analysis["strengths"]:
        print("💪 优势:")
        for strength in analysis["strengths"]:
            print(f"  ✅ {strength}")

    if analysis["bottlenecks"]:
        print("\n⚠️ 瓶颈:")
        for bottleneck in analysis["bottlenecks"]:
            print(f"  🔴 {bottleneck['component']}: {bottleneck['score']:.4f}")
            print(f"     问题: {bottleneck['issue']}")
            for suggestion in bottleneck['suggestions']:
                print(f"     → {suggestion}")

    # 显示建议
    print("\n💡 建议:")
    recommendations = get_recommendations(baseline)
    for rec in recommendations:
        print(f"  {rec}")

    # 样本展示
    print_section("📝 样本结果")
    print("前3个样本的详细结果:\n")
    for i, sample in enumerate(baseline.get("samples", [])[:3], 1):
        print(f"{i}. {sample['question'][:50]}...")
        print(f"   检索: {sample['retrieval_score']:.3f} "
              f"答案: {sample['answer_score']:.3f} "
              f"引用: {sample['citation_score']:.3f} "
              f"{'✅' if sample['pass_fail'] else '❌'}")

    print_section("✨ 评估完成")
    print(f"Run ID: {baseline.get('run_id')}")
    print(f"报告位置: {report_path}")
    print("\n下一步: ")
    print("1. 检查报告中的分析结果")
    print("2. 根据瓶颈有针对性地优化")
    print("3. 重新运行评估验证效果")


if __name__ == "__main__":
    main()
