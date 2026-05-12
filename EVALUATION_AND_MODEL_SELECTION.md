# RAG 系统评估与模型选择方案

## 📊 第一阶段：完整系统评估

你的系统有三层评估指标：

### 1️⃣ 检索质量（Retrieval Score）
```
衡量：BM25 + Dense 混合检索的准确性
范围：0.0 - 1.0
计算：考虑排序折扣的加权得分
├─ 精确匹配chunk (rank_discount * 1.0)
├─ 同文档同页 (rank_discount * 0.7)
├─ 同文档同章节 (rank_discount * 0.5)
└─ 同文档 (rank_discount * 0.2)

目标：> 0.65
```

### 2️⃣ 答案质量（Answer Score）
```
衡量：LLM生成答案与标准答案的相似度
范围：0.0 - 1.0
计算：Token重叠度 (Jaccard相似度)
公式：overlap / sqrt(pred_tokens * gold_tokens)

目标：> 0.30
```

### 3️⃣ 引用准确性（Citation Score）
```
衡量：答案中的引用是否来自正确的来源
范围：0.0 - 1.0
计算：精确引用匹配 vs 源文档匹配
├─ 精确引用匹配：100%
├─ 源文档标题匹配：70%
└─ 无引用或错误：0%

目标：> 0.50
```

### 4️⃣ 整体通过率（Pass Rate）
```
通过标准：同时满足所有三个条件
- retrieval_score > 0.0
- citation_score >= 0.5
- answer_score >= 0.15

目标：> 70%
```

---

## 🚀 执行评估的步骤

### Step 1: 确保测试数据集存在

```bash
# 查看现有评估集
python -c "
from rag_assistant.core import Database
db = Database()
eval_sets = db.fetch_all('SELECT name, COUNT(*) as case_count FROM eval_cases GROUP BY eval_set_id, name')
for row in eval_sets:
    print(f'{row[0]}: {row[1]} cases')
"
```

**预期输出**:
```
energy_starter_v1: 20 cases
```

### Step 2: 运行基线评估

```bash
# 使用当前模型（Qwen2.5）
python scripts/eval_runner.py --eval-set energy_starter_v1 --top-k 6 --json
```

**输出格式**:
```json
{
  "eval_set": "energy_starter_v1",
  "run_id": "uuid-xxx",
  "model_name": "qwen2.5:7b-instruct",
  "count": 20,
  "retrieval_mean": 0.6234,
  "answer_mean": 0.3456,
  "citation_mean": 0.7123,
  "pass_rate": 0.75,
  "samples": [...]
}
```

### Step 3: 记录和分析结果

创建评估记录表：

| 指标 | 目标 | 当前 | 评价 | 瓶颈 |
|------|------|------|------|------|
| **Retrieval Score** | > 0.65 | ___ | ☐ 达到 ☐ 未达到 | ☐ 检索 ☐ 排序 |
| **Answer Score** | > 0.30 | ___ | ☐ 达到 ☐ 未达到 | ☐ LLM能力 ☐ 上下文 |
| **Citation Score** | > 0.50 | ___ | ☐ 达到 ☐ 未达到 | ☐ 引用机制 ☐ 来源识别 |
| **Pass Rate** | > 70% | ___ | ☐ 达到 ☐ 未达到 | 综合问题 |

### Step 4: 优化建议

**如果 Retrieval Score 低**:
```bash
# 尝试调优重排序参数
python scripts/tune_rerank_candidates.py \
  --eval-set energy_starter_v1 \
  --candidates 50,75,100,120 \
  --backend bge \
  --recall-weight 0.7
```

**如果 Answer Score 低**:
- 可能需要升级LLM模型
- 或优化提示词
- 考虑使用更强的模型（Deepseek）

**如果 Citation Score 低**:
- 检查上下文长度设置
- 优化引用提取逻辑
- 确保源文档元数据准确

---

## 🤖 关于 Deepseek 的分析

### Deepseek vs Qwen2.5 对比

| 维度 | Deepseek | Qwen2.5 | 建议 |
|------|----------|---------|------|
| **中文理解能力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Deepseek 稍优 |
| **推理能力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Deepseek 明显优 |
| **模型大小** | 67B, 671B | 7B, 32B | Qwen更轻量 |
| **推理成本** | 高（大模型） | 低（7B) | Qwen更经济 |
| **部署难度** | 高（需GPU） | 中等 | Qwen更易部署 |
| **中文特定任务** | 优（微调过） | 好 | Deepseek优 |
| **实时性** | ⚠️ 慢 | ✅ 快 | Qwen更快 |

### 推荐方案

#### ✅ 方案 1: 保持 Qwen2.5（推荐）**【推荐用于生产】**

**优点**:
- ✅ 已部署和优化过
- ✅ 推理速度快（适合实时查询）
- ✅ 内存占用低
- ✅ 足够的中文能力
- ✅ 不需要重新调参

**劣势**:
- 推理能力略弱于Deepseek

**适用场景**:
- 已经满足业务需求
- 关注延迟和成本
- 单纯需要答案生成

#### ⭐ 方案 2: 使用 Deepseek 作为重排序模型（最推荐）**【混合方案】**

```
Query → 向量检索（BM25+Dense） → 候选K=50 → 
  → Deepseek重排序 → Top-6 → Qwen2.5生成答案
```

**优点**:
- ✅ Deepseek 强大的理解能力用于重排序
- ✅ Qwen2.5 快速生成答案
- ✅ 取长补短，性能最优
- ✅ 检索质量大幅提升

**劣势**:
- 延迟略增（+500ms左右）
- 需要调用两个模型

**实现方式**:
```python
# 在core.py中实现
def rerank_with_deepseek(chunks: list, query: str) -> list:
    # 使用Deepseek的cross-encoder进行重排序
    scores = deepseek_rerank(chunks, query)
    return sorted(chunks, key=lambda x: scores[x['id']], reverse=True)
```

#### 🔥 方案 3: 完全切换到 Deepseek（激进方案）**【不推荐】**

```
Query → Deepseek (不依赖RAG，直接生成)
```

**优点**:
- Deepseek的理解能力最强
- 可能减少检索依赖

**劣势**:
- ❌ 推理时间长（20-30秒）
- ❌ 成本高
- ❌ 可能产生幻觉
- ❌ 企业文档专有信息可能不在训练数据中

**不推荐原因**:
- 违反RAG设计初衷
- 企业文档需要精确检索
- Deepseek可能编造不存在的内容

---

## 📈 评估实验计划

### 实验设计

如果要验证 Deepseek 的效果：

```bash
# 实验 1: 基线（当前Qwen2.5）
python scripts/eval_runner.py --eval-set energy_starter_v1

# 实验 2: 优化检索（调优重排序）
python scripts/tune_rerank_candidates.py \
  --eval-set energy_starter_v1 \
  --candidates 50,75,100,120 \
  --backend bge

# 实验 3: 尝试Deepseek重排序
# （需要实现Deepseek重排序模块）
RERANK_MODEL=deepseek python scripts/eval_runner.py --eval-set energy_starter_v1
```

### 记录模板

```json
{
  "timestamp": "2026-05-08T14:30:00Z",
  "experiment": "baseline_qwen2.5",
  "model": "qwen2.5:7b-instruct",
  "eval_set": "energy_starter_v1",
  "retrieval_mean": 0.6234,
  "answer_mean": 0.3456,
  "citation_mean": 0.7123,
  "pass_rate": 0.75,
  "latency_p50_ms": 850,
  "latency_p99_ms": 2400,
  "notes": "基线性能，可接受"
}
```

---

## 🎯 最终建议

### 对你的项目的建议：**方案 2（混合方案）**最优

```
现状：Qwen2.5 7B 足够
目标：提升到 80%+ pass rate

实施步骤：
1. ✅ 运行基线评估（Qwen2.5）→ 记录分数
2. ⚙️ 优化检索参数（tune_rerank_candidates）
3. 🔄 如果检索score还是低于0.65，考虑用Deepseek重排序
4. 📊 A/B测试对比效果
```

### 为什么这样选择？

1. **成本平衡**: Qwen2.5的推理成本低，Deepseek只用于关键的重排序
2. **性能最优**: 取长补短，Deepseek的理解+Qwen的速度
3. **风险最低**: 如果Deepseek重排序效果不理想，可以快速回滚
4. **可扩展性**: 后续可以根据实际效果调整

### 何时切换到 Deepseek？

✅ **值得切换的信号**:
- 当前pass_rate < 60%
- Retrieval score 无法通过参数优化提升
- 需要处理复杂推理的问题

❌ **不值得切换的信号**:
- 当前系统already working（pass_rate > 70%）
- 延迟要求 < 500ms
- 部署成本限制紧张

---

## 📝 行动清单

### 本周要做的：

- [ ] **今天**: 运行完整系统评估（eval_runner.py）
- [ ] **记录**: 填写评估表
- [ ] **分析**: 识别主要瓶颈（检索/答案/引用）
- [ ] **尝试**: 运行 tune_rerank_candidates.py 优化参数
- [ ] **决策**: 基于结果决定是否需要Deepseek

### 如果需要Deepseek：

- [ ] 研究 Deepseek API 集成方案
- [ ] 实现重排序模块
- [ ] 测试Deepseek方案性能
- [ ] 对比成本和效果

---

## 💡 关键洞察

> **不是所有问题都需要更强的LLM**
>
> 你的系统三层问题可能不同：
> - 检索质量低 → 改进检索/重排序（用Deepseek）
> - 答案质量低 → 升级LLM或优化提示词
> - 引用不准 → 改进引用提取逻辑
>
> 先诊断问题，再有针对性的优化，不要盲目升级模型！

---

**建议首先运行评估，看具体的性能数据，然后我们再决定是否需要Deepseek。** 

需要我帮你写一个完整的评估脚本吗？
