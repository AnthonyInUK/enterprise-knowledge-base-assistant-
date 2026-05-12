# Energy Research Agent Interview Notes

> 项目定位：面向新能源行业公开报告/新闻资料的研究 Agent。RAG 负责查证据，direct-extractive 工具负责抽关键数字，React 工作台展示答案、引用、证据 trace 和运行指标。目标不是做一个“能聊天”的 demo，而是做一套可解释、可评测、可观测的行业研究工作流。

## 0. 30 秒项目介绍

我做了一个新能源行业研究 Agent，数据源是中国和海外新能源公司的公开报告、新闻和行业资料。系统包含数据抓取、PDF/HTML 解析、文本清洗、标题+段落 chunking、元数据和页码引用、PostgreSQL/pgvector 存储、BM25/向量混合召回、bge-reranker 重排、带引用答案生成、财报数字直抽、eval set、运行 trace 和 React 研究工作台。

这个项目的重点不是简单调用一个向量数据库，也不是普通聊天机器人，而是做了一个完整的工程闭环：先做 baseline，再用 eval 发现问题，判断问题在召回、排序、gold 数据还是答案抽取，然后逐步引入 query rewrite、reranker、rerank cache、hybrid search、direct-extractive guardrail 和前端 evidence trace，并用指标验证每一步是否真的有提升。

## 1. 数据抓取和入库

### 一句话讲法

我用脚本抓取新能源公司和行业机构的公开资料，统一落到 documents、document_versions、chunks、embeddings、entities、relations 等表里，保证后续 RAG/GraphRAG 都有稳定的数据基础。

### 为什么做

企业知识库的核心不是模型，而是数据资产。如果数据来源、版本、解析质量、页码引用都不可追踪，后面的检索和回答就很难 debug。

### 怎么做

- `documents` 存文档级元数据，比如标题、来源 URL、文件 hash、语言、文档类型。
- `document_versions` 存版本信息，避免同一份文档重跑解析时覆盖历史。
- `chunks` 存章节级和段落级内容，带 `section_title`、`page_start`、`page_end`。
- `embeddings` 存 chunk 向量，早期用 `hash-bootstrap` 占位，后续替换成 `BAAI/bge-m3`。
- `ingestion_jobs` 记录解析方式、OCR 是否触发、质量分数、是否需要人工检查。

### 指标或证据

当前本地库的 embeddings 覆盖情况：

```text
总 chunks:       12,486
BAAI/bge-m3:     12,486
hash-bootstrap:  0
覆盖率:          100%
```

这说明当前评测使用的是完整真实 embedding 覆盖，不再依赖早期的 hash-bootstrap 占位向量。

### 面试官可能问

**Q: 为什么不直接把 PDF 文本塞进向量库？**

A: 因为企业知识库需要可追踪和可维护。直接塞向量库会丢失文档版本、页码、解析质量、权限、引用来源等信息。我的设计是 PostgreSQL 存结构化元数据和 chunk，pgvector 存 embedding，这样检索结果可以回溯到文档、页码和版本。

**Q: 为什么要 document_versions？**

A: 因为解析链会不断优化，比如 chunking、OCR、清洗规则都可能变化。`document_versions` 可以保留不同解析版本，避免重跑时唯一键冲突，也方便比较新版解析是否提升检索质量。

## 2. PDF/OCR 解析和文本清洗

### 一句话讲法

我做了“文本优先，OCR 兜底”的 PDF 解析链，并记录每份文档的解析方式和质量指标。

### 为什么做

企业文档里既有可复制文本 PDF，也有扫描件。如果所有 PDF 都 OCR，成本和时间很高；如果完全不 OCR，扫描件会没有内容。所以我采用文本优先，只有文本抽取太少或质量太差时才触发 OCR。

### 怎么做

- 优先用文本抽取。
- 如果页面文本过少、空页过多、质量分低，再触发 OCR。
- 清理页眉页脚、免责声明、导航噪声。
- 做段落级重复/近重复去重。
- 在 `ingestion_jobs` 记录 `parse_method`、`ocr_used`、`quality_score`、`needs_review`。

### 面试官可能问

**Q: 页眉页脚和免责声明怎么清理？**

A: 页眉页脚通常跨页重复，免责声明也会在不同页面或文档中高频出现。我会统计跨页重复文本、短行重复模式、常见 disclaimer pattern，然后在 chunk 前过滤，避免这些噪声进入 embedding 和检索。

**Q: 为什么要记录 OCR 质量？**

A: 因为 OCR 可能产生乱码或表格错位。如果某些答案质量差，我可以回查这个文档是否走了 OCR、质量分多少、是否需要人工复核。

## 3. Chunking：标题 + 段落 + 双层结构

### 一句话讲法

我把 chunking 从零散切片改成“章节级 chunk + 段落级 chunk”的双层结构，并给段落 chunk 绑定自己的页码范围。

### 为什么做

RAG 里 chunk 太大，答案不聚焦；chunk 太碎，语义不完整。章节级 chunk 适合粗召回，段落级 chunk 适合最终回答。页码范围用于 citation。

### 怎么做

- 章节 chunk：保留大段上下文和标题结构。
- 段落 chunk：作为主要检索和回答单元。
- `parent_chunk_id`：段落挂到章节下面。
- `page_start/page_end`：每个 chunk 自己带页码，引用更准确。

### 面试官可能问

**Q: 为什么不是固定 500 token 一切？**

A: 固定 token 切法简单，但容易切断标题、表格、段落语义。我这里的文档多是年报/行业报告，天然有标题和段落结构，所以优先用“标题 + 段落”级 chunking，更适合 citation 和面试时解释。

## 4. Retrieval Baseline 和 Eval

### 一句话讲法

我先做了一个可测 baseline，然后用 eval 判断问题到底出在召回、排序还是答案生成。

### 为什么做

没有 eval，RAG 优化很容易靠感觉。我的做法是先跑 baseline，再用指标定位瓶颈。

### 核心指标

早期 baseline：

```text
retrieval_mean: 0.3708
answer_mean:    0.1110
citation_mean:  0.7333
pass_rate:      0.2333
```

清洗 eval gold + 召回优化后：

```text
retrieval_mean: 0.4933
answer_mean:    0.2855
citation_mean:  0.8067
pass_rate:      0.6333
```

### 面试官可能问

**Q: strict exact 是什么？**

A: strict exact 指检索结果必须命中 eval 标注的那个 exact chunk。它很严格，适合看精确召回，但如果 gold chunk 标注太碎或有噪声，就会低估系统能力。所以我同时看 exact、同页、同章节、同文档等更公平的 retrieval score。

**Q: 你怎么判断是召回问题还是 rerank 问题？**

A: 我写了 top-50 诊断脚本。如果 gold chunk 连 top-50 都进不来，是召回问题；如果进了 top-50 但没进 top-6，是排序/reranker 问题。

## 5. Eval Gold 清洗

### 一句话讲法

我发现自动生成 eval 的 gold chunk 有噪声，所以写了 `repair_eval_gold.py`，按问题类型在同一文档里重新选择更合理的 gold chunk。

### 为什么做

早期 eval 是自动从随机 paragraph 生成的，有些问题是财务类，但 gold chunk 却是环境风险、坏账准备、应收利息等表格残片。这会误导优化。

### 怎么做

- 判断问题类型：financial、capacity、market、technology、project。
- 在同一文档内按 BM25/业务关键词重新挑更匹配的 chunk。
- 更新 `gold_answer`、`gold_citations`、`gold_chunk_ids`。
- 把旧 gold 记录到 `metadata.gold_repair`，保证可追溯。

### 指标或证据

清洗后：

```text
recall_failure_rate:      0.0667 -> 0.0000
exact_top_candidate_rate: 0.2333 -> 0.7667
pass_rate:                0.1333 -> 0.6333
```

### 面试官可能问

**Q: 你是不是为了涨分改 gold？**

A: 不是直接把系统输出当 gold，而是在同一文档内用问题类型和业务关键词重新选择更合理的参考 chunk，并保留旧 gold 的元数据。目的是修复自动 eval 的明显噪声，让评测更接近真实问答目标。

## 6. bge-reranker 和混合排序

### 一句话讲法

我接入了 `BAAI/bge-reranker-base` 作为 cross-encoder reranker，但发现纯 bge 不一定最好，最终采用 bge 分数和原召回分 50/50 混合排序。

### 为什么做

召回阶段只能粗筛，reranker 可以逐对判断 query 和 chunk 是否匹配，提高 top-k 质量。但 cross-encoder 会丢掉一些召回阶段的先验，比如公司名、页码、章节、业务类型，所以需要混合排序。

### 指标或证据

```text
无 reranker:
retrieval_mean: 0.4933
pass_rate:      0.6333

轻量 reranker:
retrieval_mean: 0.5333
pass_rate:      0.6000

纯 bge:
retrieval_mean: 0.4933
pass_rate:      0.5667

bge + 原召回分 50/50:
retrieval_mean: 0.5344
pass_rate:      0.7000
```

### 面试官可能问

**Q: bge-reranker 和 bge-m3 区别是什么？**

A: `bge-m3` 是 embedding model，用来把文本变成向量，做 vector search。`bge-reranker` 是 cross-encoder reranker，输入是 query + chunk，输出相关性分数，用于重排候选。

**Q: 为什么纯 bge 反而不最好？**

A: 因为它只看 query-chunk 语义，不知道我们召回阶段的一些结构化先验，比如目标公司、文档标题、页码、章节类型。混合排序可以把语义判断和召回先验结合起来。

## 7. Rerank Cache 和 Candidate 调优

### 一句话讲法

我发现 cross-encoder reranker 延迟高，于是把 query-chunk pair 的 bge 分数缓存到 PostgreSQL，并通过候选数实验把默认 rerank candidates 收敛到 30。

### 为什么做

bge-reranker 是逐对打分。一个问题如果有 50 个候选，就要跑 50 次 query-chunk pair scoring。候选越多，冷查询越慢。所以需要用 eval gate 比较 `20/30/50` 这类候选数，而不是盲目追求更大的 candidate pool。

### 怎么做

- 新增 `rerank_cache` 表：`query_hash + chunk_id + rerank_model -> score`。
- rerank 前先查 cache。
- miss 才跑 bge。
- bge 分数写回 cache。
- debug 里记录 cache hits/misses。
- 用 `golden_dataset.json` 做候选数调优，比较 Hit@5、MRR、P.Hit@5 和延迟。
- 把默认 `RAG_RERANK_CANDIDATES` 从实验态的大候选池收敛为 `30`。

### 指标或证据

```text
golden_dataset.json, 51 cases:
candidates=20 Hit@5=1.000 MRR=0.904 P.Hit@5=1.000
candidates=30 Hit@5=1.000 MRR=0.908 P.Hit@5=1.000
```

默认选择 `30`，因为它保持 Hit@5 100%，MRR 比 20 更稳，同时比更大的候选池冷查询更快。

```text
Tesla automotive revenue query:
answer/retrieval cache 关闭时，调优后约 2.2s
answer cache 命中时约 57ms
```

### 面试官可能问

**Q: cache 什么时候有效？**

A: 对重复 query、相似 eval、热门问题非常有效。对全新 query 仍然需要冷启动打分，所以还要配合 candidate 数调优和更好的召回缩小候选池。

**Q: 为什么 candidates=50？**

A: 我没有保留 50 作为默认值。现在默认是 30，因为 regression 里 Hit@5 仍然是 100%，MRR 0.908，比 20 更稳，也比更大的候选池快。50 可以作为高召回实验配置，但生产默认要平衡质量和延迟。

**Q: Hit@5 和 MRR 分别看什么？**

A: Hit@5 看正确证据有没有进入前 5，MRR 看正确证据排得有多靠前。Hit@5 100% 说明证据找得到，MRR 0.908/0.915 说明多数 case 的正确证据排在第 1 或第 2。

## 8. Hybrid Search：BM25 + Vector Search + RRF

### 一句话讲法

我把 retrieval 从单一路径升级为 hybrid search：BM25/lexical 找精确词，vector search 找语义相似，再用 RRF 融合候选，最后交给 reranker。

### 为什么做

BM25 擅长精确匹配，比如公司名、财务术语、年份。向量检索擅长语义匹配，比如“收入为什么下降”和“gross profit margins declined”。两者结合可以提高召回覆盖。

### 当前进度

- 已新增 `scripts/embed_chunks.py`，用 `BAAI/bge-m3` 生成真实 embedding。
- 已在 `RetrievalService` 加 `RAG_ENABLE_VECTOR=1` 控制 vector search。
- 已实现 RRF 融合 lexical rank 和 vector rank。
- 已验证 query embedding -> pgvector search -> RRF 融合链路。

当前真实 embedding 覆盖率：

```text
总 chunks:       12,486
BAAI/bge-m3:     12,486
hash-bootstrap:  0
覆盖率:          100%
```

### 面试官可能问

**Q: 为什么要 hybrid search，而不是只用向量？**

A: 因为企业文档查询里有很多精确锚点，比如公司名、年份、财务术语、页码和产品名。BM25 对这些精确词更稳，向量检索对语义改写更稳，RRF 能把两类候选融合起来，再交给 reranker 排序。

**Q: RRF 是什么？**

A: RRF 是 Reciprocal Rank Fusion，只看排名不看原始分数。它适合融合 BM25 和 vector search，因为两者分数尺度不同。公式大概是 `1 / (k + rank)`，一个 chunk 如果在两个召回路径里都排得靠前，总分会更高。

## 9. Answer Generation 和 Grounded Answer

### 一句话讲法

当前 answer generation 已接入 Ollama/Qwen，并保留 extractive/direct-extractive 兜底：普通总结交给 LLM，财报数字类高置信问题优先用证据行直接抽取，减少幻觉和相邻指标误选。

### 为什么这样做

RAG 要先找对，再回答好。如果 retrieval 不稳定，直接接 LLM 只会把问题藏起来。现在 retrieval/rerank/eval 成型后，LLM 用来组织答案，但对表格数字类问题要加 deterministic guardrail，因为 LLM 很容易在同一个表格里选中相邻但不同口径的数字。

### 怎么做

- 本地启动 Ollama。
- 配置 `OLLAMA_URL` 和 `OLLAMA_MODEL`。
- prompt 里要求只能基于 Sources 回答，必须带引用，不足就说不足。
- eval 时区分 retrieval score 和 answer score。
- 对高置信度数字事实题，先尝试 direct-extractive：根据问题里的年份和指标行标签从 evidence 中抽值。
- 如果 direct-extractive 不命中，再调用 LLM；如果 LLM 输出包含证据中不存在的数字，用 soft number guard 清理。

### 面试官可能问

**Q: 为什么不用 LLM 直接回答？**

A: 企业知识库的问题需要基于私有/特定文档回答，LLM 本身不知道这些文档，也容易幻觉。RAG 的价值是把答案 grounded 到可引用的文档片段上。

**Q: 为什么要 direct-extractive，不完全依赖 prompt？**

A: prompt 能减少错误，但不能保证表格数字不串行。比如同一段证据里可能同时有 `Automotive sales`、`Total automotive revenues`、`Total revenues`。direct-extractive 对这类高置信结构化问题更稳定，也更容易解释。

## 10. 最近一次排障：评测误判、脏 gold 和数字抽取

### 一句话讲法

我遇到过一次看起来像“模型很差”的结果，但通过拆分 retrieval、citation、answer 三类指标，最后发现主要问题不是模型能力，而是 eval gold 噪声、评分方式不匹配，以及 LLM 在财报表格里选错相邻数字。

### 现象

第一次跑端到端 `eval_runner` 时结果很差：

```text
count:          40
retrieval_mean: 0.3437
answer_mean:    0.0539
citation_mean:  0.6550
pass_rate:      0.2000
```

清洗一轮 gold 之后，citation 变好了，但 answer 仍然很低：

```text
retrieval_mean: 0.4194
answer_mean:    0.0472
citation_mean:  0.7975
pass_rate:      0.1250
```

这个结果一开始很容易误判成“模型不行，需要微调”。但抽样看 `gold_answer` 和实际回答后，发现很多 gold 是网页 cookie、HTML 导航、目录页、表格残片，或者是英文原文摘录，而模型输出是中文总结。当前 `answer_score` 又是 token overlap，所以中英文改写会被严重低估。

### 怎么定位

我没有直接改模型，而是分三步定位：

1. 先跑手工整理的 `data/golden_dataset.json`，只看检索：

```text
Hybrid+Rerank:
Cases   = 51
Hit@5   = 100.0%
MRR     = 0.915
P.Hit@5 = 100.0%
```

这说明 retrieval 本身已经很强，Top-5 没有失败案例。

2. 再跑少量 answer keyword eval：

```text
答案质量评测 5 条样本: 60.0% (3/5)
答案质量评测 10 条样本: 90.0% (9/10)
```

3. 对失败样本逐条看 evidence 和 answer：

- `What was Tesla's automotive revenue in 2023?`
  - 证据里同时有 `Automotive sales 78,509` 和 `Total automotive revenues 82,419`。
  - LLM 选了相邻但不同口径的 `Automotive sales`。
  - eval 里的 expected keyword 原来是 `82,418`，和证据里的 `82,419` 也差 1。

- `How many vehicles did Tesla deliver in 2023?`
  - Top-1 chunk 已经有 `delivered 1,808,581 consumer vehicles`。
  - LLM 却回答成 `473,382 combined Model 3 and Model Y`，这是另一个相邻指标：同比增加量，不是总交付量。

- `How many Supercharger stations does Tesla operate globally?`
  - gold 期待 `5,265`。
  - 当前库里 Tesla 年报片段只说有 global Supercharger network，没有给出这个具体数量。
  - 所以模型说“资料中没有直接提及具体数量”反而是合理的，问题在 gold case。

### 怎么解决

我做了三个修复：

1. 用 `repair_eval_gold.py` 修复自动生成 eval gold 中明显的 cookie、HTML、目录、噪声片段，并把旧 gold 写入 `metadata.gold_repair` 保持可追溯。

2. 对 `golden_dataset.json` 里的错误关键词做事实修正。例如 Tesla automotive revenue 从：

```json
"expected_keywords": ["82,418", "82418"]
```

修成：

```json
"expected_keywords": ["82,419", "82419", "Total automotive revenues"]
```

3. 在 `RetrievalService` 里加了一个通用的 direct-extractive 层，用“年份 + 指标行标签”抽取高置信度数字事实，避免 LLM 在财报表格中选错相邻数字。

这个修复不是写死 Tesla case。最开始我确实写过 Tesla 2023 的特例，后来意识到这有 benchmark overfitting 风险，于是改成更通用的规则：

- 从问题里抽目标年份，比如 `2023`。
- 根据问题意图匹配指标行，比如 `Total automotive revenues`、`Total revenues`、`Net income`、`Research and development`。
- 从匹配行按年份列取对应数值。
- 对 “delivered vehicles” 这类问题，用年份窗口匹配 `delivered X consumer vehicles`。

### 结果

修复后，之前两条失败样本变成 direct-extractive：

```text
What was Tesla's automotive revenue in 2023?
=> The total automotive revenues figure for 2023 was $82,419 million [1].

How many vehicles did Tesla deliver in 2023?
=> The company delivered 1,808,581 consumer vehicles in 2023 [1].
```

10 条答案小样本达到：

```text
关键词命中率: 90.0% (9/10)
```

剩下 1 条失败被确认为 eval case 期望了当前文档里不存在的数字。

### 面试官可能问

**Q: 你是不是为了涨分改 gold？**

A: 我会区分两件事：第一，修复明显错误的 gold，比如 cookie、HTML、目录页、证据里不存在的数字；第二，不能把模型输出直接写成 gold。我的做法是保留修复元数据，并优先用手工整理的 `golden_dataset.json` 做主评测。

**Q: direct-extractive 会不会过拟合 benchmark？**

A: 如果写公司名和固定问题特例，那就是过拟合。我后来改成按年份和指标行标签通用抽取。它本质上是财报 QA 里的 deterministic guardrail，用来处理表格数字和相邻指标混淆，和 LLM 生成互补。

**Q: 为什么不直接微调？**

A: 因为这次问题不是模型知识缺失，而是评测数据、评分方法和表格数字抽取的问题。先修 eval 和抽取逻辑，成本更低，也更可解释。只有当检索正确、证据干净、抽取规则覆盖不了，而模型仍长期总结不稳定时，才考虑微调。

## 11. 总体复盘：我怎么 debug RAG

如果面试官问“你怎么优化这个 RAG 系统”，可以按这个顺序回答：

1. 先做 baseline，跑 eval。
2. 用 top-50 诊断判断问题在召回还是排序。
3. 发现 eval gold 有噪声，清洗 gold。
4. 做 query rewrite、目标文档识别、BM25/IDF 权重。
5. 接 bge-reranker，并比较纯 bge、轻量 reranker、混合排序。
6. 发现 reranker 延迟高，加 PostgreSQL rerank cache。
7. 做 candidate 数调优，比较质量和延迟。
8. 接 hybrid search，用 BM25 + vector + RRF 扩展召回。
9. 把 retrieval eval 和 answer eval 拆开看，避免把 gold 噪声误判成模型问题。
10. 对高置信度财报数字题加 direct-extractive guardrail，减少相邻指标误选。
11. 最后再考虑 prompt、模型或微调。

## 12. React 研究工作台

### 一句话讲法

我没有把前端做成普通聊天气泡，而是做成一个 Apple/macOS 风格的研究工作台：左侧输入问题、示例和运行指标，右侧展示答案、sources、retrieved chunks、matched terms、debug 和 evidence trace。

### 为什么做

RAG 项目最容易被看成“又一个聊天机器人”。我希望前端体现它真正的价值：答案能不能追溯到证据，检索是否命中正确文档，是否用了 LLM，延迟多少，reranker 和 vector search 是否正常。

### 怎么做

- 前端使用 Vite + React + TypeScript。
- 新增 `POST /v1/research/ask`，返回 `answer`、`sources`、`retrieved_chunks`、`debug`。
- 新增 `GET /v1/ops/metrics`，前端展示 requests、errors、p95 latency、pending reviews。
- UI 采用双区布局：侧栏负责 query/examples/metrics，主区域负责 answer/evidence trace。
- 视觉上参考 macOS 工作台：半透明 panel、紧凑工具栏、状态 chip、低噪声卡片和右侧 evidence 列表，让用户先看到研究结果，再按需展开调试信息。
- 默认端口是 `5174`，避免和其他本地项目冲突。

### 面试官可能问

**Q: 为什么不用普通 chat UI？**

A: 因为这个项目的核心不是“聊天”，而是可验证的研究工作流。普通 chat UI 会隐藏 retrieval 和 citation 细节；研究工作台可以直接展示 evidence trail，让用户知道答案来自哪些页、哪些 chunks、哪些匹配词。

**Q: 前端如何帮助 debug？**

A: 前端展示 `retrieved_chunks`、matched terms、rank、score、debug backend 和 latency。这样当答案不准时，我可以马上判断是召回错、rerank 错、证据缺失，还是 LLM 生成错。

## 13. 生产化薄层：可观测性、安全和回归门槛

### 一句话讲法

我给这个 Agent 原型补了一层轻量生产化能力：API request trace、latency/error 统计、可选 API key、限流、人工审核表、agent tool trace 表，以及 retrieval regression gate。

### 为什么做

工业级 Agent 不只是能回答，还要能解释、能监控、能回归测试、能限制访问、能追踪工具调用。否则出了错误很难定位，也很难安全地放到多人环境。

### 怎么做

- `api_request_logs`：记录 request_id、path、status_code、latency、error。
- `answer_reviews`：人工审核关键事实，支持 pending/approved/rejected/needs_revision。
- `agent_tool_traces`：为后续 agent 工具调用记录 tool input/output、status、latency、error。
- GraphRAG fact loop：
  - `research_facts` 存 company、metric、value、period、source_citation、source_chunk_id。
  - `POST /v1/research/ask` 会把高置信数字答案抽成结构化 fact。
  - `GET /v1/graph/facts` 可以查询公司事实图。
  - `GET/PATCH /v1/reviews` 支持人工 approve/reject/needs_revision。
- Async job queue：
  - 复用 `ingestion_jobs` 做 PostgreSQL-backed queue。
  - `POST /v1/jobs` 创建异步任务并立即返回 `job_id`。
  - `scripts/job_worker.py` 后台 claim job，执行后写回 `succeeded/retry/failed` 和 result。
  - 使用 `FOR UPDATE SKIP LOCKED` 避免多个 worker 抢同一任务。
  - 支持 `smoke_test`、`embed_missing_chunks`、`ingest_pdf` 三类任务。
- RAG cache：
  - `query_embedding_cache` 复用 query embedding。
  - `retrieval_result_cache` 复用 top-k chunk ids。
  - `answer_cache` 复用最终答案。
  - `document_parse_cache` 复用 PDF/OCR 解析结果。
- 冷查询优化：
  - `RAG_RERANK_CANDIDATES` 默认收敛为 `30`。
  - `sql/008_vector_indexes.sql` 增加 pgvector HNSW index。
  - `scripts/warmup_rag.py` 和 `RAG_WARMUP_ON_STARTUP=1` 支持启动时预加载 chunks、IDF、embedding model 和 reranker。
  - Apple Silicon 上 embedding model 优先使用 MPS。
- FastAPI middleware：
  - 自动生成或透传 `x-request-id`。
  - 返回 `x-process-time-ms`。
  - `RAG_API_KEYS` 启用 API key 鉴权。
  - `RAG_RATE_LIMIT_PER_MIN` 启用简单限流。
- `/v1/ops/metrics`：给前端 dashboard 提供 24h request/error/latency 指标。
- `scripts/regression_check.py`：用 `golden_dataset.json` 做 retrieval regression gate。
- `.github/workflows/ci.yml`：在 push / pull request 时自动跑 Python 编译检查、guardrail 单测和 React production build。

### 指标或证据

当前 regression gate：

```text
Hit@5     = 1.000
MRR       = 0.915
P.Hit@5   = 1.000
Result    = passed
```

GraphRAG + Human review 小闭环验证：

```text
Question: What was Tesla's automotive revenue in 2023?
Extracted fact:
Tesla -> total automotive revenues -> 82,419 USD million -> 2023 Annual Report p.51

Review:
pending -> approved
research_facts.review_status = approved
```

异步任务队列验证：

```text
POST /v1/jobs smoke_test -> queued
python3 scripts/job_worker.py --once -> succeeded
GET /v1/jobs/{job_id} -> result={"message":"worker ok"}
```

### 面试官可能问

**Q: 这是不是完整工业级系统？**

A: 还不是完整生产系统，更准确地说是面向工业研究工作流的 Agent 原型。它已经有可观测性、CI 单测/构建、回归测试、API 安全和人工审核表的骨架；如果继续生产化，下一步会加队列、权限 ACL enforcement、备份、部署和更完整的 tool governance。

**Q: 为什么要 request_id？**

A: 因为一次 Agent 请求可能经过 API、retrieval、rerank、LLM、cache、tool call 多个环节。request_id 可以把这些日志串起来，方便排查慢请求、失败请求和错误答案。

## 14. 当前状态和下一步边界

### 一句话讲法

当前项目已经不是只剩“微调”。核心 RAG 链路、eval、答案 guardrail、API、前端和轻量生产化能力都已经打通。后面更适合按“展示价值”和“工业化深度”分阶段做，而不是无限加功能。

### 现在已经完成

- RAG ingestion、chunking、embedding、hybrid retrieval、rerank、citation 和 answer generation。
- 手工 golden dataset + retrieval regression gate，当前 51 cases，Hit@5 100%、MRR 0.915、P.Hit@5 100%。
- 10 条 answer keyword 小样本达到 90%，并修复了 automotive revenue、vehicle delivery、Supercharger 无证据这类问题。
- React research workbench 可以从前端查询、查看答案、sources、evidence trace 和 ops metrics。
- API 侧有 request logging、latency/error 统计、可选 API key、限流、人工审核表和 tool trace 表。
- 后端异步队列已落地：API 入队、worker 后台处理、状态查询、失败重试和结果回写形成闭环。
- PostgreSQL cache 层已覆盖 query embedding、retrieval result、answer 和 document parse/OCR，cache key 包含 corpus fingerprint，避免文档更新后读到旧证据。
- 冷查询优化已覆盖 candidate tuning、pgvector HNSW index、模型 warmup 和默认配置收敛；Tesla 财务样例在关闭 answer/retrieval cache 时从约 15.3s 降到约 2.2s。
- GraphRAG 小闭环已落地：关键数字答案会进入 `research_facts`，形成 company -> metric -> value -> source 的事实图。
- Human-in-the-loop 小闭环已落地：关键 fact 会进入 pending review，审核通过后同步标记为 approved fact。
- GitHub Actions CI 已经覆盖 Python guardrail tests 和 React build；重型 RAG regression 仍保留为本地 gate。

### 接下来最值得做的 3 件事

1. **Demo 收尾**：录一条稳定 demo 路线，准备 3 个代表性问题：财务数字、交付/产能、证据不足时拒答。
2. **报告导出**：把 approved facts 自动整理成一页 research brief。
3. **GraphRAG 扩展**：从财务数字扩展到产能、产品、市场区域和风险因素。

### 可以暂时不做

- 大规模微调：当前主要问题不是模型知识缺失，先不做。
- 复杂多 Agent 编排：先把 RAG tool、抽取 tool、报告 tool 三个工具治理好。
- 完整生产部署：现在可以解释为 prototype with production-oriented guardrails，不需要声称已经是完整生产系统。

### 面试官可能问

**Q: 前端做完之后还缺什么？**

A: 缺的不是“能不能用”，而是更深的工业化能力：权限 ACL、备份、异步任务队列、报告导出、更完整的 human review workflow，以及把本地 RAG regression 迁移到可复现的 CI fixture。当前版本已经足够展示一个工程化 RAG Agent 的核心闭环。

**Q: GraphRAG 是必须的吗？**

A: 不是必须。当前 RAG 已经能完成证据问答。GraphRAG 的价值在于实体关系和跨文档归纳，比如公司、产品、指标、项目、供应链关系。我的下一步会从小图谱开始，而不是为了概念直接上复杂图系统。

## 15. 简历 bullet 备选

- 构建面向新能源行业公开资料的研究 Agent，完成 PDF/HTML 采集、解析、清洗、标题+段落 chunking、pgvector 存储、引用追踪、eval 闭环和 React evidence workbench。
- 设计 retrieval eval 体系和 top-50 recall diagnostic，定位召回与排序瓶颈，并修复自动生成 eval gold 噪声，使 pass_rate 从 0.13 提升到 0.63。
- 接入 BAAI/bge-reranker-base cross-encoder，并将 bge 分数与原召回分混合排序，使 retrieval_mean 提升至 0.5344，pass_rate 达到 0.70。
- 为 cross-encoder reranker 增加 PostgreSQL pair-score cache，将热缓存场景下 per-case 延迟约从 2.5s 降至 1.17s。
- 优化 first-query latency：将 rerank candidates 收敛到 30，增加 pgvector HNSW index 和 startup warmup，使 Tesla 财务样例在关闭 answer/retrieval cache 时约从 15.3s 降至 2.2s。
- 实现 BM25/lexical + bge-m3 vector search + RRF 的 hybrid retrieval 框架，在手工 golden dataset 上达到 51 cases、Hit@5 100%、MRR 0.915、P.Hit@5 100%。
- 实现 GraphRAG fact loop 和 human review workflow，将关键财务数字抽成 company -> metric -> value -> source，并支持 pending/approved/rejected 状态治理。
- 实现 PostgreSQL-backed async job queue，用 `FOR UPDATE SKIP LOCKED` 支持多 worker 安全 claim，长任务通过 `ingestion_jobs` 状态机异步执行和重试。
- 拆分检索评测与答案关键词评测，定位端到端低分来自 gold 噪声和数字抽取误差；加入通用 direct-extractive guardrail 后，10 条答案样本关键词命中率达到 90%。
- 增加 API request trace、ops metrics、可选 API key、限流、人工审核表和 regression gate，将 RAG demo 推进到可观测、可回归的研究 Agent 原型。
