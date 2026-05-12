# 企业文档知识库 - 数据处理管道指南

## 📊 项目概述
这是一个企业文档知识库查询助手系统，将企业文档（PDF年报、报告等）转换为可查询的知识库。

---

## 🔄 数据处理流程

### 整体流程图
```
原始文档 → 收集 → 解析 → 清洁 → 分块 → 向量化 → 数据库存储 → 搜索/查询
```

---

## 📋 核心脚本说明

### 1️⃣ **bootstrap_db.py** - 初始化数据库
**作用**: 创建PostgreSQL数据库schema和必要的表结构

**执行命令**:
```bash
python scripts/bootstrap_db.py
```

**关键操作**:
- 执行 `sql/schema.sql` 初始化表结构
- 添加列扩展（parent_chunk_id, chunk_level等）
- 创建索引优化查询性能

**输出**: 
```
initialized schema from sql/schema.sql
```

---

### 2️⃣ **collect_energy_data.py** - 数据收集
**作用**: 从web下载能源领域的文档（年报、报告等）

**源数据配置**: `data/sources/energy_sources.json`

**执行命令**:
```bash
python scripts/collect_energy_data.py
```

**流程**:
1. 读取 `energy_sources.json` 配置文件
2. 根据 `source_kind` 分类处理:
   - `pdf`: 直接下载PDF文件
   - `page`: 下载HTML页面，从页面中提取PDF链接
3. 保存文件到 `data/raw/energy/` 结构
4. 生成 `_collection_manifest.jsonl` 元数据文件

**输出结构**:
```
data/raw/energy/
├── first-solar/
│   ├── first-solar-annual-reports/
│   │   ├── index.html
│   │   └── pdfs/
│   │       ├── 2020-Annual-Report.pdf
│   │       ├── 2021-AR.pdf
│   │       └── ...
│   └── [其他FIRST SOLAR相关文档]
├── tesla/
│   └── tesla-2024-annual-report/
│       ├── index.html
│       └── pdfs/
│           └── tsla-20241231-gen.pdf
├── catl/
├── byd/
├── iea/
└── _collection_manifest.jsonl
```

**_collection_manifest.jsonl 格式**:
```json
{
  "source_id": "first-solar-annual-2024",
  "title": "First Solar 2024 Annual Report",
  "source_url": "https://...",
  "artifact_type": "pdf",
  "local_path": "data/raw/energy/first-solar/...",
  "content_type": "application/pdf",
  "sha256": "abc123...",
  "company": "First Solar",
  "category": "energy",
  "downloaded_at": "2024-05-06T..."
}
```

---

### 3️⃣ **ingest_energy_data.py** - 文档处理和分块 ⭐ 核心脚本
**作用**: 将PDF转换为结构化的chunks，存入数据库

**执行命令**:
```bash
python scripts/ingest_energy_data.py [--limit N] [--skip-ocr]
```

**核心处理流程**:

#### A. PDF文本提取（三种方式）
1. **pdftotext**: 优先使用（快速）
   - 使用 `pdftotext -layout` 命令
   - 保持原文档布局
   
2. **OCR Fallback**: 当文本提取失败时
   - 使用 `pdftoppm` 转换为图像
   - 使用 `tesseract` 进行OCR识别
   - 支持中英文混合识别

3. **质量评估**:
   - 计算文本密度 = 提取字符数 / 页数
   - 如果文本提取字符 < 40%文件大小，触发OCR

#### B. 文本清洁 (clean_pdf_pages)
清除噪声和样板文本:
- ✂️ 移除重复出现的页脚/页头（如页码、公司名等）
- 🗑️ 移除纯数字行、空行、符号行
- 🔗 移除常见法律模板文本（"All rights reserved", "Forward-looking statements"等）
- 📏 移除仅包含多列空格的行（表格边界）

#### C. 结构识别
识别章节和段落:
- 识别**标题行**（通常是大写或加粗的短行）
- **段落检测**: 将文本分割为逻辑段落
- **表格识别**: 检测多列数据结构

#### D. 分块策略 (Three-level hierarchy)

**Level 1: Chapter Chunks** (完整章节)
- 包含整个章节的所有段落
- 使用于全局搜索

**Level 2: Paragraph Chunks** (段落级)
- 单个段落或小段落组
- 更精细的信息检索

**Level 3: Sentence Chunks** (句子级)
- 当段落过长时自动分割
- 最细粒度的搜索单位

#### E. 元数据附加
每个chunk包含:
```json
{
  "source_id": "tesla-2024",
  "artifact_type": "pdf",
  "chunk_level": "chapter",
  "section_index": 0,
  "section_title": "Business Overview",
  "paragraph_count": 5,
  "page_start": 1,
  "page_end": 3
}
```

#### F. 向量化
- 使用简单hash向量作为bootstrap（dimension=1024）
- 后续可用真实Embedding模型替换

**输出到数据库**:
- `documents` 表: 原始文档记录
- `document_versions` 表: 文档版本
- `chunks` 表: 分块内容和元数据
- `embeddings` 表: 向量表示
- `ingestion_jobs` 表: 处理记录

---

### 4️⃣ **seed_sample_data.py** - 种子数据加载
**作用**: 加载seed_docs目录中的Markdown文档作为样本数据

**执行命令**:
```bash
python scripts/seed_sample_data.py
```

**处理流程**:
1. 读取 `data/raw/seed_docs/*.md` 文件
2. 按 `#` 标题分割为sections
3. 为每个section生成向量
4. 插入到数据库

---

### 5️⃣ **embed_chunks.py** - 向量嵌入（可选）
**作用**: 用真实Embedding模型（BAAI/bge-m3）替换bootstrap向量

**执行命令**:
```bash
# 嵌入所有未处理的chunks
python scripts/embed_chunks.py --limit 4000 --batch-size 16

# 重新嵌入所有chunks
python scripts/embed_chunks.py --force --batch-size 16
```

**参数说明**:
- `--model`: 向量模型（默认: BAAI/bge-m3）
- `--batch-size`: 批处理大小（默认: 16）
- `--limit`: 限制处理数量（可选）
- `--force`: 重新嵌入已有向量的chunks

**处理流程**:
1. 从数据库查询未嵌入的chunks
2. 使用sentence-transformers加载模型
3. 批量向量化文本
4. 验证向量维度（必须=1024）
5. 存储到embeddings表

---

## 🔧 其他辅助脚本

### find_catl_pdfs.py
查找CATL相关的PDF文件

### extract_graph.py
从PDF中提取表格和图表

### answer_question.py
基于知识库回答问题的示例脚本

---

## 📈 数据库表结构

```sql
documents
├── id (UUID)
├── source_type (local_file, web, etc)
├── title (文档标题)
├── file_hash (文件哈希，用于去重)
└── metadata (JSONB)

document_versions
├── id (UUID)
├── document_id (FK)
├── version_no
├── content_hash
└── metadata

chunks ⭐ 核心表
├── id (UUID)
├── document_version_id (FK)
├── section_title (章节标题)
├── text (文本内容) ⭐
├── chunk_level (chapter/paragraph/sentence)
├── chunk_index
├── page_start/page_end
├── token_count
└── metadata (JSONB)

embeddings ⭐ 向量表
├── chunk_id (FK)
├── embedding (vector[1024])
├── embedding_model (BAAI/bge-m3)
└── dimension

ingestion_jobs
├── document_id (FK)
├── job_type
├── status
└── metadata
```

---

## 🚀 完整执行顺序

### **第一次启动** (从零开始)

```bash
# 1. 启动Docker容器
docker-compose up -d

# 2. 初始化数据库schema
python scripts/bootstrap_db.py

# 3. 收集能源领域数据（可选，只需一次）
python scripts/collect_energy_data.py

# 4. 处理并导入能源文档 ⭐ 最耗时
python scripts/ingest_energy_data.py

# 5. 加载种子文档
python scripts/seed_sample_data.py

# 6. 生成真实向量（可选，耗时）
python scripts/embed_chunks.py --limit 4000 --batch-size 16
```

### **数据丢失后恢复** (您当前的情况)

```bash
# 1. 确保Docker运行
docker-compose up -d

# 2. 清空数据库
psql -h localhost -p 15432 -U rag -d rag_assistant -c "
TRUNCATE embeddings CASCADE;
TRUNCATE chunks CASCADE;
TRUNCATE document_versions CASCADE;
TRUNCATE documents CASCADE;
TRUNCATE ingestion_jobs CASCADE;
"

# 3. 重新初始化schema
python scripts/bootstrap_db.py

# 4. 重新导入数据
python scripts/ingest_energy_data.py

# 5. 重新加载种子数据
python scripts/seed_sample_data.py

# 6. （可选）重新生成向量
python scripts/embed_chunks.py --force
```

---

## 📊 预期结果

执行完成后，应该看到：

```
✓ Database initialized
✓ Collected X artifacts from energy sources
✓ Ingested X documents into Y chunks
  - Total pages processed: N
  - Total text characters: M
  - OCR usage: Z%
  - Removed noise lines: K
✓ Seeded X documents from seed_docs/
✓ Embedded X chunks with BAAI/bge-m3
```

数据库统计：
```sql
SELECT 
  COUNT(*) FILTER(WHERE source_type='local_file') as seed_docs,
  COUNT(*) FILTER(WHERE source_type='web') as ingested_documents,
  (SELECT COUNT(*) FROM chunks) as total_chunks,
  (SELECT COUNT(*) FROM embeddings) as embedded_chunks;
```

---

## ⚠️ 常见问题

### Q: ingest_energy_data.py 很慢？
**A**: 
- 首次运行需要处理PDF提取和清洁，耗时正常
- 使用 `--limit N` 参数处理部分文件测试
- 确保 `pdftotext` 和 `tesseract` 已安装

### Q: embed_chunks.py 失败（psycopg错误）?
**A**: 
- 确保PostgreSQL容器运行: `docker ps | grep postgres`
- 检查 `.env` 中 DATABASE_URL 配置正确

### Q: 向量维度不匹配错误?
**A**:
- 确保使用 `BAAI/bge-m3` 模型（维度=1024）
- 或更新 `sql/schema.sql` 中的向量维度定义

---

## 🔗 关键环境变量

```env
DATABASE_URL=postgresql://rag:rag@localhost:15432/rag_assistant
REDIS_URL=redis://localhost:16379/0
RAG_EMBED_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

---

**最后更新**: 2026-05-08
**项目状态**: 数据处理管道完成 → LLM接入中 → 微调评估阶段
