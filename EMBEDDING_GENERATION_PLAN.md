# 60000+ 向量批量生成计划

## 📊 任务规模

```
总chunks数: 60,000+
待生成向量: ~60,000
模型: BAAI/bge-m3 (335M参数)
硬件: Apple Silicon (M1/M2/M3)
预期耗时: 12-18 分钟
```

---

## ⚡ 执行步骤

### Step 1: 清理旧数据（可选）

```bash
# 如果要完全重新生成（推荐，确保一致性）
psql -h localhost -p 15432 -U rag -d rag_assistant << EOF
DELETE FROM embeddings WHERE embedding_model = 'BAAI/bge-m3';
-- 验证删除
SELECT COUNT(*) as remaining_embeddings FROM embeddings;
EOF
```

### Step 2: 开启监控终端（重要！）

在**第一个终端**中运行：
```bash
python scripts/monitor_embedding_progress.py
```

这会实时显示：
```
[████████████████████░░░░░░░░░░░░░░░░░░] 50.0% | 已完成: 30000/60000 | 待处理: 30000 | 速率: 185 chunks/s | 耗时: 2.7m | 预计完成: 14:35:22
```

### Step 3: 在第二个终端开始生成向量

```bash
# 推荐方案：一次性全部生成（Apple Silicon优化）
python scripts/embed_chunks.py --force --batch-size 32
```

**参数详解**:
- `--force`: 强制重新生成（即使已有向量也会覆盖）
- `--batch-size 32`: 对M系列GPU的最优批大小
- 无 `--limit`: 处理所有chunks

### Step 4: 实时观察进度

监控窗口会显示：
- 🟩 进度条
- 📊 已完成/总数
- ⚡ 实时速率（chunks/s）
- ⏱️ 已耗时
- 🎯 预计完成时间

---

## 🔧 参数优化对比

### 默认配置（太慢）
```bash
python scripts/embed_chunks.py --batch-size 16 --limit 4000
# ❌ 耗时太长，限制了数量
```

### ✅ 推荐配置
```bash
python scripts/embed_chunks.py --force --batch-size 32
# ✅ 快速完成，优化了批大小
```

### 激进配置（可能OOM）
```bash
python scripts/embed_chunks.py --force --batch-size 64
# ⚠️ 可能导致内存溢出，不推荐
```

---

## 📈 预期性能指标

| 指标 | 预期值 |
|------|--------|
| **总向量数** | 60,000+ |
| **处理速率** | 180-220 chunks/s |
| **总耗时** | 12-18 分钟 |
| **峰值内存** | 2-3 GB |
| **GPU占用** | ~60-80% |

---

## ⚠️ 可能遇到的问题

### 问题 1: 内存不足 (OOM)
**症状**: `RuntimeError: CUDA out of memory` 或进程被杀死

**解决方案**:
```bash
# 降低batch-size
python scripts/embed_chunks.py --force --batch-size 24

# 或分批处理
python scripts/embed_chunks.py --batch-size 16 --limit 20000
```

### 问题 2: 数据库连接超时
**症状**: `psycopg.OperationalError: timeout`

**解决方案**:
```bash
# 检查PostgreSQL是否运行
docker ps | grep postgres

# 重启PostgreSQL
docker-compose restart postgres
```

### 问题 3: 向量维度不匹配
**症状**: `RuntimeError: Embedding dimension X does not match embeddings.embedding vector(1024)`

**原因**: 模型或配置变了

**解决方案**: 检查 `.env` 中的 `RAG_EMBED_MODEL` 和维度配置

---

## 💾 监控数据库进度

如果监控脚本崩溃，可以手动查询：

```bash
psql -h localhost -p 15432 -U rag -d rag_assistant << EOF
-- 统计嵌入进度
SELECT
  (SELECT COUNT(*) FROM chunks) as total_chunks,
  (SELECT COUNT(*) FROM embeddings WHERE embedding_model = 'BAAI/bge-m3') as embedded_chunks,
  (SELECT COUNT(*) FROM chunks c WHERE NOT EXISTS (
    SELECT 1 FROM embeddings e 
    WHERE e.chunk_id = c.id AND e.embedding_model = 'BAAI/bge-m3'
  )) as pending_chunks;
EOF
```

---

## 🎯 完整执行流程（推荐）

### Terminal 1: 监控
```bash
cd /path/to/enterprise-knowledge-base-assistant
python scripts/monitor_embedding_progress.py
```

### Terminal 2: 生成
```bash
cd /path/to/enterprise-knowledge-base-assistant
# 看到监控启动后，运行这个
python scripts/embed_chunks.py --force --batch-size 32
```

**预期输出顺序**:
```
Terminal 2:
{'model': 'BAAI/bge-m3', 'selected': 60000, 'embedded': 0, 'dimension': None}
embedded 100/60000 chunks with BAAI/bge-m3
embedded 200/60000 chunks with BAAI/bge-m3
...

Terminal 1:
[████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 7.5% | 已完成: 4500/60000 | 待处理: 55500 | 速率: 195 chunks/s | ...
[█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 8.2% | 已完成: 4900/60000 | 待处理: 55100 | 速率: 192 chunks/s | ...
...
[██████████████████████████████████████░] 99.0% | 已完成: 59400/60000 | 待处理: 600 | 速率: 188 chunks/s | 预计完成: 14:35:42
[██████████████████████████████████████] 100.0% | 已完成: 60000/60000 | 待处理: 0 | 速率: 190 chunks/s | 耗时: 5.3m

✅ 完成！总耗时: 5.3m
   总向量数: 60,000
   平均速率: 189 chunks/s
```

---

## ✅ 完成检查清单

- [ ] PostgreSQL 容器运行中 (`docker ps | grep postgres`)
- [ ] 启动监控脚本 (`monitor_embedding_progress.py`)
- [ ] 运行生成命令 (batch-size 32)
- [ ] 观察进度达到 100%
- [ ] 验证向量数量正确

```bash
# 最后验证
psql -h localhost -p 15432 -U rag -d rag_assistant -c \
  "SELECT COUNT(*) FROM embeddings WHERE embedding_model = 'BAAI/bge-m3';"
# 应该显示: ~60000
```

---

## 📝 预期完成时间表

| 时间点 | 进度 | 耗时 |
|--------|------|------|
| 开始 | 0% | 0m |
| 1分钟后 | ~12% | 1m |
| 3分钟后 | ~36% | 3m |
| 5分钟后 | ~60% | 5m |
| 8分钟后 | ~96% | 8m |
| **完成** | **100%** | **~15m** |

---

**开始时间**: 2026-05-08  
**更新状态**: 准备批量生成60000+向量
