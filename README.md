# Copyless — 面向学术论文的语义查重检测系统

> 一个深度查重系统，结合大规模句向量索引与多阶段语义/词法融合判定，精准识别学术论文中的抄袭、改写与引用行为。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.10%2B-dc382c.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 核心特性

- **句级深度检测** — 不是简单的关键词匹配，而是对每一句进行语义相似度分析，能捕捉改写和细微措辞变化。
- **稠密+稀疏混合检索** — 结合 Qwen3-0.6B 稠密向量与哈希词袋稀疏向量，通过倒数排名融合（RRF）策略实现高召回率。
- **多阶段判定管线** — 两阶段检索 → 重排序 → 决策树分类，同时利用余弦相似度和归一化编辑距离进行综合判定。
- **智能引用识别** — 自动解析参考文献章节，通过上下文窗口扫描区分合法引用与抄袭行为。
- **异步在线服务** — 生产级 FastAPI 服务，支持异步任务队列、后台 Worker（线程池执行器处理 CPU 密集任务）以及 Webhook 回调通知。
- **完整评测框架** — 内置句级和文档级评测，支持 Precision/Recall/F1 指标、延迟画像（P95/P99）和吞吐量统计。

---

## 🏗️ 系统架构

```
┌─────────────────────────── 离线索引管线 ───────────────────────────┐
│                                                                     │
│  PDF/LaTeX  ──→  文本提取      ──→  清洗分句      ──→  向量编码    │
│  (PyMuPDF)      (extract.py)      (preprocess.py)    (Qwen3-0.6B)  │
│                                                           │         │
│                                            稠密向量 + 稀疏向量      │
│                                                           │         │
│                                                  ┌────────▼──────┐  │
│                                                  │    Qdrant     │  │
│                                                  │  向量数据库    │  │
│                                                  └───────┬───────┘  │
└──────────────────────────────────────────────────────────────────────┘
														   │
┌─────────────────────────── 在线查重服务 ───────────────────────────┐
│                                                                     │
│  用户论文  ──→  FastAPI  ──→  任务队列   ──→  异步 Worker          │
│  (POST /v1/    (api.py)     (tasks.py)      (worker.py)            │
│   papers/check)                                  │                  │
│                                   ┌──────────────┘                  │
│                                   ▼                                 │
│                          ┌─────────────────┐                        │
│                          │ 逐句处理:        │                        │
│                          │ 1. 向量编码      │                        │
│                          │ 2. 向量检索      │                        │
│                          │ 3. Top-K 重排序  │                        │
│                          │ 4. 编辑距离计算  │                        │
│                          │ 5. 决策树分类    │                        │
│                          │ 6. 引用识别      │                        │
│                          └────────┬────────┘                        │
│                                   ▼                                 │
│                           报告生成                                   │
│                     (总体分数 + 来源排名                              │
│                      + 逐句详情)                                     │
│                                   │                                 │
│                         GET /v1/reports/{id}                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 技术栈

| 层级 | 技术方案 |
|------|----------|
| **向量模型** | Qwen3-0.6B（Mean Pooling + L2 归一化，GPU FP16 推理） |
| **向量数据库** | Qdrant（HNSW 索引，余弦相似度，稠密+稀疏混合检索） |
| **Web 框架** | FastAPI + Uvicorn（async lifespan，后台 Worker） |
| **文本提取** | PyMuPDF（PDF）、pylatexenc（LaTeX）、正则回退 |
| **分句工具** | NLTK Punkt、spaCy、自定义中英文混合分句器 |
| **相似度指标** | 余弦相似度（语义） + 归一化 Levenshtein 距离（词法） |
| **融合策略** | 倒数排名融合（Reciprocal Rank Fusion, RRF） |
| **任务队列** | 内存异步队列，支持 TTL 自动清理 |
| **评测体系** | 自研框架：Precision/Recall/F1，P95/P99 延迟 |

---

## 📁 项目结构

```
Copyless/
├── src/
│   ├── pipeline.py          # 离线批处理 CLI 入口（提取 → 编码 → 索引）
│   ├── extract.py           # PDF/LaTeX 文本提取（含异常处理）
│   ├── preprocess.py        # 文本清洗、分句（NLTK/spaCy/混合模式）
│   ├── embedding.py         # Qwen3-0.6B 句向量编码器（GPU 批量编码、L2 归一化）
│   ├── qdrant_io.py         # Qdrant 集合管理 & 批量向量操作
│   ├── hybrid_search.py     # 稠密+稀疏混合检索（RRF 融合）
│   ├── benchmark.py         # 句级 & 文档级评测框架
│   ├── metrics.py           # Precision/Recall/F1 & 延迟统计
│   └── service/
│       ├── api.py           # FastAPI 路由（异步提交 + 结果轮询）
│       ├── config.py        # 统一配置（pydantic-settings，.env 支持）
│       ├── models.py        # Pydantic 数据模型（请求、报告、任务状态）
│       ├── tasks.py         # 内存任务队列（TTL 自动清理）
│       ├── retrieval.py     # 检索管线（预处理 → 编码 → 检索）
│       ├── worker.py        # 异步后台 Worker（线程池执行 CPU 密集任务）
│       ├── citations.py     # 参考文献解析 & 上下文窗口引用检测
│       ├── report.py        # 报告聚合（总体评分、来源排名）
│       ├── utils.py         # 编辑距离、决策树、加权评分
│       └── templates/       # Web 前端模板（交互式演示页面）
├── scripts/                 # 数据同步 & 环境搭建辅助脚本
├── requirements.txt         # 依赖清单（含版本上下限约束）
├── Doc.md                   # 技术设计文档
└── README.md
```

---

## 🚀 快速开始

### 1. 环境搭建

```bash
# 推荐 Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# （可选）spaCy 分句模型
python -m spacy download en_core_web_sm
```

### 2. 离线索引管线

从学术论文构建句向量索引库：

```bash
python -m src.pipeline \
	--input data/pdf_extracted \
	--collection copyless-main \
	--model models/Qwen3-0.6B \
	--device cuda \
	--batch-size 256 \
	--sentence-splitter nltk \
	--workers 4 \
	--qdrant-url http://localhost:6333
```

**空运行模式**（无需 Qdrant）：

```bash
python -m src.pipeline \
	--input data/sample \
	--collection test \
	--model dummy \
	--dry-run \
	--dump outputs/sample.jsonl
```

### 3. 启动在线查重服务

```bash
uvicorn src.service.api:app --host 0.0.0.0 --port 8080
```

**提交论文：**
```bash
curl -X POST http://localhost:8080/v1/papers/check \
	-H 'Content-Type: application/json' \
	-d '{"content": "论文文本内容...", "callback_url": "https://example.com/hook"}'
```

**查询结果：**
```bash
curl http://localhost:8080/v1/reports/<task_id>
```

**Web 演示页面：** 访问 `http://localhost:8080/` 即可使用浏览器交互界面。

### 4. 运行评测

```bash
# 句级评测
python -m src.benchmark sentences \
	--data data/bench/sentences.jsonl \
	--model models/Qwen3-0.6B \
	--device cuda --threshold 0.85

# 文档级评测
python -m src.benchmark documents \
	--data data/bench/docs.jsonl \
	--model models/Qwen3-0.6B \
	--device cuda
```

---

## 🧠 核心算法

### 相似度判定决策树

系统采用**规则决策树**，融合语义（余弦）和词法（编辑距离）信号，实现可解释的分类判定：

```
IF    Sim_lev ≥ T_lev_high (0.99)      →  完全相同（Identical）
ELIF  Sim_lev ≥ T_lev_med  (0.90)
	  AND Sim_cos ≥ T_cos_high (0.95)  →  微调修改（Minor Changes）
ELIF  Sim_cos ≥ T_cos_mid  (0.88)      →  改写（Paraphrased）
ELSE                                    →  原创（Original）

// 后处理：若分类为微调修改或改写，
// 且上下文窗口中存在对匹配来源的引用
//    →  覆盖为已引用（Cited）
```

### 加权融合评分

```
Score_final = 0.7 × Sim_cosine + 0.3 × Sim_levenshtein
```

### 文档总体相似度

```
Score = (N_identical × 1.0 + N_minor × 0.8 + N_paraphrased × 0.6) / N_total
```

### 引用检测管线

1. **参考文献解析** — 提取文末 Bibliography，将标签 `[1]`、`[Author 2025]` 映射到论文 ID
2. **行内引用定位** — 基于正则表达式检测正文中的引用标记
3. **上下文窗口扫描** — 检查被标记句子 ±K 范围内是否存在相关引用
4. **状态覆盖** — 确认引用存在后，将 `微调修改`/`改写` 重分类为 `已引用`

---

## 📊 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/papers/check` | POST | 提交论文进行查重（异步） |
| `/v1/reports/{task_id}` | GET | 轮询任务状态和获取报告 |
| `/v1/benchmarks/run` | POST | 提交评测任务 |
| `/` | GET | Web 交互演示页面 |

### 报告结构

```json
{
	"overall_similarity_score": 0.235,
	"summary": {
		"total_sentences": 500,
		"identical_count": 20,
		"minor_changes_count": 45,
		"paraphrased_count": 58,
		"cited_count": 12,
		"original_count": 365
	},
	"top_sources": [
		{"paper_id": "arXiv:2401.12345", "score": 15.2, "sentence_count": 18}
	],
	"sentence_details": [...]
}
```

---

## ⚙️ 配置项

所有配置均可通过环境变量（前缀 `COPYLESS_`）或 `.env` 文件覆盖：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `COPYLESS_QDRANT_URL` | `http://localhost:6333` | Qdrant 服务地址 |
| `COPYLESS_EMBEDDING_MODEL` | `models/Qwen3-0.6B` | 句向量模型路径 |
| `COPYLESS_TOP_K` | `5` | 每句检索候选数 |
| `COPYLESS_T_COS_HIGH` | `0.95` | 余弦相似度高阈值 |
| `COPYLESS_T_COS_MID` | `0.88` | 余弦相似度中阈值 |
| `COPYLESS_T_LEV_HIGH` | `0.99` | 编辑距离高阈值 |
| `COPYLESS_T_LEV_MED` | `0.90` | 编辑距离中阈值 |
| `COPYLESS_WORKER_COUNT` | `2` | 后台 Worker 数量 |

---

## 📈 评测指标

评测框架覆盖以下维度：

- **准确性**：句级 & 文档级 Precision / Recall / F1
- **延迟**：编码和检索的平均、P95、P99 延迟
- **吞吐量**：编码 Sentences/sec，检索 Queries/sec
- **后端对比**：内存模式（NumPy）vs Qdrant 实际检索

---

## 🗺️ 路线图

- [ ] 跨语言检测（多语言 Embedding 模型）
- [ ] 公式 & 表格相似度检测
- [ ] 在查重语料上微调句向量模型
- [ ] 分布式任务队列（Redis/Celery）以支持生产级扩展
- [ ] 交互式 HTML 报告（高亮 Diff 视图）
- [ ] Helm Chart & Docker Compose 一键部署

---

## 📖 参考资料

- [Qdrant 文档](https://qdrant.tech/documentation/)
- [PAN 查重语料库](https://pan.webis.de/data.html)
- [Qwen3 模型](https://huggingface.co/Qwen)
- [Copyleaks](https://copyleaks.com/) — 行业参考报告格式

---

## License

MIT
