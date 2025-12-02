## Copyless 技术概览

Copyless 是一套面向学术论文的深度查重系统原型，目标是结合大规模句向量索引与多阶段精排，识别复制、改写与引用等多类型重复行为。本文档覆盖当前仓库内所有主要脚本与工具的用途，帮助你快速搭建环境、执行离线索引、扩展在线查重能力并完成性能评估。

> **温馨提示**：在线查重服务、引用识别、报告生成与评测脚本尚未完备。README 会说明这些模块的预期设计、待办事项与推荐实现思路，便于后续迭代。

> **默认设定**：全仓库默认使用 Qwen3-0.6B 作为句向量模型，并假设运行环境具备 NVIDIA GPU；示例中的 `--device cuda`、`CUDA_VISIBLE_DEVICES` 等参数均以此为前提。仅在单元测试或快速验证时才会使用 `dummy` 模型。

> **Doc.md 标准**：`Doc.md` 是 Copyless 的正式技术规范，包含系统架构、流程与评测准则。README 关注操作指南与快速上手；遇到冲突或疑问时，请以 `Doc.md` 内容为准，并按本文所列清单自检实现是否符合标准。

---

## 目录

1. [项目结构](#项目结构)
2. [环境准备](#环境准备)
3. [离线数据处理与索引构建](#离线数据处理与索引构建)
4. [核心脚本与模块用法](#核心脚本与模块用法)
5. [在线查重服务](#在线查重服务)
6. [性能评估与基准测试](#性能评估与基准测试)
7. [在线查重服务设计草案](#在线查重服务设计草案)
8. [常见问题与排错](#常见问题与排错)
9. [后续路线图](#后续路线图)
10. [参考资料](#参考资料)

---

## 项目结构

```text
Copyless/
├── data/                    # 示例数据与抽取结果
├── models/Qwen3-0.6B/       # 默认离线句向量模型（Qwen3-0.6B）
├── outputs/                 # dry-run 或导出 JSONL 的默认目录
├── scripts/                 # 辅助脚本（保留核心抽取/同步工具）
├── src/
│   ├── pipeline.py          # 离线批处理入口（CLI）
│   ├── extract.py           # PDF / LaTeX 文本抽取工具
│   ├── preprocess.py        # 文本清洗与分句策略
│   ├── embedding.py         # Qwen 句向量编码器封装
│   ├── qdrant_io.py         # Qdrant 集合与向量写入封装
│   └── benchmark.py         # 预留的质量/性能基准工具
├── requirements.txt         # Python 依赖清单
└── README.md                # 本文档
```

---

## 环境准备

1. **Python 版本**：推荐 Python 3.10+。
2. **虚拟环境**：建议使用 `venv` 或 `conda` 隔离依赖。
3. **安装依赖**：

```bash
pip install -r requirements.txt
```

4. **额外依赖**：
	 - PyMuPDF（`fitz`）用于 PDF 文本抽取，requirements 已包含。
	 - spaCy 英文模型：若需使用 `sentence_splitter=spacy`，需额外下载。

```bash
python -m spacy download en_core_web_sm
```

5. **GPU 环境**：默认在 NVIDIA GPU 上运行 Qwen3-0.6B 编码，请确认已安装兼容的驱动与 CUDA（建议 11.8+）。若需在多 GPU 机器上运行，可通过 `CUDA_VISIBLE_DEVICES` 限定使用的设备。

6. **Qdrant**：
	- 本地嵌入式模式：无需额外服务，`pipeline.py` 通过 `--qdrant-path` 创建 SQLite 存储。
	- 远程服务模式：可直接对接已有 Qdrant 集群，或参考仓库附带的 [`docker-compose.qdrant.yml`](./docker-compose.qdrant.yml) 快速启动。详尽配置手册见 [`docs/qdrant_remote_setup.md`](./docs/qdrant_remote_setup.md)。默认连接地址 `http://localhost:6333`，可通过环境变量 `QDRANT_URL` 和 `QDRANT_API_KEY` 覆盖。

7. **模型准备**：仓库自带 `models/Qwen3-0.6B`。如需替换，可将 `--model` 指向 HuggingFace 名称或本地路径。

## Doc.md 对齐检查清单

- **数据与索引流程**：与 `Doc.md` 第 2 章保持一致，确保离线抽取 → 清洗分句 → 向量化 → 写入 Qdrant 的顺序与参数配置同步。
- **判定逻辑**：引用 `Doc.md` 第 3 章的语义/词法融合与判定阈值，更新 `src/service/utils.py` 等模块中的规则时务必同步文档说明。
- **报告结构**：生成的查重报告字段需覆盖 `Doc.md` 第 4 章列出的总体指标、Top-N 源文献与句子级详情。
- **评测流程**：使用 `Doc.md` 第 6 章定义的指标与基准数据集运行 `src/benchmark.py`，在提交结果或上线前记录 Precision/Recall/F1 与延迟分位数。
- **部署准则**：部署在线服务时参考 `Doc.md` 第 5 章 API 定义与任务流转约定，保证接口兼容性与可观测性。

---

## 离线数据处理与索引构建

离线流程负责从 PDF/TeX 文件中提取句子文本，生成向量并写入 Qdrant 集合。主入口为 `src/pipeline.py`（CLI）。

### 基本用法

```bash
python -m src.pipeline \
	--input data/pdf_extracted \            # 原始论文目录
	--collection copyless-main \            # Qdrant 集合名称
	--model models/Qwen3-0.6B \              # 句向量模型
	--device cuda \                           # 默认使用 GPU
	--batch-size 256 \                       # 编码批大小
	--upsert-batch 1024 \                    # Qdrant upsert 批大小
	--sentence-splitter nltk \               # 分句策略
	--workers 4 \                            # 抽取/分句线程数
	--qdrant-url http://localhost:6333       # 若使用远程 Qdrant
```

> **快速自检 (Doc.md 2.1 对齐)**：在挂载数据后，可先抽取一小批 `.gz` 样本并使用 `dummy` 模型做 dry-run，验证抽取/分句/嵌入流程；示例命令：
>
> ```bash
> source .venv/bin/activate
> python -m src.pipeline \
>   --input /mnt/data/user/shao_bing/sample_unpacked_small \
>   --collection copyless-test \
>   --model dummy \
>   --sentence-splitter nltk \
>   --workers 1 \
>   --dry-run \
>   --dump outputs/sample_test.jsonl
> ```
>
> 运行完成后，检查终端统计与 `outputs/sample_test.jsonl` 中的 payload/embedding 字段，确认与 `Doc.md` 章节 3、4 的字段定义一致。

### 常用选项说明

| 参数 | 说明 |
| --- | --- |
| `--input` (必填) | 包含 `.pdf` 或 `.tex` 的根目录，会递归扫描。
| `--collection` (必填) | 写入的 Qdrant 集合名，自动新建或复用。
| `--model` | 句向量模型名称或路径。默认指向仓库内置模型。
| `--batch-size` | 单次送入模型的句子数量，需平衡显存/内存。
| `--upsert-batch` | 向 Qdrant 批量 upsert 的大小。
| `--sentence-splitter` | `nltk` / `spacy` / `mixed`。详见下节。
| `--spacy-model` | 使用 spaCy 分句时的模型名称。
| `--workers` | 并行抽取和切句的线程数。设为 1 时串行。
| `--dry-run` | 仅执行抽取+嵌入，不连接 Qdrant，可配合 `--dump` 查看输出。
| `--dump` | 指定 JSONL 输出路径，记录每个句子的 embedding 与元数据。
| `--comparison-collection` | 可选，生成一个对比集合（例如不同模型的向量）。
| `--comparison-model` | 对比集合使用的模型。缺省时与主模型一致。
| `--qdrant-path` | 指向本地嵌入式 Qdrant 的数据目录。
| `--recreate` | 若集合已经存在，强制重建（谨慎使用）。
| `--device` | 指定嵌入模型的 PyTorch 设备，如 `cuda`, `cuda:0`, `cpu`。默认值为 `cuda`（若存在 GPU）。 |
| `--device-map` | 可选，指定 transformers `device_map` 策略（如 auto、balanced），需安装 accelerate。 |

> **数据同步小工具**：如果需要从 `ceph` 挂载的 `/mnt/data/corpus/DocAI/arXiv_2023-12/{src,pdf}` 持续下载原始数据，可使用 `scripts/sync_arxiv_data.py`。例如：
>
> ```bash
> python scripts/sync_arxiv_data.py --components src pdf
> ```
> 该脚本基于 `rsync --append-verify`，支持断点续传和 `--dry-run` 预览。

> **多 GPU 编码小贴士**：安装 `accelerate` 后可通过 `--device cuda --device-map auto` 自动将 Qwen 模型分配到多块 GPU；或指定 `--device cuda:0` 固定在单卡运行。

向量写入时，每条记录的 payload 默认包含：

- `path`：原始文件的绝对/相对路径；
- `paper_id`：根据文件名猜测的 arXiv ID（若可解析）；
- `sent_index`：句子在全文中的顺序编号；
- `char_start` / `char_end`：句子在清洗后文本中的字符区间；
- `text`：句子原文；
- `embedding_model`：生成该向量的模型名称。

执行结束后，CLI 会输出一个 JSON 总结，包括处理文件数、句子数、平均时延等指标。例如：

```json
{
	"files_processed": 120,
	"sentences_processed": 48567,
	"dim_primary": 4096,
	"primary_model": "models/Qwen3-0.6B",
	"comparison": {
		"collection": null,
		"model": null
	},
	"latency_ms": {
		"extract_avg": 142.73,
		"encode_primary_avg": 58.31,
		"encode_comparison_avg": 0.0
	},
	"dry_run": false
}
```

---

## 核心脚本与模块用法

### `src/extract.py`

- `extract_text(path: Path) -> Optional[str]`：根据文件后缀自动调用 PDF 或 TeX 抽取。
- `extract_text_from_pdf`：依赖 PyMuPDF，逐页提取纯文本。
- `extract_text_from_tex`：使用 `pylatexenc` 转换为纯文本；若未安装则回退到正则清理。
- **建议拓展**：
	- 增加异常日志、解析失败重试；
	- 针对复杂公式/表格做定制清洗；
	- 支持更多文件类型（如 HTML、DOCX）。

### `src/preprocess.py`

- `clean_text`：统一空白符、标准化换行、去除制控制字符。
- `sentences_nltk` / `sentences_spacy`：基于 NLTK Punkt 或 spaCy pipeline 的分句。
- `split_mixed_sentences`：针对中英文混排的启发式分句。
- `get_sentence_splitter`：根据配置名称返回对应的分句函数。
- `chunk_iter`：通用的批量迭代器，可供后续模块复用。
- **待完善**：
	- 引用标记定位与上下文窗口扫描；
	- 参考文献解析并建立映射。

### `src/embedding.py`

- `Embedder`：封装 Qwen3-0.6B（默认）或其他指定模型的推理逻辑，针对 GPU 环境做了优化。
	- 支持 `batch_size`、`max_length`、L2 归一化。
	- 仍保留 `model_name="dummy"` 作为单元测试/离线验证的轻量模式，但部署时建议始终使用 Qwen 模型。
- **实用方法**：
	- `dim` 属性：获取模型输出维度。
	- `encode(sentences)`：输入字符串可迭代对象，返回 List[List[float]]。
- **扩展方向**：增加量化、模型缓存、多 GPU 支持。

### `src/qdrant_io.py`

- `ensure_collection`：根据向量维度与 HNSW 参数创建/重建集合。
- `upsert_points`：支持自定义批量大小的向量写入。
- `search_points` & `batch_search`：对单向量或批量执行余弦/点积/欧氏检索。
- **建议改进**：
	- 暴露副本、副本因子、量化配置；
	- 编写 schema 验证，确保 payload 字段统一；
	- 支持过滤器、按论文 ID 的分片策略。

### `src/hybrid_search.py`

- `ensure_hybrid_collection`：创建带有命名稠密/稀疏向量的集合，可选内存/磁盘模式。
- `build_hybrid_points`：利用 `Embedder` 生成稠密向量，并基于哈希词袋构建稀疏向量，返回可直接写入 Qdrant 的 `PointStruct`。
- `run_rrf_hybrid_query` / `rerank_with_dense`：封装 Query API，先融合稀疏/稠密候选，再使用稠密向量进行重排。
- `text_to_sparse_vector`：轻量哈希词袋编码器，无需额外依赖，适合快速原型验证。

### `src/pipeline.py`

- CLI 入口，整合抽取、预处理、嵌入、Qdrant 写入、可选 JSONL 导出。
- 支持比较模型、dry-run、日志配置、并行。
- 输出处理统计信息。
- **待办**：
	- 将引用、段落上下文等额外元数据纳入 payload；
	- 与在线查重服务打通（例如记录任务进度、写入 Kafka 等）。

### `src/service/`

- `config.py`：统一管理服务配置（Qdrant 连接、阈值、批大小、任务队列设置等），支持环境变量覆盖。
- `models.py`：基于 Pydantic 的入参、出参与任务状态模型，覆盖任务生命周期与报告结构。
- `tasks.py`：简易的内存/可选磁盘持久化任务队列，实现提交、获取、完成、失败状态流转。
- `retrieval.py`：封装句子预处理、嵌入编码、Qdrant 检索逻辑。
- `utils.py`：提供归一化编辑距离、判定决策树、加权得分及总体相似度等通用函数。
- `report.py`：聚合句子级结果，生成文档级综述、Top 源文献贡献度与详细列表。
- `worker.py`：异步后台 worker，按任务逐句检索、判定、生成报告并执行可选回调。
- `api.py`：FastAPI 入口，暴露异步提交 (`POST /v1/papers/check`) 与结果查询 (`GET /v1/reports/{task_id}`)。
- 运行服务：

```bash
uvicorn src.service.api:app --host 0.0.0.0 --port 8080 --reload
```

- 示例请求：

```bash
curl -X POST http://localhost:8080/v1/papers/check \
	-H 'Content-Type: application/json' \
	-d '{"content": "Your paper text..."}'

curl http://localhost:8080/v1/reports/<task_id>
```

### `src/benchmark.py`

- 提供 `sentences` 与 `documents` 两个子命令，分别针对句级、文档级基准进行评测。
- 自动统计精度/召回/F1、延迟分位数、编码/检索吞吐量，可选 in-memory 或 Qdrant 后端。
- 可用 JSONL 数据驱动评估：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.benchmark sentences --data data/bench/sentences.jsonl --model models/Qwen3-0.6B --device cuda --threshold 0.85
CUDA_VISIBLE_DEVICES=0 python -m src.benchmark documents --data data/bench/docs.jsonl --model models/Qwen3-0.6B --device cuda
```

- 评测结果可用于阈值调优、不同模型对比及部署前验收。

---

## 在线查重服务

该仓库已提供 FastAPI + 背景 worker 的最小可用在线查重服务原型：

1. 启动服务（确保使用虚拟环境的解释器）：

```bash
source .venv/bin/activate  # 如果尚未激活虚拟环境
uvicorn src.service.api:app --host 0.0.0.0 --port 8080
```

> 若不想激活虚拟环境，可直接使用绝对路径：

```bash
./.venv/bin/uvicorn src.service.api:app --host 0.0.0.0 --port 8080
```

2. 提交任务：

```bash
curl -X POST http://localhost:8080/v1/papers/check \
	-H 'Content-Type: application/json' \
	-d '{"content": "This is a demo paper."}'
```

3. 轮询结果：`curl http://localhost:8080/v1/reports/<task_id>`

4. 可通过环境变量或 `.env` 覆盖 `COPYLESS_QDRANT_URL`、`COPYLESS_EMBEDDING_MODEL`、`COPYLESS_TOP_K` 等配置；任务队列支持内存或磁盘缓存。

服务流程包括句子预处理→向量检索→语义/词法融合判定→引用占位符→报告聚合→可选回调，便于后续接入 UI 或扩展精排模型。

5. 内置网页测试：访问 `http://localhost:8080/` 将打开一个简洁的 Web Demo，可直接在浏览器中粘贴文本、提交查重并实时查看 JSON 报告。

### 系统 API 设计

在线查重服务遵循 RESTful 与无状态设计，提供异步任务接口，便于长耗时论文处理。

#### 5.1 异步查重接口

- **Endpoint**：`POST /v1/papers/check`
- **功能**：提交一篇论文进行查重，立即返回任务 ID。
- **Request Body (JSON)**：

```json
{
	"content": "The full text of the paper...",
	"callback_url": "https://your-service.com/webhook/notify"
}
```

`callback_url` 为可选字段，用于任务完成后的回调通知。

- **成功响应 (202 Accepted)**：

```json
{
	"task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
	"status": "pending",
	"message": "Your paper has been queued for checking. Use the task_id to poll for results."
}
```

#### 5.2 结果查询接口

- **Endpoint**：`GET /v1/reports/{task_id}`
- **功能**：根据任务 ID 查询查重结果。
- **成功响应 (200 OK)**：
	- 若任务仍在处理：

```json
{
	"task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
	"status": "processing"
}
```

	- 若任务已完成：

```json
{
	"task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
	"status": "completed",

#### 5.3 基准评测接口

- **Endpoint**：`POST /v1/benchmarks/run`
- **功能**：将句级或文档级评测任务排入队列，后台 worker 会复用 `src/benchmark.py` 计算精度、召回、延迟等指标。
- **Request Body (JSON)**：

```json
{
	"kind": "sentences",
	"data_path": "/abs/path/to/sent_bench.jsonl",
	"model": "models/Qwen3-0.6B",
	"threshold": 0.82,
	"backend": "inmem",
	"callback_url": "https://your-service.com/hooks/bench"
}
```

`kind` 支持 `sentences` 与 `documents`。`data_path` 必须可被服务访问。除非显式设定，评测任务默认关闭 `tqdm` 进度条（`show_progress=false`），避免占满日志，可在请求中覆盖参数。

- **响应 (202 Accepted)**：与查重接口相同，返回 `task_id`，后续通过 `/v1/reports/{task_id}` 查询。

- **完成态示例**（`benchmark` 字段新增在 `TaskStatusResponse` 内）：

```json
{
	"task_id": "0b3ef35d-5dec-4e70-9c9f-28fd02d5a1f8",
	"status": "completed",
	"report": null,
	"benchmark": {
		"kind": "sentences",
		"data_path": "/abs/path/to/sent_bench.jsonl",
		"config": {
			"backend": "inmem",
			"batch_size": 256,
			"collection": "copyless_bench_tmp",
			"doc_min_pairs": 3,
			"doc_min_ratio": 0.05,
			"model": "models/Qwen3-0.6B",
			"qdrant_api_key": null,
			"qdrant_url": "http://localhost:6333",
			"show_progress": false,
			"sim_threshold": 0.8
		},
		"result": {
			"metrics": {
				"precision": 0.91,
				"recall": 0.88,
				"f1": 0.89
			},
			"throughput": {
				"encode_sents_per_sec": 1350.2,
				"query_per_sec": 1180.5
			}
		}
	}
}
```

若配置了 `callback_url`，任务完成后同样会推送包含 `benchmark` 字段的完整任务状态。
	"report": {
		"overall_similarity_score": 0.235,
		"summary": {
			"total_sentences": 500,
			"identical_count": 20,
			"minor_changes_count": 45,
			"paraphrased_count": 58
		},
		"top_sources": [
			{
				"paper_id": "arXiv:2401.12345",
				"similarity_score": 0.152,
				"url": "https://arxiv.org/abs/2401.12345"
			},
			{
				"paper_id": "arXiv:2308.06789",
				"similarity_score": 0.083,
				"url": "https://arxiv.org/abs/2308.06789"
			}
		],
		"sentence_details": [
			{
				"original_sentence": "This method achieves state-of-the-art results.",
				"status": "minor_changes",
				"similarity_score": 0.96,
				"matched_source": {
					"sentence": "Our approach obtains state-of-the-art performance.",
					"paper_id": "arXiv:2401.12345"
				}
			},
			{
				"original_sentence": "The experiment was conducted in a controlled environment.",
				"status": "original",
				"similarity_score": 0.55,
				"matched_source": null
			}
		]
	}
}
```

---

## 性能评估与基准测试

利用 `src/benchmark.py`，可快速对不同模型、阈值与检索后端进行离线评估，并输出：

- 句级/文档级 Precision、Recall、F1；
- 编码与检索的平均延迟、P95/P99；
- 吞吐量（句/秒、查询/秒）；
- 可扩展至 PAN、arXiv 自建样本等基准数据集。

建议维护一个持续集成的基准套件：每次模型或阈值更新前运行评测，确保召回与准确率稳定，并据此调整决策树阈值或 rerank 策略。

更多实践要点：

1. **准确性指标**：Precision / Recall / F1（句级与文档级）。
2. **性能指标**：平均延迟、P95/P99、吞吐量（句/秒或请求/秒）。
3. **数据集**：
	 - PAN-PC-11、PAN-2013/2014 等公开抄袭检测数据集；
	 - 自建 arXiv 子集用于快速回归测试。
4. **评测流程建议**：
	 - 编写 `src/benchmark.py` 的 `evaluate` 子命令，接收数据集路径、候选 Top-K、阈值配置；
	 - 输出详细的混淆矩阵、ROC 曲线数据、最优阈值建议；
	 - 对在线服务进行压测（如 locust、wrk）收集延迟与吞吐。

---

## 在线查重服务设计草案

> 目前 FastAPI 服务提供了最小可用版本，以下设计草案聚焦于后续增强（精排、引用识别、持久化任务系统等）。

1. **REST API**：
	 - `POST /v1/papers/check`：提交论文内容，返回任务 ID。
	 - `GET /v1/reports/{task_id}`：查询任务状态与最终报告。
	 - 可使用 FastAPI / Flask 构建，配合 Redis / PostgreSQL 存储任务状态。

2. **处理流程**：
	 - 将提交的论文按 `preprocess` 模块进行清洗与分句。
	 - 通过 `Embedder` 编码，调用 `qdrant_io.search_points` 获取 Top-K 候选。
	 - 对候选句子执行编辑距离（Levenshtein）与翻译回溯比对，结合阈值判定等级。
	 - 根据引用识别结果将 Minor/Paraphrased 纠正为 Cited。

3. **结果聚合**：
	 - 统计句级结果，按照权重计算整体重复率：
		 $$ Score = \frac{N_i \cdot 1.0 + N_m \cdot 0.8 + N_p \cdot 0.6}{N_{total}} $$
	 - 汇总 Top-N 来源文献及各自贡献率。
	 - 生成报告 JSON 及可视化 HTML。

4. **扩展建议**：
	 - 在 `outputs/` 中缓存中间结果，便于调试；
	 - 支持 webhook 通知（`callback_url`）。

---

## 常见问题与排错

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `ModuleNotFoundError: fitz` | PyMuPDF 未安装 | `pip install pymupdf`，或检查虚拟环境。 |
| `RuntimeError: spaCy 模型未安装` | 未下载 `en_core_web_sm` | 运行 `python -m spacy download en_core_web_sm`。 |
| Qdrant 连接失败 | 服务未启动或防火墙限制 | 确认 Qdrant 服务地址/端口，或使用 `--qdrant-path` 启动嵌入式模式。 |
| CUDA OOM | 批量过大或模型太大 | 减小 `--batch-size`，或在 CPU 上运行。 |
| dry-run 输出为空 | 文件无文本或分句器被禁用 | 检查输入文件格式，确认 `extract_text` 是否成功。 |

---

## 后续路线图

1. **在线服务实现**：基于 FastAPI + Celery/Redis 搭建异步任务处理。
2. **引用解析**：实现参考文献提取、引用标注匹配与 Cited 判定逻辑。
3. **二阶段精排**：集成 Levenshtein、翻译回译、依存句法等特征。
4. **报告生成**：构建交互式 HTML 报告（可参考 Copyleaks 展示）。
5. **评测框架**：补全 `src/benchmark.py`，自动化运行 PAN 数据集评估。
6. **部署与监控**：提供 Dockerfile、Helm Chart，与 Prometheus/Grafana 监控。

---

## 参考资料

- [Qdrant 官方文档](https://qdrant.tech/documentation/)
- [Copyleaks 报告示例](https://app.copyleaks.com/dashboard/v1/account/scans)
- [PAN Plagiarism Corpus](https://pan.webis.de/data.html)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/index)

如需进一步的实现或扩展协助，欢迎继续提问。
