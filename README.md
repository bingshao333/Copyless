# Copyless — Semantic Plagiarism Detection for Academic Papers

> A deep plagiarism detection system that combines large-scale sentence vector indexing with multi-stage semantic/lexical fusion to identify copying, paraphrasing, and citation behaviors in academic papers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.10%2B-dc382c.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Key Features

- **Sentence-Level Deep Detection** — Goes beyond keyword matching by performing semantic similarity analysis on every sentence, catching paraphrasing and subtle rewording.
- **Hybrid Retrieval (Dense + Sparse)** — Combines Qwen3-0.6B dense embeddings with hashed bag-of-words sparse vectors, fused via Reciprocal Rank Fusion (RRF) for high recall.
- **Multi-Stage Judgment Pipeline** — Two-stage retrieval → reranking → decision tree classification using both cosine similarity and normalized Levenshtein distance.
- **Smart Citation Awareness** — Automatically parses reference sections and performs context-window citation scanning to distinguish legitimate citations from plagiarism.
- **Async Online Service** — Production-ready FastAPI service with async task queue, background workers (thread pool executor for CPU-bound tasks), and webhook callbacks.
- **Comprehensive Benchmarking** — Built-in sentence-level and document-level evaluation framework with Precision/Recall/F1 metrics, latency profiling, and throughput measurement.

---

## 🏗️ System Architecture

```
┌────────────────────────── Offline Pipeline ──────────────────────────┐
│                                                                       │
│  PDF/LaTeX  ──→  Text Extraction  ──→  Clean & Segment  ──→  Encode  │
│  (PyMuPDF)      (extract.py)         (preprocess.py)     (Qwen3-0.6B)│
│                                                                │      │
│                                               Dense + Sparse Vectors  │
│                                                                │      │
│                                                     ┌──────────▼──┐   │
│                                                     │   Qdrant    │   │
│                                                     │   Vector DB │   │
│                                                     └──────┬──────┘   │
└────────────────────────────────────────────────────────────────────────┘
															 │
┌────────────────────────── Online Service ─────────────────────────────┐
│                                                                       │
│  User Paper  ──→  FastAPI  ──→  Task Queue  ──→  Async Workers       │
│  (POST /v1/      (api.py)      (tasks.py)       (worker.py)          │
│   papers/check)                                       │               │
│                                    ┌──────────────────┘               │
│                                    ▼                                  │
│                           ┌────────────────┐                          │
│                           │ For each sent:  │                          │
│                           │ 1. Encode       │                          │
│                           │ 2. Vector Search│                          │
│                           │ 3. Rerank Top-K │                          │
│                           │ 4. Levenshtein  │                          │
│                           │ 5. Decision Tree│                          │
│                           │ 6. Citation Scan│                          │
│                           └───────┬────────┘                          │
│                                   ▼                                   │
│                           Report Generation                           │
│                        (overall score + top                            │
│                         sources + details)                             │
│                                   │                                   │
│                          GET /v1/reports/{id}                          │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Embedding Model** | Qwen3-0.6B (mean pooling + L2 norm, FP16 on GPU) |
| **Vector Database** | Qdrant (HNSW index, cosine similarity, hybrid dense+sparse) |
| **Web Framework** | FastAPI + Uvicorn (async lifespan, background workers) |
| **Text Extraction** | PyMuPDF (PDF), pylatexenc (LaTeX), regex fallback |
| **NLP Tokenization** | NLTK Punkt, spaCy, custom mixed CJK/EN splitter |
| **Similarity Metrics** | Cosine similarity (semantic) + Normalized Levenshtein (lexical) |
| **Fusion Strategy** | Reciprocal Rank Fusion (RRF) for hybrid retrieval |
| **Task Queue** | In-memory async queue with TTL-based cleanup |
| **Benchmarking** | Custom framework: Precision/Recall/F1, P95/P99 latency |

---

## 📁 Project Structure

```
Copyless/
├── src/
│   ├── pipeline.py          # Offline batch processing CLI (extract → encode → index)
│   ├── extract.py           # PDF/LaTeX text extraction with error handling
│   ├── preprocess.py        # Text cleaning, sentence segmentation (NLTK/spaCy/mixed)
│   ├── embedding.py         # Qwen3-0.6B sentence encoder (GPU, batched, L2 norm)
│   ├── qdrant_io.py         # Qdrant collection management & batch vector operations
│   ├── hybrid_search.py     # Dense+Sparse hybrid retrieval with RRF fusion
│   ├── benchmark.py         # Sentence-level & document-level evaluation framework
│   ├── metrics.py           # Precision/Recall/F1 & latency statistics
│   └── service/
│       ├── api.py           # FastAPI endpoints (async submit + result polling)
│       ├── config.py        # Unified config via pydantic-settings (.env support)
│       ├── models.py        # Pydantic schemas for requests, reports, task states
│       ├── tasks.py         # In-memory task queue with TTL cleanup
│       ├── retrieval.py     # Retrieval pipeline (preprocess → encode → search)
│       ├── worker.py        # Async background workers (thread pool for CPU tasks)
│       ├── citations.py     # Reference parsing & context-window citation detection
│       ├── report.py        # Report aggregation (overall score, top sources)
│       ├── utils.py         # Levenshtein distance, decision tree, weighted scoring
│       └── templates/       # Web UI template for interactive demo
├── scripts/                 # Data sync & environment setup utilities
├── requirements.txt         # Pinned dependencies with version bounds
├── Doc.md                   # Technical specification document
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) spaCy model for sentence splitting
python -m spacy download en_core_web_sm
```

### 2. Offline Indexing Pipeline

Build the sentence vector index from academic papers:

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

**Dry-run mode** (no Qdrant required):

```bash
python -m src.pipeline \
	--input data/sample \
	--collection test \
	--model dummy \
	--dry-run \
	--dump outputs/sample.jsonl
```

### 3. Start Online Detection Service

```bash
uvicorn src.service.api:app --host 0.0.0.0 --port 8080
```

**Submit a paper:**
```bash
curl -X POST http://localhost:8080/v1/papers/check \
	-H 'Content-Type: application/json' \
	-d '{"content": "Your paper text...", "callback_url": "https://example.com/hook"}'
```

**Poll results:**
```bash
curl http://localhost:8080/v1/reports/<task_id>
```

**Web Demo:** Visit `http://localhost:8080/` for an interactive browser-based UI.

### 4. Run Benchmarks

```bash
# Sentence-level evaluation
python -m src.benchmark sentences \
	--data data/bench/sentences.jsonl \
	--model models/Qwen3-0.6B \
	--device cuda --threshold 0.85

# Document-level evaluation
python -m src.benchmark documents \
	--data data/bench/docs.jsonl \
	--model models/Qwen3-0.6B \
	--device cuda
```

---

## 🧠 Core Algorithms

### Similarity Judgment Decision Tree

The system uses a **rule-based decision tree** that combines semantic (cosine) and lexical (Levenshtein) signals for explainable classification:

```
IF    Sim_lev ≥ T_lev_high (0.99)      →  Identical
ELIF  Sim_lev ≥ T_lev_med  (0.90)
	  AND Sim_cos ≥ T_cos_high (0.95)  →  Minor Changes
ELIF  Sim_cos ≥ T_cos_mid  (0.88)      →  Paraphrased
ELSE                                    →  Original

// Post-processing: if classified as Minor Changes or Paraphrased
// AND context window contains citation to the matched source
//    →  Override to Cited
```

### Weighted Similarity Score

```
Score_final = 0.7 × Sim_cosine + 0.3 × Sim_levenshtein
```

### Overall Document Similarity

```
Score = (N_identical × 1.0 + N_minor × 0.8 + N_paraphrased × 0.6) / N_total
```

### Citation Detection Pipeline

1. **Reference Section Parsing** — Extract bibliography, map labels `[1]`, `[Author 2025]` to paper IDs
2. **Inline Citation Localization** — Regex-based detection of citation markers in body text
3. **Context Window Scanning** — Check ±K sentences around flagged content for relevant citations
4. **Status Override** — Reclassify `minor_changes`/`paraphrased` → `cited` when citation is confirmed

---

## 📊 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/papers/check` | POST | Submit paper for plagiarism detection (async) |
| `/v1/reports/{task_id}` | GET | Poll task status and retrieve report |
| `/v1/benchmarks/run` | POST | Submit benchmark evaluation task |
| `/` | GET | Interactive Web Demo UI |

### Report Structure

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

## ⚙️ Configuration

All settings can be overridden via environment variables (prefix `COPYLESS_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `COPYLESS_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `COPYLESS_EMBEDDING_MODEL` | `models/Qwen3-0.6B` | Sentence embedding model |
| `COPYLESS_TOP_K` | `5` | Number of candidates per query |
| `COPYLESS_T_COS_HIGH` | `0.95` | High cosine similarity threshold |
| `COPYLESS_T_COS_MID` | `0.88` | Mid cosine similarity threshold |
| `COPYLESS_T_LEV_HIGH` | `0.99` | High Levenshtein similarity threshold |
| `COPYLESS_T_LEV_MED` | `0.90` | Medium Levenshtein similarity threshold |
| `COPYLESS_WORKER_COUNT` | `2` | Number of background workers |

---

## 📈 Benchmark Metrics

The evaluation framework measures:

- **Accuracy**: Sentence-level & document-level Precision / Recall / F1
- **Latency**: Average, P95, P99 for encoding and retrieval
- **Throughput**: Sentences/sec (encoding), Queries/sec (retrieval)
- **Backends**: In-memory (NumPy) or Qdrant for comparison

---

## 🗺️ Roadmap

- [ ] Cross-lingual detection via multilingual embeddings
- [ ] Formula & table similarity detection
- [ ] Fine-tuned sentence embedding model on plagiarism corpus
- [ ] Distributed task queue (Redis/Celery) for production scale
- [ ] Interactive HTML report with highlighted diff view
- [ ] Helm Chart & Docker Compose for one-click deployment

---

## 📖 References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PAN Plagiarism Corpus](https://pan.webis.de/data.html)
- [Qwen3 Model](https://huggingface.co/Qwen)
- [Copyleaks](https://copyleaks.com/) — Industry reference for report format

---

## License

MIT
