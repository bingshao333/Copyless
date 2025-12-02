from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SentenceStatus(str, Enum):
    IDENTICAL = "identical"
    MINOR_CHANGES = "minor_changes"
    PARAPHRASED = "paraphrased"
    CITED = "cited"
    ORIGINAL = "original"
    ERROR = "error"


class MatchedSource(BaseModel):
    sentence: str = Field(..., description="匹配到的源句文本")
    paper_id: Optional[str] = Field(None, description="源论文 ID，例如 arXiv 标识")
    archive_path: Optional[str] = Field(None, description="源归档所在路径")
    member: Optional[str] = Field(None, description="源文件中的成员名称/路径")
    sent_index: Optional[int] = Field(None, description="源文件内的句子索引")
    similarity_score: float = Field(..., description="语义相似度得分")
    lexical_score: float = Field(..., description="词法/编辑距离相似度")


class SentenceReport(BaseModel):
    index: int
    text: str
    status: SentenceStatus
    similarity_score: float = Field(..., description="综合相似度得分（0-1）")
    semantic_score: float = Field(..., description="语义相似度得分（0-1）")
    lexical_score: float = Field(..., description="词法相似度得分（0-1）")
    has_citation: bool = Field(False, description="上下文是否存在引用对应源文献")
    matched_source: Optional[MatchedSource] = None
    errors: Optional[List[str]] = None


class ReportSummary(BaseModel):
    total_sentences: int
    identical_count: int
    minor_changes_count: int
    paraphrased_count: int
    cited_count: int
    original_count: int


class SourceContribution(BaseModel):
    paper_id: Optional[str]
    score: float
    sentence_count: int
    weight: float
    extra: Dict[str, str] = Field(default_factory=dict)


class ReportPayload(BaseModel):
    overall_similarity_score: float
    summary: ReportSummary
    top_sources: List[SourceContribution]
    sentence_details: List[SentenceReport]


class BenchmarkKind(str, Enum):
    SENTENCES = "sentences"
    DOCUMENTS = "documents"


class BenchmarkRequest(BaseModel):
    kind: BenchmarkKind
    data_path: str = Field(..., description="Absolute path to benchmark JSONL dataset")
    model: Optional[str] = Field(None, description="Override embedding model")
    batch_size: Optional[int] = Field(None, description="Override embedding batch size")
    threshold: Optional[float] = Field(None, description="Similarity threshold")
    backend: Optional[str] = Field(None, description="Search backend, inmem or qdrant")
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    collection: Optional[str] = None
    doc_min_pairs: Optional[int] = Field(None, description="Document min matching sentence count")
    doc_min_ratio: Optional[float] = Field(None, description="Document min matching ratio")
    callback_url: Optional[str] = Field(None, description="Optional callback to receive results")
    show_progress: Optional[bool] = Field(None, description="Enable tqdm progress logging")


class BenchmarkPayload(BaseModel):
    kind: BenchmarkKind
    data_path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any]


class CheckRequest(BaseModel):
    content: str
    callback_url: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class CheckAcceptedResponse(BaseModel):
    task_id: str
    status: Literal["pending"] = "pending"
    message: str = "Your paper has been queued for checking."


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskState(BaseModel):
    task_id: str
    status: TaskStatus
    submitted_at: dt.datetime
    started_at: Optional[dt.datetime] = None
    completed_at: Optional[dt.datetime] = None
    report: Optional[ReportPayload] = None
    error: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    benchmark: Optional[BenchmarkPayload] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    report: Optional[ReportPayload] = None
    error: Optional[str] = None
    submitted_at: dt.datetime
    started_at: Optional[dt.datetime] = None
    completed_at: Optional[dt.datetime] = None
    benchmark: Optional[BenchmarkPayload] = None