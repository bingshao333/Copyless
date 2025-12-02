from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPYLESS_", env_file=Path(".env"), extra="allow")

    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: Optional[Path] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "copyless_full"
    qdrant_timeout: Optional[int] = Field(default=120, description="Qdrant 超时时长 (秒)")
    qdrant_prefer_grpc: bool = False

    # Embedding model
    embedding_model: str = Field(default=str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B"))
    embedding_batch_size: int = 128

    # Sentence splitting
    sentence_splitter: str = "nltk"  # nltk | spacy | mixed
    spacy_model: str = "en_core_web_sm"

    # Retrieval parameters
    top_k: int = 5
    score_threshold: float = 0.7
    rerank_top_k: int = 5

    # Thresholds for classification
    t_lev_high: float = 0.99
    t_lev_med: float = 0.9
    t_cos_high: float = 0.95
    t_cos_mid: float = 0.88

    # Citation detection
    citation_window: int = 2

    # Task execution
    worker_count: int = 2
    max_queue_size: int = 32

    # Storage for task states (simple in-memory fallback)
    use_disk_cache: bool = False
    cache_dir: Path = Field(default=Path("storage/tasks"))

    # Benchmark defaults
    benchmark_sentence_threshold: float = 0.8
    benchmark_doc_threshold: float = 0.8

    # API limits
    max_content_length: int = 2_000_000


@functools.lru_cache(maxsize=1)
def get_settings() -> ServiceSettings:
    return ServiceSettings()
