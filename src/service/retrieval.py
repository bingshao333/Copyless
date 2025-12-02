from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.embedding import Embedder
from src.preprocess import clean_text, sentences_nltk, sentences_spacy, split_mixed_sentences
from .config import ServiceSettings


def _resolve_splitter(settings: ServiceSettings):
    splitter = settings.sentence_splitter.lower()
    if splitter == "nltk":
        return sentences_nltk
    if splitter == "spacy":
        return lambda text: sentences_spacy(text, model=settings.spacy_model)
    if splitter == "mixed":
        return split_mixed_sentences
    raise ValueError(f"Unsupported sentence splitter: {settings.sentence_splitter}")


class RetrievalPipeline:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.embedder = Embedder(model_name=settings.embedding_model, batch_size=settings.embedding_batch_size)
        self.splitter = _resolve_splitter(settings)
        self.client = self._build_client(settings)

    @staticmethod
    def _build_client(settings: ServiceSettings) -> QdrantClient:
        if settings.qdrant_path is not None:
            return QdrantClient(path=str(settings.qdrant_path))
        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        if settings.qdrant_prefer_grpc:
            kwargs["prefer_grpc"] = True
        if settings.qdrant_timeout is not None:
            kwargs["timeout"] = settings.qdrant_timeout
        return QdrantClient(**kwargs)

    def preprocess(self, content: str) -> List[str]:
        text = clean_text(content)
        sentences = [s for s in self.splitter(text) if s]
        return sentences

    def encode_sentences(self, sentences: Sequence[str]) -> List[List[float]]:
        return self.embedder.encode(sentences)

    def search_similar(self, sentence_vec: Sequence[float], exclude_ids: Optional[List[str]] = None):
        filter_cond = None
        if exclude_ids:
            filter_cond = qm.Filter(must_not=[qm.HasIdCondition(has_id=exclude_ids)])
        return self.client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=sentence_vec,
            limit=self.settings.top_k,
            score_threshold=self.settings.score_threshold,
            with_payload=True,
            with_vectors=False,
            query_filter=filter_cond,
            search_params=qm.SearchParams(hnsw_ef=128),
        )

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        num = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return num / denom
