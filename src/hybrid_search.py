"""Utilities for building hybrid search queries with Qdrant's Query API.

This module provides helpers to:

1. Create or reset a Qdrant collection configured with both dense and sparse vectors.
2. Convert raw text into deterministic sparse bag-of-words representations suitable for
   Qdrant sparse vectors.
3. Populate a collection with demo documents that include dense and sparse embeddings.
4. Execute hybrid queries that combine dense and sparse retrieval via the Query API.

It relies on the repository's :class:`~src.embedding.Embedder` abstraction to produce dense
representations. For sparse vectors, a lightweight hashed bag-of-words encoder is used so
no external dependency (e.g. scikit-learn) is required.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .embedding import Embedder

TOKEN_PATTERN = re.compile(r"[\w']+")
DEFAULT_HASH_MODULO = 2 ** 20  # 约 100 万维度，可覆盖 demo 需求


@dataclass
class HybridDocument:
    """Holds the minimal data required to index a document for hybrid search."""

    doc_id: str
    text: str
    metadata: Mapping[str, object]


@dataclass
class HybridQueryResult:
    """Convenience wrapper for a Query API response point."""

    point_id: str
    score: float
    payload: Mapping[str, object]


def _hash_token(token: str, modulo: int = DEFAULT_HASH_MODULO) -> int:
    h = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:12], 16) % modulo


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def text_to_sparse_vector(text: str, modulo: int = DEFAULT_HASH_MODULO) -> qm.SparseVector:
    tokens = tokenize(text)
    if not tokens:
        return qm.SparseVector(indices=[], values=[])

    counts: dict[int, float] = {}
    for tok in tokens:
        idx = _hash_token(tok, modulo=modulo)
        counts[idx] = counts.get(idx, 0.0) + 1.0

    inv_doc_len = 1.0 / math.sqrt(len(tokens))
    indices, values = zip(*sorted(counts.items()))
    scaled_values = [v * inv_doc_len for v in values]
    return qm.SparseVector(indices=list(indices), values=scaled_values)


def ensure_hybrid_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    sparse_name: str = "bow",
    dense_name: str = "dense",
    reset: bool = False,
    on_disk: bool = False,
) -> None:
    if reset:
        try:
            client.delete_collection(collection)
        except Exception:
            pass

    try:
        client.get_collection(collection)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config={
            dense_name: qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
                on_disk=on_disk,
            )
        },
        sparse_vectors_config={sparse_name: qm.SparseVectorParams()},
    )


def build_hybrid_points(
    embedder: Embedder,
    documents: Sequence[HybridDocument],
    sparse_name: str = "bow",
    dense_name: str = "dense",
) -> List[qm.PointStruct]:
    texts = [doc.text for doc in documents]
    dense_vectors = embedder.encode(texts)

    points: List[qm.PointStruct] = []
    for doc, dense_vec in zip(documents, dense_vectors):
        try:
            uuid.UUID(str(doc.doc_id))
            point_id = str(doc.doc_id)
        except (ValueError, AttributeError, TypeError):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc.doc_id)))

        vector = {
            dense_name: list(dense_vec),
            sparse_name: text_to_sparse_vector(doc.text),
        }
        payload = {"text": doc.text, "doc_id": doc.doc_id, **doc.metadata}
        points.append(qm.PointStruct(id=point_id, vector=vector, payload=payload))
    return points


def upsert_documents(
    client: QdrantClient,
    collection: str,
    points: Sequence[qm.PointStruct],
) -> None:
    if not points:
        return
    client.upsert(collection_name=collection, points=points, wait=True)


def run_rrf_hybrid_query(
    client: QdrantClient,
    collection: str,
    dense_query: Sequence[float],
    sparse_query: qm.SparseVector,
    dense_limit: int = 20,
    sparse_limit: int = 20,
    fusion_limit: int = 10,
    dense_name: str = "dense",
    sparse_name: str = "bow",
    with_payload: bool = True,
) -> List[HybridQueryResult]:
    response = client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=list(dense_query), using=dense_name, limit=dense_limit),
            qm.Prefetch(query=sparse_query, using=sparse_name, limit=sparse_limit),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=fusion_limit,
        with_payload=with_payload,
    )

    results: List[HybridQueryResult] = []
    for point in response.points:
        results.append(
            HybridQueryResult(point_id=str(point.id), score=point.score, payload=point.payload or {})
        )
    return results


def rerank_with_dense(
    client: QdrantClient,
    collection: str,
    dense_query: Sequence[float],
    candidate_prefetch: Sequence[qm.Prefetch],
    limit: int = 10,
    dense_name: str = "dense",
    with_payload: bool = True,
) -> List[HybridQueryResult]:
    response = client.query_points(
        collection_name=collection,
        prefetch=list(candidate_prefetch),
        query=list(dense_query),
        using=dense_name,
        limit=limit,
        with_payload=with_payload,
    )

    results: List[HybridQueryResult] = []
    for point in response.points:
        results.append(
            HybridQueryResult(point_id=str(point.id), score=point.score, payload=point.payload or {})
        )
    return results


def build_prefetch_branches(
    dense_query: Sequence[float],
    sparse_query: qm.SparseVector,
    dense_name: str = "dense",
    sparse_name: str = "bow",
    dense_limit: int = 50,
    sparse_limit: int = 50,
) -> Tuple[qm.Prefetch, qm.Prefetch]:
    dense_branch = qm.Prefetch(query=list(dense_query), using=dense_name, limit=dense_limit)
    sparse_branch = qm.Prefetch(query=sparse_query, using=sparse_name, limit=sparse_limit)
    return dense_branch, sparse_branch