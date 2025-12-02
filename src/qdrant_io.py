from __future__ import annotations

from typing import List, Dict, Any, Optional, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    distance: str = "Cosine",
    on_disk: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construct: int = 512,
    hnsw_full_scan_threshold: int = 1000,
    force_recreate: bool = False,
):
    if not force_recreate:
        try:
            client.get_collection(collection)
            return
        except Exception:
            pass

    # 兼容大小写输入
    dist_name = distance.upper()
    if not hasattr(qm.Distance, dist_name):
        raise ValueError(f"Unsupported distance '{distance}'. Use one of: COSINE, DOT, EUCLID")
    dist = getattr(qm.Distance, dist_name)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=dist, on_disk=on_disk),
        hnsw_config=qm.HnswConfigDiff(
            m=hnsw_m,
            ef_construct=hnsw_ef_construct,
            full_scan_threshold=hnsw_full_scan_threshold,
        ),
    )


def upsert_points(
    client: QdrantClient,
    collection: str,
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
    batch_size: int = 1024,
):
    assert len(vectors) == len(payloads)
    if ids is not None:
        assert len(ids) == len(vectors)

    n = len(vectors)
    for i in range(0, n, batch_size):
        batch_vecs = vectors[i : i + batch_size]
        batch_payloads = payloads[i : i + batch_size]
        if ids is not None:
            batch_ids = ids[i : i + batch_size]
            batch = qm.Batch(ids=cast(List[qm.ExtendedPointId], batch_ids), vectors=batch_vecs, payloads=batch_payloads)
        else:
            batch = qm.Batch(vectors=batch_vecs, payloads=batch_payloads)
        client.upsert(
            collection_name=collection,
            points=batch,
            wait=True,
        )


def search_points(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    top_k: int = 5,
    with_payload: bool = True,
    with_vectors: bool = False,
    query_filter: Optional[qm.Filter] = None,
    hnsw_ef: int = 128,
):
    """执行单向量检索，返回 Qdrant search 结果列表。"""
    return client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=with_payload,
        with_vectors=with_vectors,
        query_filter=query_filter,
        search_params=qm.SearchParams(hnsw_ef=hnsw_ef),
    )


def batch_search(
    client: QdrantClient,
    collection: str,
    query_vectors: List[List[float]],
    top_k: int = 5,
    with_payload: bool = True,
    hnsw_ef: int = 128,
):
    """批量检索：对多条向量逐一调用 search。简单串行实现。"""
    out = []
    for v in query_vectors:
        out.append(
            client.search(
                collection_name=collection,
                query_vector=v,
                limit=top_k,
                with_payload=with_payload,
                search_params=qm.SearchParams(hnsw_ef=hnsw_ef),
            )
        )
    return out
