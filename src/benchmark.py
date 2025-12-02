from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, List, Optional, Sequence, Tuple

import click
from tqdm import tqdm

from .embedding import Embedder, _default_model_path
from .qdrant_io import ensure_collection, upsert_points
from .extract import extract_text
from .preprocess import clean_text, sentences_nltk, chunk_iter
from .metrics import confusion_from_sets, precision_recall_f1, latency_summary, throughput


# 输入格式约定
# 1) 句子级评测 JSONL：每行 {"id": str, "text": str, "dupes": [str, ...]}
#    其中 dupes 为与该句等价/重复的其他句子的 id 列表。评测会在全集内做最近邻匹配并阈值判断，
#    输出预测对 (anchor_id, matched_id)，与真值对进行比较。
# 2) 文档级评测 JSONL：每行 {"doc_id": str, "path": str}，将从 path 抽取文本、切句、编码，
#    利用句子级匹配聚合到文档层面：若跨文档匹配句子数 >= K 或占比 >= R，则判为重复文档对。


@dataclass
class BenchConfig:
    backend: str = "inmem"  # inmem | qdrant
    model: str = _default_model_path()
    batch_size: int = 256
    # 相似度阈值（余弦相似度），用于判定句子重复
    sim_threshold: float = 0.8
    # 文档级聚合阈值：至少匹配句子数
    doc_min_pairs: int = 3
    # 文档级聚合阈值：匹配句子占比（相对于较短文档句子数）
    doc_min_ratio: float = 0.05
    # Qdrant 选项
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection: str = "copyless_bench_tmp"
    show_progress: bool = True


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    num = float((va * vb).sum())
    da = float((va * va).sum()) ** 0.5
    db = float((vb * vb).sum()) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _top1_match_inmem(vecs: Sequence[Sequence[float]], ids: Sequence[str], thr: float) -> List[Tuple[str, str]]:
    import numpy as np

    V = np.asarray(vecs, dtype=float)
    # 余弦相似：向量已归一化时，点积即相似度；否则我们显式归一化
    norms = (V * V).sum(axis=1) ** 0.5
    norms[norms == 0] = 1.0
    Vn = V / norms[:, None]

    sims = Vn @ Vn.T
    np.fill_diagonal(sims, -1.0)
    argmax = sims.argmax(axis=1)
    maxval = sims.max(axis=1)
    pairs: List[Tuple[str, str]] = []
    for i, j in enumerate(argmax):
        if maxval[i] >= thr:
            pairs.append((ids[i], ids[int(j)]))
    return pairs


def _pairs_from_truth(samples: Iterable[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    pairs = []
    for s in samples:
        sid = s["id"]
        for d in s.get("dupes", []) or []:
            if sid == d:
                continue
            a, b = sorted([sid, d])
            pairs.append((a, b))
    # 去重
    return sorted(set(pairs))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_sentence_samples(path: Path) -> List[Dict[str, Any]]:
    samples = load_jsonl(path)
    for s in samples:
        if "id" not in s or "text" not in s:
            raise ValueError("sentence benchmark input must contain 'id' and 'text'")
    return samples


def load_document_samples(path: Path) -> List[Dict[str, Any]]:
    docs = load_jsonl(path)
    for d in docs:
        if "doc_id" not in d or "path" not in d:
            raise ValueError("document benchmark input must contain 'doc_id' and 'path'")
    return docs


def run_sentence_benchmark(samples: Sequence[Mapping[str, Any]], cfg: BenchConfig) -> Dict[str, Any]:
    ids = [str(s["id"]) for s in samples]
    texts = [str(s["text"]) for s in samples]

    embedder = Embedder(model_name=cfg.model, batch_size=cfg.batch_size)

    vecs: List[List[float]] = []
    per_sent_lat_ms: List[float] = []
    enc_start = time.perf_counter()
    for chunk in chunk_iter(texts, cfg.batch_size):
        b0 = time.perf_counter()
        v = embedder.encode(chunk)
        b1 = time.perf_counter()
        vecs.extend(v)
        batch_ms = (b1 - b0) * 1000.0
        per_sent_lat_ms.extend([batch_ms / max(1, len(chunk))] * len(chunk))
    enc_end = time.perf_counter()

    if cfg.backend == "inmem":
        preds = _top1_match_inmem(vecs, ids, cfg.sim_threshold)
        preds = sorted(set(tuple(sorted(p)) for p in preds))
        search_lat = {"mode": "inmem", "per_query": latency_summary(per_sent_lat_ms)}
        query_tp = throughput(len(ids), (enc_end - enc_start))
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
        ensure_collection(client, cfg.collection, vector_size=embedder.dim)

        payloads = [{"sid": sid} for sid in ids]
        upsert_points(client, cfg.collection, vecs, payloads, ids)

        preds_pairs: List[Tuple[str, str]] = []
        q_lat_ms: List[float] = []
        q0 = time.perf_counter()
        for sid, v in tqdm(list(zip(ids, vecs)), desc="Qdrant query", disable=not cfg.show_progress):
            s0 = time.perf_counter()
            res = client.search(
                collection_name=cfg.collection,
                query_vector=v,
                limit=1,
                with_payload=True,
                score_threshold=cfg.sim_threshold,
                query_filter=qm.Filter(must_not=[qm.HasIdCondition(has_id=[sid])]),
            )
            s1 = time.perf_counter()
            q_lat_ms.append((s1 - s0) * 1000.0)
            if res:
                match = res[0]
                mid = str(match.id)
                a, b = sorted([sid, mid])
                preds_pairs.append((a, b))
        q1 = time.perf_counter()
        preds = sorted(set(preds_pairs))
        search_lat = {"mode": "qdrant", "per_query": latency_summary(q_lat_ms), "total_ms": (q1 - q0) * 1000.0}
        query_tp = throughput(len(ids), (q1 - q0))

    truth_pairs = _pairs_from_truth(samples)

    c = confusion_from_sets(preds, truth_pairs)
    prf = precision_recall_f1(**c)

    enc_ms = (enc_end - enc_start) * 1000
    enc_lat = {"encode_total_ms": enc_ms, **latency_summary(per_sent_lat_ms)}
    total_sents = len(texts)
    enc_tp = throughput(total_sents, enc_end - enc_start)

    return {
        "counts": c,
        "metrics": prf,
        "latency": {"encode": enc_lat, "search": search_lat},
        "throughput": {"encode_sents_per_sec": enc_tp, "query_per_sec": query_tp},
        "n_samples": total_sents,
        "dim": embedder.dim,
    }


def run_document_benchmark(docs: Sequence[Mapping[str, Any]], cfg: BenchConfig) -> Dict[str, Any]:
    docs_list = list(docs)
    embedder = Embedder(model_name=cfg.model, batch_size=cfg.batch_size)

    doc2sents: Dict[str, List[str]] = {}
    all_sents: List[str] = []
    sent_doc: List[str] = []

    t_extract_start = time.perf_counter()
    per_doc_lat_ms: List[float] = []
    for doc in tqdm(docs_list, desc="Extract/segment", disable=not cfg.show_progress):
        path = Path(doc["path"])
        d0 = time.perf_counter()
        raw = extract_text(path)
        if not raw:
            doc2sents[doc["doc_id"]] = []
        else:
            txt = clean_text(raw)
            sents = sentences_nltk(txt)
            doc2sents[doc["doc_id"]] = sents
            all_sents.extend(sents)
            sent_doc.extend([doc["doc_id"]] * len(sents))
        d1 = time.perf_counter()
        per_doc_lat_ms.append((d1 - d0) * 1000.0)
    t_extract_end = time.perf_counter()

    t_enc_start = time.perf_counter()
    vecs: List[List[float]] = []
    per_sent_lat_ms: List[float] = []
    if all_sents:
        for chunk in chunk_iter(all_sents, cfg.batch_size):
            b0 = time.perf_counter()
            v = embedder.encode(chunk)
            b1 = time.perf_counter()
            vecs.extend(v)
            batch_ms = (b1 - b0) * 1000.0
            per_sent_lat_ms.extend([batch_ms / max(1, len(chunk))] * len(chunk))
    t_enc_end = time.perf_counter()

    preds_pairs: List[Tuple[str, str]] = []
    if vecs:
        import numpy as np

        V = np.asarray(vecs, dtype=float)
        norms = (V * V).sum(axis=1) ** 0.5
        norms[norms == 0] = 1.0
        Vn = V / norms[:, None]
        sims = Vn @ Vn.T
        np.fill_diagonal(sims, -1.0)
        for i in range(sims.shape[0]):
            sim_row = sims[i]
            same = [j for j, doc_id in enumerate(sent_doc) if doc_id == sent_doc[i]]
            if same:
                sim_row = sim_row.copy()
                for idx in same:
                    sim_row[idx] = -1.0
            j = int(sim_row.argmax())
            if sim_row[j] >= cfg.sim_threshold:
                a = f"{sent_doc[i]}:{i}"
                b = f"{sent_doc[j]}:{j}"
                x, y = sorted([a, b])
                preds_pairs.append((x, y))
        preds_pairs = sorted(set(preds_pairs))

    from collections import defaultdict

    doc_pair_counts = defaultdict(int)
    for anchor, match in preds_pairs:
        a_doc = anchor.split(":", 1)[0]
        b_doc = match.split(":", 1)[0]
        if a_doc == b_doc:
            continue
        x, y = sorted([a_doc, b_doc])
        doc_pair_counts[(x, y)] += 1

    def _qualify(x: str, y: str, cnt: int) -> bool:
        n_x = max(1, len(doc2sents.get(x, [])))
        n_y = max(1, len(doc2sents.get(y, [])))
        base = min(n_x, n_y)
        return cnt >= cfg.doc_min_pairs or (cnt / base) >= cfg.doc_min_ratio

    doc_preds = sorted({pair for pair, cnt in doc_pair_counts.items() if _qualify(pair[0], pair[1], cnt)})

    doc_truth_samples = [{"id": d["doc_id"], "dupes": d.get("dupes")}
                         for d in docs_list]
    doc_truth_pairs = _pairs_from_truth(doc_truth_samples)

    doc_counts = confusion_from_sets(doc_preds, doc_truth_pairs) if doc_truth_pairs else None
    doc_metrics = precision_recall_f1(**doc_counts) if doc_counts else None

    extract_ms = (t_extract_end - t_extract_start) * 1000
    encode_ms = (t_enc_end - t_enc_start) * 1000
    return {
    "n_docs": len(docs_list),
        "n_sents": len(all_sents),
        "n_doc_pairs_pred": len(doc_preds),
        "latency_ms": {
            "extract_segment_total": extract_ms,
            "extract_segment": latency_summary(per_doc_lat_ms),
            "encode_total": encode_ms,
            "encode_per_sentence": latency_summary(per_sent_lat_ms),
        },
        "throughput": {
            "sents_per_sec_encode": throughput(len(all_sents), t_enc_end - t_enc_start),
            "docs_per_sec_extract": throughput(len(docs_list), t_extract_end - t_extract_start),
        },
        "dim": embedder.dim,
        "doc_level": {
            "pred_pairs": len(doc_preds),
            "truth_pairs": len(doc_truth_pairs),
            "counts": doc_counts,
            "metrics": doc_metrics,
        },
    }


@click.group()
def cli():
    pass


@cli.command(name="sentences")
@click.option("--data", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="句子级 JSONL 文件")
@click.option("--model", default=_default_model_path(), show_default=True)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--threshold", default=0.8, show_default=True, help="余弦相似度阈值")
@click.option("--backend", type=click.Choice(["inmem", "qdrant"]), default="inmem", show_default=True)
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--qdrant-api-key", default=None)
@click.option("--collection", default="copyless_bench_tmp", show_default=True)
def bench_sentences(data: Path, model: str, batch_size: int, threshold: float, backend: str, qdrant_url: str, qdrant_api_key: Optional[str], collection: str):
    cfg = BenchConfig(model=model, batch_size=batch_size, sim_threshold=threshold, backend=backend, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, collection=collection)

    samples = load_sentence_samples(data)
    result = run_sentence_benchmark(samples, cfg)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@cli.command(name="documents")
@click.option("--data", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="文档级 JSONL：{doc_id, path}")
@click.option("--model", default=_default_model_path(), show_default=True)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--threshold", default=0.8, show_default=True, help="句子匹配相似度阈值")
@click.option("--doc-min-pairs", default=3, show_default=True)
@click.option("--doc-min-ratio", default=0.05, show_default=True)
def bench_documents(data: Path, model: str, batch_size: int, threshold: float, doc_min_pairs: int, doc_min_ratio: float):
    cfg = BenchConfig(model=model, batch_size=batch_size, sim_threshold=threshold, doc_min_pairs=doc_min_pairs, doc_min_ratio=doc_min_ratio)

    docs = load_document_samples(data)
    result = run_document_benchmark(docs, cfg)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
