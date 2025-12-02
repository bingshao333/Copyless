from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import click
from tqdm import tqdm

from src.embedding import Embedder
from src.extract import extract_text
from src.preprocess import (
    clean_text,
    sentences_nltk,
    sentences_spacy,
    split_mixed_sentences,
)
from src.qdrant_io import ensure_collection, upsert_points

logger = logging.getLogger("copyless.pipeline")

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B")


@dataclass
class PipelineConfig:
    input_dir: Path
    collection: str
    model: str = DEFAULT_MODEL_PATH
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_path: Optional[str] = None
    qdrant_prefer_grpc: bool = False
    qdrant_timeout: Optional[int] = None
    device: Optional[str] = None
    device_map: Optional[str] = None
    batch_size: int = 256
    upsert_batch: int = 1024
    sentence_splitter: str = "nltk"
    spacy_model: str = "en_core_web_sm"
    workers: int = 4
    qdrant_distance: str = "Cosine"
    on_disk: bool = True
    hnsw_m: int = 32
    hnsw_ef_construct: int = 512
    hnsw_full_scan_threshold: int = 1000
    recreate_collection: bool = False


@dataclass
class ComparisonConfig:
    collection: str
    model: str
    batch_size: Optional[int] = None


@dataclass
class FileProcessResult:
    path: Path
    sentences: List[str]
    paper_id: Optional[str]
    elapsed_ms: float
    spans: List[Tuple[int, int]]


class BatchUploader:
    def __init__(self, client, collection: str, batch_size: int) -> None:
        self.client = client
        self.collection = collection
        self.batch_size = batch_size
        self.vectors: List[List[float]] = []
        self.payloads: List[dict] = []
        self.ids: List[str] = []

    def add_many(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
        ids: Sequence[str],
    ) -> None:
        self.vectors.extend([list(vec) for vec in vectors])
        self.payloads.extend(payloads)
        self.ids.extend(ids)
        if len(self.vectors) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.vectors:
            return
        upsert_points(
            self.client,
            self.collection,
            vectors=list(self.vectors),
            payloads=list(self.payloads),
            ids=list(self.ids),
            batch_size=self.batch_size,
        )
        self.vectors.clear()
        self.payloads.clear()
        self.ids.clear()


def _iter_files(root: Path):
    exts = {".pdf", ".tex"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _guess_paper_id(p: Path) -> Optional[str]:
    m = _ARXIV_ID_RE.search(str(p)) or _ARXIV_ID_RE.search(p.stem)
    if m:
        return f"arXiv:{m.group(1)}"
    return None


def _resolve_splitter(cfg: PipelineConfig) -> Callable[[str], List[str]]:
    key = cfg.sentence_splitter.lower()
    if key == "nltk":
        return sentences_nltk
    if key == "spacy":
        return lambda text: sentences_spacy(text, model=cfg.spacy_model)
    if key == "mixed":
        return split_mixed_sentences
    raise ValueError(
        f"Unsupported sentence splitter '{cfg.sentence_splitter}'. 可选值: nltk | spacy | mixed"
    )


def _match_sentences_to_spans(text: str, sentences: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    n = len(text)
    for sent in sentences:
        raw = sent
        if not raw:
            spans.append((cursor, cursor))
            continue

        idx = text.find(raw, cursor)
        length = len(raw)
        candidate: Optional[str] = None

        if idx == -1:
            candidate = raw.strip()
            if candidate:
                idx = text.find(candidate, cursor)
                length = len(candidate)

        if idx == -1 and candidate:
            idx = text.find(candidate)

        if idx == -1:
            idx = cursor
            length = len(raw.strip()) or len(raw)

        start = max(0, idx)
        end = min(n, start + length)
        spans.append((start, end))
        cursor = end
    return spans


def _process_single_file(
    path: Path, splitter: Callable[[str], List[str]]
) -> Optional[FileProcessResult]:
    t0 = time.perf_counter()
    raw = extract_text(path)
    if not raw:
        return None
    txt = clean_text(raw)
    sentences = splitter(txt)
    if not sentences:
        return None
    paper_id = _guess_paper_id(path)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    spans = _match_sentences_to_spans(txt, sentences)
    return FileProcessResult(
        path=path,
        sentences=sentences,
        paper_id=paper_id,
        elapsed_ms=elapsed_ms,
        spans=spans,
    )


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


@click.command()
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--collection", required=True, type=str)
@click.option(
    "--model",
    default=DEFAULT_MODEL_PATH,
    show_default=True,
    help="主索引用的嵌入模型 (HuggingFace 名称或本地路径)",
)
@click.option(
    "--qdrant-url",
    default="http://localhost:6333",
    show_default=True,
    envvar="QDRANT_URL",
)
@click.option("--qdrant-api-key", default=None, envvar="QDRANT_API_KEY")
@click.option(
    "--qdrant-path",
    default=None,
    help="可选：使用本地嵌入式 Qdrant 存储路径 (无需独立服务)",
)
@click.option(
    "--qdrant-prefer-grpc/--qdrant-rest",
    default=False,
    show_default=True,
    help="连接远程 Qdrant 时是否优先使用 gRPC",
)
@click.option(
    "--qdrant-timeout", default=None, type=int, help="Qdrant 客户端网络超时时长（秒）"
)
@click.option(
    "--device",
    default=None,
    show_default=True,
    help="PyTorch 设备字符串，如 cuda、cuda:0、cpu。为空则自动选择",
)
@click.option(
    "--device-map",
    default=None,
    show_default=True,
    help="Transformers device_map，用于 accelerate 多卡部署，如 auto/balanced",
)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--upsert-batch", default=1024, show_default=True)
@click.option(
    "--sentence-splitter",
    type=click.Choice(["nltk", "spacy", "mixed"], case_sensitive=False),
    default="nltk",
    show_default=True,
)
@click.option(
    "--spacy-model",
    default="en_core_web_sm",
    show_default=True,
    help="sentence_splitter=spacy 时使用的模型名",
)
@click.option(
    "--workers", default=4, show_default=True, help="抽取+切句并行线程数 (<=1 时串行)"
)
@click.option("--qdrant-distance", default="Cosine", show_default=True)
@click.option("--on-disk/--in-memory", default=True, show_default=True)
@click.option("--hnsw-m", default=32, show_default=True)
@click.option("--hnsw-ef-construct", default=512, show_default=True)
@click.option("--hnsw-full-scan-threshold", default=1000, show_default=True)
@click.option(
    "--recreate/--no-recreate", default=False, help="如已存在集合是否强制重建"
)
@click.option(
    "--comparison-collection", default=None, help="可选：创建一个对比 Qdrant 集合"
)
@click.option(
    "--comparison-model", default=None, help="对比集合使用的模型，未提供则与主模型一致"
)
@click.option(
    "--comparison-batch-size",
    default=None,
    type=int,
    help="对比模型的编码批大小 (默认与主模型相同)",
)
@click.option(
    "--dry-run/--no-dry-run", default=False, help="仅验证抽取/切句/嵌入，不连接 Qdrant"
)
@click.option(
    "--dump",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="dry-run 或生产时输出 JSONL 结果",
)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    input_dir: Path,
    collection: str,
    model: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    qdrant_path: Optional[str],
    qdrant_prefer_grpc: bool,
    qdrant_timeout: Optional[int],
    device: Optional[str],
    device_map: Optional[str],
    batch_size: int,
    upsert_batch: int,
    sentence_splitter: str,
    spacy_model: str,
    workers: int,
    qdrant_distance: str,
    on_disk: bool,
    hnsw_m: int,
    hnsw_ef_construct: int,
    hnsw_full_scan_threshold: int,
    recreate: bool,
    comparison_collection: Optional[str],
    comparison_model: Optional[str],
    comparison_batch_size: Optional[int],
    dry_run: bool,
    dump: Optional[Path],
    log_level: str,
):
    _setup_logging(log_level)
    logger.info("启动 Copyless 索引管线：input=%s collection=%s", input_dir, collection)

    cfg = PipelineConfig(
        input_dir=input_dir,
        collection=collection,
        model=model,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_path=qdrant_path,
        qdrant_prefer_grpc=qdrant_prefer_grpc,
        qdrant_timeout=qdrant_timeout,
        device=device,
        device_map=device_map,
        batch_size=batch_size,
        upsert_batch=upsert_batch,
        sentence_splitter=sentence_splitter,
        spacy_model=spacy_model,
        workers=workers,
        qdrant_distance=qdrant_distance,
        on_disk=on_disk,
        hnsw_m=hnsw_m,
        hnsw_ef_construct=hnsw_ef_construct,
        hnsw_full_scan_threshold=hnsw_full_scan_threshold,
        recreate_collection=recreate,
    )

    comparison_cfg = None
    if comparison_collection:
        comparison_cfg = ComparisonConfig(
            collection=comparison_collection,
            model=comparison_model or model,
            batch_size=comparison_batch_size or batch_size,
        )

    splitter_fn = _resolve_splitter(cfg)
    try:
        splitter_fn("Warm up sentence splitter.")
    except Exception:
        logger.debug("分句器预热完成")

    embedder = Embedder(
        model_name=cfg.model,
        batch_size=cfg.batch_size,
        device=cfg.device,
        device_map=cfg.device_map,
    )
    if (
        comparison_cfg
        and comparison_cfg.model == cfg.model
        and (comparison_cfg.batch_size or cfg.batch_size) == cfg.batch_size
    ):
        comparison_embedder = embedder
    elif comparison_cfg:
        comparison_embedder = Embedder(
            model_name=comparison_cfg.model,
            batch_size=comparison_cfg.batch_size or cfg.batch_size,
            device=cfg.device,
            device_map=cfg.device_map,
        )
    else:
        comparison_embedder = None

    client = None
    if not dry_run:
        from qdrant_client import QdrantClient

        if cfg.qdrant_path:
            client = QdrantClient(path=cfg.qdrant_path)
        else:
            client_kwargs: dict[str, Any] = {"url": cfg.qdrant_url}
            if cfg.qdrant_api_key:
                client_kwargs["api_key"] = cfg.qdrant_api_key
            if cfg.qdrant_prefer_grpc:
                client_kwargs["prefer_grpc"] = True
            if cfg.qdrant_timeout is not None:
                client_kwargs["timeout"] = cfg.qdrant_timeout
            client = QdrantClient(**client_kwargs)
        ensure_collection(
            client,
            cfg.collection,
            embedder.dim,
            distance=cfg.qdrant_distance,
            on_disk=cfg.on_disk,
            hnsw_m=cfg.hnsw_m,
            hnsw_ef_construct=cfg.hnsw_ef_construct,
            hnsw_full_scan_threshold=cfg.hnsw_full_scan_threshold,
            force_recreate=cfg.recreate_collection,
        )
        if comparison_cfg:
            ensure_collection(
                client,
                comparison_cfg.collection,
                (comparison_embedder or embedder).dim,
                distance=cfg.qdrant_distance,
                on_disk=cfg.on_disk,
                hnsw_m=cfg.hnsw_m,
                hnsw_ef_construct=cfg.hnsw_ef_construct,
                hnsw_full_scan_threshold=cfg.hnsw_full_scan_threshold,
                force_recreate=cfg.recreate_collection,
            )

    primary_uploader = (
        BatchUploader(client, cfg.collection, cfg.upsert_batch)
        if (not dry_run and client)
        else None
    )
    comparison_uploader = (
        BatchUploader(client, comparison_cfg.collection, cfg.upsert_batch)
        if (not dry_run and client and comparison_cfg)
        else None
    )

    files = list(_iter_files(cfg.input_dir))
    if not files:
        summary = {
            "files_processed": 0,
            "sentences_processed": 0,
            "dim_primary": embedder.dim,
            "primary_model": cfg.model,
            "comparison": None,
            "latency_ms": {},
            "dry_run": dry_run,
        }
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if dump:
        dump.parent.mkdir(parents=True, exist_ok=True)
        if dump.exists():
            dump.unlink()
        dump_f = dump.open("a", encoding="utf-8")
    else:
        dump_f = None

    total_sent = 0
    total_files = 0
    extract_lat_ms: List[float] = []
    encode_lat_ms: List[float] = []
    comparison_encode_lat_ms: List[float] = []

    worker = partial(_process_single_file, splitter=splitter_fn)

    def _handle_result(res: Optional[FileProcessResult]):
        nonlocal total_sent, total_files
        if res is None:
            return
        total_files += 1
        total_sent += len(res.sentences)
        extract_lat_ms.append(res.elapsed_ms)

        base_id = hashlib.md5(str(res.path).encode()).hexdigest()
        payload_base = {
            "path": str(res.path),
            "paper_id": res.paper_id,
        }

        encode_start = time.perf_counter()
        primary_vecs = embedder.encode(res.sentences)
        encode_lat_ms.append((time.perf_counter() - encode_start) * 1000.0)

        ids: List[str] = []
        primary_payloads: List[dict] = []
        payload_commons: List[dict] = []
        for i, sent in enumerate(res.sentences):
            sid_hex = hashlib.md5(f"{base_id}:{i}".encode()).hexdigest()
            sid = str(uuid.UUID(sid_hex))
            ids.append(sid)
            char_start, char_end = res.spans[i] if i < len(res.spans) else (None, None)
            common = {
                **payload_base,
                "sent_index": i,
                "text": sent,
                "char_start": char_start,
                "char_end": char_end,
            }
            payload_commons.append(common)
            primary_payloads.append({**common, "embedding_model": cfg.model})

        if comparison_cfg:
            if comparison_embedder is embedder:
                comp_vecs = primary_vecs
            else:
                if comparison_embedder is None:
                    raise RuntimeError("comparison_embedder is not initialized")
                comp_start = time.perf_counter()
                comp_vecs = comparison_embedder.encode(res.sentences)
                comparison_encode_lat_ms.append(
                    (time.perf_counter() - comp_start) * 1000.0
                )
            comp_payloads = [
                {**common, "embedding_model": comparison_cfg.model}
                for common in payload_commons
            ]
        else:
            comp_vecs = None
            comp_payloads = None

        if dump_f:
            for idx, (sid, common) in enumerate(zip(ids, payload_commons)):
                record = {
                    "id": sid,
                    **common,
                    "models": {cfg.collection: cfg.model},
                    "embeddings": {cfg.collection: primary_vecs[idx]},
                }
                if comparison_cfg and comp_vecs is not None:
                    record["models"][comparison_cfg.collection] = comparison_cfg.model
                    record["embeddings"][comparison_cfg.collection] = comp_vecs[idx]
                dump_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if primary_uploader:
            primary_uploader.add_many(primary_vecs, primary_payloads, ids)
        if comparison_uploader and comp_vecs is not None and comp_payloads is not None:
            comparison_uploader.add_many(comp_vecs, comp_payloads, ids)

    if cfg.workers <= 1:
        iterator = (worker(f) for f in files)
        for res in tqdm(iterator, total=len(files), desc="Extract+segment"):
            try:
                _handle_result(res)
            except Exception as exc:
                logger.exception(
                    "处理文件 %s 失败: %s", getattr(res, "path", "unknown"), exc
                )
    else:
        with futures.ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            future_map = {executor.submit(worker, f): f for f in files}
            for fut in tqdm(
                futures.as_completed(future_map),
                total=len(files),
                desc="Extract+segment",
            ):
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.error("处理文件 %s 失败: %s", future_map[fut], exc)
                    continue
                try:
                    _handle_result(res)
                except Exception as exc:
                    logger.exception(
                        "处理文件 %s 失败: %s",
                        getattr(res, "path", future_map.get(fut)),
                        exc,
                    )

    if primary_uploader:
        primary_uploader.flush()
    if comparison_uploader:
        comparison_uploader.flush()
    if dump_f:
        dump_f.close()

    summary = {
        "files_processed": total_files,
        "sentences_processed": total_sent,
        "dim_primary": embedder.dim,
        "primary_model": cfg.model,
        "comparison": {
            "collection": comparison_cfg.collection if comparison_cfg else None,
            "model": comparison_cfg.model if comparison_cfg else None,
        },
        "latency_ms": {
            "extract_avg": (
                (sum(extract_lat_ms) / len(extract_lat_ms)) if extract_lat_ms else 0.0
            ),
            "encode_primary_avg": (
                (sum(encode_lat_ms) / len(encode_lat_ms)) if encode_lat_ms else 0.0
            ),
            "encode_comparison_avg": (
                (sum(comparison_encode_lat_ms) / len(comparison_encode_lat_ms))
                if comparison_encode_lat_ms
                else 0.0
            ),
        },
        "dry_run": dry_run,
    }

    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
