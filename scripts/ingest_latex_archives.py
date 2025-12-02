#!/usr/bin/env python3
"""Ingest LaTeX sources packaged inside tar/tar.gz archives into Qdrant.

The arXiv bulk source dataset stores each paper inside a tar archive (optionally
wrapped in gzip). This utility streams those archives without unpacking them to
disk, extracts LaTeX-relevant files, splits them into sentences, embeds each
sentence, and writes the vectors into a Qdrant collection.

Example (dummy embedding, limited sample):

    python scripts/ingest_latex_archives.py \
        --input data/latex_extracted/arXiv_src_0001_001 \
        --collection copyless_latex_dummy \
        --model dummy \
        --max-archives 1 --max-files 5

To run against the full dataset with the Qwen3-0.6B model, drop the limits and
set ``--model`` to the local model path or a HuggingFace identifier.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, List

import click
from tqdm import tqdm

from src.embedding import Embedder
from src.preprocess import clean_text, sentences_nltk, sentences_spacy, split_mixed_sentences
from src.qdrant_io import ensure_collection
from src.pipeline import BatchUploader

try:  # 兼容缺失 qdrant_client 的环境
    from qdrant_client import QdrantClient
except Exception as exc:  # pragma: no cover - 在运行时提示
    QdrantClient = None  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover - 成功导入无需提示
    _IMPORT_ERROR = None


LATEX_SUFFIXES = (".tex", ".ltx")
DEFAULT_MAX_BYTES = 2_000_000  # 单个文件最大读取字节数 (~2 MB)


@dataclass
class IngestConfig:
    input_root: Path
    collection: str
    model: str
    batch_size: int
    upsert_batch: int
    sentence_splitter: str
    spacy_model: str
    max_bytes: int
    max_archives: Optional[int]
    max_files: Optional[int]
    dry_run: bool
    dump_path: Optional[Path]
    qdrant_url: str
    qdrant_api_key: Optional[str]
    qdrant_path: Optional[Path]
    qdrant_timeout: Optional[int]
    prefer_grpc: bool
    log_level: str
    on_disk: bool
    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_full_scan_threshold: int
    recreate_collection: bool


@dataclass
class LatexSource:
    archive_path: Path
    member_name: str
    text: str
    truncated: bool


def _iter_archives(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _iter_latex_sources(path: Path, max_bytes: int) -> Iterator[LatexSource]:
    if tarfile.is_tarfile(path):
        # ``r:*`` 自动识别压缩格式（包含 gzip/bzip2/xz 等）
        with tarfile.open(path, mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                yield from _iter_member_sources(path, tar, member, max_bytes)
    else:
        suffix = path.suffix.lower()
        if suffix == ".gz":
            with gzip.open(path, "rb") as fh:
                data = fh.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            if truncated:
                data = data[:max_bytes]
            text = _decode_bytes(data)
            yield LatexSource(path, path.stem, text, truncated)
        elif suffix in LATEX_SUFFIXES:
            with path.open("rb") as fh:
                data = fh.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            if truncated:
                data = data[:max_bytes]
            text = _decode_bytes(data)
            yield LatexSource(path, path.name, text, truncated)
        else:
            logging.debug("skip non-Latex file %s", path)


def _iter_member_sources(
    archive_path: Path, tar: tarfile.TarFile, member: tarfile.TarInfo, max_bytes: int
) -> Iterator[LatexSource]:
    fileobj = tar.extractfile(member)
    if fileobj is None:
        return

    name = member.name
    lower_name = name.lower()

    if lower_name.endswith(LATEX_SUFFIXES):
        if member.size and member.size > max_bytes:
            logging.debug(
                "skip member %s (%.2f MB > limit)",
                name,
                member.size / 1_048_576,
            )
            return
        data = fileobj.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        text = _decode_bytes(data)
        yield LatexSource(archive_path=archive_path, member_name=name, text=text, truncated=truncated)
        return

    if lower_name.endswith(".gz"):
        try:
            compressed = fileobj.read()
            decompressed = gzip.decompress(compressed)
        except OSError:
            logging.debug("failed to decompress gzip member %s", name)
            return

        buffer = io.BytesIO(decompressed)
        try:
            with tarfile.open(fileobj=buffer, mode="r:*") as inner_tar:
                for inner_member in inner_tar.getmembers():
                    if not inner_member.isfile():
                        continue
                    inner_name = inner_member.name
                    if not inner_name.lower().endswith(LATEX_SUFFIXES):
                        continue
                    if inner_member.size and inner_member.size > max_bytes:
                        logging.debug(
                            "skip nested member %s (%.2f MB > limit)",
                            inner_name,
                            inner_member.size / 1_048_576,
                        )
                        continue
                    inner_file = inner_tar.extractfile(inner_member)
                    if inner_file is None:
                        continue
                    data = inner_file.read(max_bytes + 1)
                    truncated = len(data) > max_bytes
                    if truncated:
                        data = data[:max_bytes]
                    text = _decode_bytes(data)
                    compound_name = f"{name}!{inner_name}"
                    yield LatexSource(
                        archive_path=archive_path,
                        member_name=compound_name,
                        text=text,
                        truncated=truncated,
                    )
        except tarfile.TarError:
            truncated = len(decompressed) > max_bytes
            data = decompressed[:max_bytes] if truncated else decompressed
            text = _decode_bytes(data)
            yield LatexSource(
                archive_path=archive_path,
                member_name=name,
                text=text,
                truncated=truncated,
            )
        return


def _resolve_splitter(name: str, spacy_model: str):
    key = name.strip().lower()
    if key == "nltk":
        return sentences_nltk
    if key == "spacy":
        return lambda text: sentences_spacy(text, model=spacy_model)
    if key == "mixed":
        return split_mixed_sentences
    raise ValueError(f"Unsupported sentence splitter '{name}'. 可选值: nltk | spacy | mixed")


def _guess_paper_id(source: LatexSource) -> Optional[str]:
    import re

    pattern = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
    for target in (str(source.archive_path), source.member_name):
        match = pattern.search(target)
        if match:
            return f"arXiv:{match.group(1)}"
    return None


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def _ensure_qdrant_collection(cfg: IngestConfig, embedder: Embedder) -> Optional[BatchUploader]:
    if cfg.dry_run:
        return None
    if QdrantClient is None:
        raise RuntimeError(
            "qdrant_client 未安装，无法连接 Qdrant"
        ) from _IMPORT_ERROR

    if cfg.qdrant_path:
        client = QdrantClient(path=str(cfg.qdrant_path))
    else:
        client_kwargs = {"url": cfg.qdrant_url}
        if cfg.qdrant_api_key:
            client_kwargs["api_key"] = cfg.qdrant_api_key
        if cfg.prefer_grpc:
            client_kwargs["prefer_grpc"] = True
        if cfg.qdrant_timeout is not None:
            client_kwargs["timeout"] = cfg.qdrant_timeout
        client = QdrantClient(**client_kwargs)

    ensure_collection(
        client,
        cfg.collection,
        embedder.dim,
        distance="Cosine",
        on_disk=cfg.on_disk,
        hnsw_m=cfg.hnsw_m,
        hnsw_ef_construct=cfg.hnsw_ef_construct,
        hnsw_full_scan_threshold=cfg.hnsw_full_scan_threshold,
        force_recreate=cfg.recreate_collection,
    )
    return BatchUploader(client, cfg.collection, cfg.upsert_batch)


def _locate_sentence_spans(text: str, sentences: Sequence[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    spans: List[Tuple[Optional[int], Optional[int]]] = []
    cursor = 0
    for sent in sentences:
        if not sent:
            spans.append((None, None))
            continue
        idx = text.find(sent, cursor)
        if idx == -1:
            idx = text.find(sent)
        if idx == -1:
            spans.append((None, None))
            continue
        start = idx
        end = idx + len(sent)
        spans.append((start, end))
        cursor = end
    return spans


def _build_payloads(
    source: LatexSource,
    sentences: Sequence[str],
    spans: Sequence[Tuple[Optional[int], Optional[int]]],
    model: str,
) -> Tuple[Sequence[str], Sequence[dict]]:
    base = hashlib.md5(f"{source.archive_path}:{source.member_name}".encode()).hexdigest()
    paper_id = _guess_paper_id(source)

    ids = []
    payloads = []
    for idx, sent in enumerate(sentences):
        sid_hex = hashlib.md5(f"{base}:{idx}".encode()).hexdigest()
        sid = str(uuid.UUID(sid_hex))
        ids.append(sid)
        span = spans[idx] if idx < len(spans) else (None, None)
        char_start, char_end = span
        payloads.append(
            {
                "archive_path": str(source.archive_path),
                "member": source.member_name,
                "sent_index": idx,
                "text": sent,
                "paper_id": paper_id,
                "embedding_model": model,
                "truncated": source.truncated,
                "char_start": char_start,
                "char_end": char_end,
                "char_length": len(sent),
            }
        )
    return ids, payloads


def _write_dump(dump_path: Path, records: Iterable[dict]) -> None:
    with dump_path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@click.command()
@click.option("--input", "input_root", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--collection", required=True, type=str)
@click.option("--model", default=None, type=str, help="嵌入模型 (默认为仓库内置 Qwen3-0.6B)")
@click.option("--batch-size", default=256, show_default=True)
@click.option("--upsert-batch", default=1024, show_default=True)
@click.option(
    "--sentence-splitter",
    type=click.Choice(["nltk", "spacy", "mixed"], case_sensitive=False),
    default="nltk",
    show_default=True,
)
@click.option("--spacy-model", default="en_core_web_sm", show_default=True)
@click.option("--max-bytes", default=DEFAULT_MAX_BYTES, show_default=True, help="单个 LaTeX 文件最大读取字节数")
@click.option("--max-archives", type=int, default=None, help="可选：限制处理的归档数量")
@click.option("--max-files", type=int, default=None, help="可选：限制处理的 LaTeX 文件数量")
@click.option("--dry-run/--no-dry-run", default=False, show_default=True)
@click.option("--dump", "dump_path", type=click.Path(path_type=Path), default=None, help="可选：输出 JSONL 结果")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option(
    "--qdrant-path",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="可选：使用本地嵌入式 Qdrant 存储路径",
)
@click.option("--qdrant-api-key", default=None)
@click.option("--qdrant-timeout", type=int, default=None)
@click.option("--prefer-grpc/--prefer-rest", default=False, show_default=True)
@click.option("--log-level", default="INFO", show_default=True)
@click.option("--on-disk/--in-memory", default=True, show_default=True)
@click.option("--hnsw-m", default=16, show_default=True)
@click.option("--hnsw-ef-construct", default=100, show_default=True)
@click.option("--hnsw-full-scan-threshold", default=10_000, show_default=True)
@click.option("--recreate/--no-recreate", default=False, show_default=True)
def main(
    input_root: Path,
    collection: str,
    model: Optional[str],
    batch_size: int,
    upsert_batch: int,
    sentence_splitter: str,
    spacy_model: str,
    max_bytes: int,
    max_archives: Optional[int],
    max_files: Optional[int],
    dry_run: bool,
    dump_path: Optional[Path],
    qdrant_url: str,
    qdrant_path: Optional[Path],
    qdrant_api_key: Optional[str],
    qdrant_timeout: Optional[int],
    prefer_grpc: bool,
    log_level: str,
    on_disk: bool,
    hnsw_m: int,
    hnsw_ef_construct: int,
    hnsw_full_scan_threshold: int,
    recreate: bool,
) -> None:
    """入口函数：遍历归档并写入 Qdrant。"""

    _setup_logging(log_level)

    cfg = IngestConfig(
        input_root=input_root,
        collection=collection,
        model=model or str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B"),
        batch_size=batch_size,
        upsert_batch=upsert_batch,
        sentence_splitter=sentence_splitter,
        spacy_model=spacy_model,
        max_bytes=max_bytes,
        max_archives=max_archives,
        max_files=max_files,
        dry_run=dry_run,
        dump_path=dump_path,
        qdrant_url=qdrant_url,
    qdrant_path=qdrant_path,
        qdrant_api_key=qdrant_api_key,
        qdrant_timeout=qdrant_timeout,
        prefer_grpc=prefer_grpc,
        log_level=log_level,
        on_disk=on_disk,
        hnsw_m=hnsw_m,
        hnsw_ef_construct=hnsw_ef_construct,
        hnsw_full_scan_threshold=hnsw_full_scan_threshold,
        recreate_collection=recreate,
    )

    splitter_fn = _resolve_splitter(cfg.sentence_splitter, cfg.spacy_model)

    embedder = Embedder(model_name=cfg.model, batch_size=cfg.batch_size)
    uploader = _ensure_qdrant_collection(cfg, embedder)

    archives_processed = 0
    files_processed = 0
    sentences_processed = 0
    truncated_files = 0

    if cfg.dump_path:
        cfg.dump_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.dump_path.exists():
            cfg.dump_path.unlink()

    archive_iter: Iterable[Path] = _iter_archives(cfg.input_root)
    if cfg.max_archives is not None:
        archive_iter = (
            path for idx, path in enumerate(archive_iter) if idx < cfg.max_archives
        )

    for archive_path in tqdm(archive_iter, desc="Archives"):
        archives_processed += 1
        try:
            sources_iter = _iter_latex_sources(archive_path, cfg.max_bytes)
        except Exception as exc:
            logging.error("解析归档 %s 失败: %s", archive_path, exc)
            continue

        for source in sources_iter:
            if cfg.max_files is not None and files_processed >= cfg.max_files:
                break
            files_processed += 1
            if source.truncated:
                truncated_files += 1

            cleaned = clean_text(source.text)
            sentences = [s for s in splitter_fn(cleaned) if s]
            if not sentences:
                continue

            vectors = embedder.encode(sentences)
            sentences_processed += len(sentences)

            spans = _locate_sentence_spans(cleaned, sentences)
            ids, payloads = _build_payloads(source, sentences, spans, cfg.model)

            if cfg.dump_path:
                records = []
                for idx, sent in enumerate(sentences):
                    records.append(
                        {
                            "id": ids[idx],
                            "text": sent,
                            "payload": payloads[idx],
                            "embedding": vectors[idx],
                            "collection": cfg.collection,
                        }
                    )
                _write_dump(cfg.dump_path, records)

            if uploader:
                uploader.add_many(vectors, payloads, ids)

        if cfg.max_files is not None and files_processed >= cfg.max_files:
            break

    if uploader:
        uploader.flush()

    summary = {
        "archives_processed": archives_processed,
        "files_processed": files_processed,
        "sentences_processed": sentences_processed,
        "truncated_files": truncated_files,
        "model": cfg.model,
        "collection": cfg.collection,
        "dry_run": cfg.dry_run,
    }

    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
