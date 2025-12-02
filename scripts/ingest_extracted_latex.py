#!/usr/bin/env python3
"""遍历目录下的 LaTeX 文件，分句，向量化并入 Qdrant 数据库。

用途示例:
    python scripts/ingest_extracted_latex.py /mnt/data/user/shao_bing/src_extracted --collection copyless_shao --batch-size 256 --dummy

该脚本将：
 - 遍历给定目录中的 .tex/.tex.gz/.gz 文件
 - 从 LaTeX 内容中移除注释、环境与数学表达式（简单启发式）
 - 使用仓库中的分句器将文本切分为句子
 - 批量调用 `Embedder.encode` 获取向量
 - 使用 `qdrant_io.ensure_collection` 与 `upsert_points` 将向量与 payload 写入 Qdrant

脚本支持 `--dummy` 模式，使用 Embedder(dummy) 快速产生可复现向量，便于本地测试而不加载大型模型。
"""

from __future__ import annotations

import argparse
import gzip
import os
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from src.preprocess import clean_text, get_sentence_splitter
from src.embedding import Embedder
from src.qdrant_io import ensure_collection, upsert_points
from src.service.config import ServiceSettings


def extract_text_from_tex(content: str) -> str:
    r"""简单从 LaTeX 内容中抽取可见文本：
    - 移除注释（% 到行尾，注意 \% 保留）
    - 移除数学环境 $...$, $$...$$, \[...\], \(...\) 与常见 equation 环境（启发式）
    - 移除 \begin{...} ... \end{...} 的某些环境（figure, table 等），仅做简单文本移除
    该函数是启发式的，目的是得到可供分句的纯文本。
    """
    import re

    # 去掉行内注释（保留被转义的 %）
    content = re.sub(r"(?<!\\)%.*$", "", content, flags=re.MULTILINE)

    # 移除 math 模式
    content = re.sub(r"\$\$.*?\$\$", " ", content, flags=re.DOTALL)
    content = re.sub(r"\$.*?\$", " ", content, flags=re.DOTALL)
    content = re.sub(r"\\\[.*?\\\]", " ", content, flags=re.DOTALL)
    content = re.sub(r"\\\(.*?\\\)", " ", content, flags=re.DOTALL)

    # 移除常见环境块（figure, table, align, equation, lstlisting 等）
    content = re.sub(r"\\begin\{(?:figure|table|align\*?|equation\*?|alignat|lstlisting|verbatim)[^}]*\}.*?\\end\{[^}]*\}", " ", content, flags=re.DOTALL)

    # 移除命令 \command[...]{...} 或 \command{...} 的命令名，仅保留大括号内的内容或移除整个命令（启发式）
    # 这里先把简单命令去掉，保留花括号内文本
    content = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^}]*)\}", r" \1 ", content)
    # 移除剩余的命令 \command 或 \command[...]
    content = re.sub(r"\\[a-zA-Z@]+(?:\[[^\]]*\])?", " ", content)

    # 清理多余空白
    return clean_text(content)


def iter_tex_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lowered = fn.lower()
            if lowered.endswith(".tex") or lowered.endswith(".tex.gz") or lowered.endswith(".gz"):
                yield Path(dirpath) / fn


def build_payload(file_path: Path, sent_index: int, sentence: str) -> Dict[str, Any]:
    # Handle potential surrogate characters in file path for JSON serialization
    path_str = str(file_path)
    try:
        path_str.encode("utf-8")
    except UnicodeEncodeError:
        # If it fails, it likely has surrogates. We replace them to ensure JSON compatibility.
        path_str = path_str.encode("utf-8", "surrogateescape").decode("utf-8", "replace")

    return {
        "path": path_str,
        "sent_index": sent_index,
        "text": sentence,
    }


def build_point_id(file_path: Path, sent_index: int) -> str:
    # Path names from archives may contain surrogate characters; encode safely.
    path_bytes = str(file_path).encode("utf-8", "surrogateescape")
    name = f"{path_bytes.hex()}::{sent_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def process_directory(
    src_dir: Path,
    collection: str,
    model_name: str | None,
    batch_size: int,
    splitter_name: str,
    qdrant_client_kwargs: dict | None = None,
    dummy: bool = False,
    no_upsert: bool = False,
):
    settings = ServiceSettings()
    if model_name is not None:
        settings.embedding_model = model_name
    settings.embedding_batch_size = batch_size

    splitter = get_sentence_splitter(splitter_name)

    # 构建 embedder
    emb = Embedder(model_name="dummy" if dummy else settings.embedding_model, batch_size=settings.embedding_batch_size)

    # 先确保 collection
    from qdrant_client import QdrantClient

    q_kwargs = dict(qdrant_client_kwargs) if qdrant_client_kwargs else {}
    q_kwargs.setdefault("url", settings.qdrant_url)
    q_kwargs.setdefault("timeout", 60.0)
    client = QdrantClient(**q_kwargs)

    ensure_collection(client, collection, vector_size=emb.dim, distance="Cosine", on_disk=True)

    texts_buf: List[str] = []
    payloads_buf: List[Dict[str, Any]] = []
    ids_buf: List[str] = []
    count = 0
    files_seen = 0

    for fp in iter_tex_files(src_dir):
        files_seen += 1
        try:
            if str(fp).lower().endswith(".gz"):
                with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        plain = extract_text_from_tex(text)
        if not plain:
            continue
        sents = [s for s in splitter(plain) if s]
        print(f"File: {fp} -> {len(sents)} sentences")
        # 分批编码并缓存入库
        for i, sent in enumerate(sents):
            payload = build_payload(fp, i, sent)
            texts_buf.append(sent)
            payloads_buf.append(payload)
            ids_buf.append(build_point_id(fp, i))

            # 当达到 batch_size 时，编码并上载
            if len(texts_buf) >= batch_size:
                vecs = emb.encode(texts_buf)
                if not no_upsert:
                    upsert_points(client, collection, vecs, payloads_buf, ids=ids_buf, batch_size=1024)
                count += len(vecs)
                texts_buf = []
                payloads_buf = []
                ids_buf = []

    # 处理尾部残余
    if payloads_buf:
        texts = texts_buf if texts_buf else [p["text"] for p in payloads_buf]
        vecs = emb.encode(texts)
        if not no_upsert:
            upsert_points(client, collection, vecs, payloads_buf, ids=ids_buf, batch_size=1024)
        count += len(vecs)

    print(f"Processed {files_seen} files in {src_dir}")

    print(f"Finished. Inserted {count} sentences into collection '{collection}'.")


def main():
    parser = argparse.ArgumentParser(description="Ingest extracted LaTeX files into Qdrant by sentence embeddings")
    parser.add_argument("src_dir", help="目录，包含 .tex 文件", type=Path)
    parser.add_argument("--collection", help="Qdrant collection 名称", default="copyless_latex")
    parser.add_argument("--model", help="embedding model path 或名称", default=None)
    parser.add_argument("--batch-size", help="句子编码批大小", type=int, default=128)
    parser.add_argument("--splitter", help="句子切分器: nltk|spacy|mixed", default="mixed")
    parser.add_argument("--qdrant-url", help="Qdrant 服务 URL", default=None)
    parser.add_argument("--dummy", help="使用 dummy embedder（快速）", action="store_true")
    parser.add_argument("--no-upsert", help="仅到向量化，不实际写入 Qdrant", action="store_true")
    args = parser.parse_args()

    q_kwargs = {}
    if args.qdrant_url:
        q_kwargs["url"] = args.qdrant_url

    process_directory(
        args.src_dir,
        collection=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
        splitter_name=args.splitter,
        qdrant_client_kwargs=q_kwargs if q_kwargs else None,
        dummy=args.dummy,
        no_upsert=args.no_upsert,
    )
if __name__ == "__main__":
    main()
