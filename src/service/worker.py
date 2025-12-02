from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .citations import CitationIndex
from .config import ServiceSettings, get_settings
from .models import BenchmarkKind, BenchmarkPayload, MatchedSource, SentenceReport, SentenceStatus
from .report import aggregate_sentence_reports
from .retrieval import RetrievalPipeline
from .tasks import TaskQueue
from .utils import decision_tree_classification, normalized_levenshtein, weighted_score
from ..benchmark import BenchConfig, load_document_samples, load_sentence_samples, run_document_benchmark, run_sentence_benchmark

logger = logging.getLogger(__name__)


async def start_workers(queue: TaskQueue, settings: Optional[ServiceSettings] = None):
    settings = settings or get_settings()
    pipelines = [RetrievalPipeline(settings) for _ in range(settings.worker_count)]

    def process_check_task(task, metadata: Dict[str, str], pipeline: RetrievalPipeline):
        content = metadata.get("content")
        if not content:
            raise ValueError("Missing content")

        sentences = pipeline.preprocess(content)
        vectors = pipeline.encode_sentences(sentences)

        citation_index = CitationIndex.build(content, sentences, settings.citation_window)
        status_priority = {
            SentenceStatus.IDENTICAL: 5,
            SentenceStatus.CITED: 4,
            SentenceStatus.MINOR_CHANGES: 3,
            SentenceStatus.PARAPHRASED: 2,
            SentenceStatus.ORIGINAL: 1,
        }

        sentence_reports: List[SentenceReport] = []

        for idx, (sentence, vec) in enumerate(zip(sentences, vectors)):
            results = pipeline.search_similar(vec)
            if not results:
                sentence_reports.append(
                    SentenceReport(
                        index=idx,
                        text=sentence,
                        status=SentenceStatus.ORIGINAL,
                        similarity_score=0.0,
                        semantic_score=0.0,
                        lexical_score=0.0,
                        has_citation=False,
                        matched_source=None,
                    )
                )
                continue

            best_candidate = None

            for match in results[: settings.rerank_top_k]:
                payload = match.payload or {}
                candidate_sentence = payload.get("text", "")
                semantic_score = float(match.score or 0.0)
                lexical_score = normalized_levenshtein(sentence, candidate_sentence)
                has_citation = citation_index.has_citation(idx, payload.get("paper_id"))
                status = SentenceStatus(
                    decision_tree_classification(semantic_score, lexical_score, has_citation, settings)
                )
                similarity_score = weighted_score(semantic_score, lexical_score)

                candidate = {
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "similarity_score": similarity_score,
                    "has_citation": has_citation,
                    "status": status,
                    "payload": payload,
                }

                if (
                    best_candidate is None
                    or similarity_score > best_candidate["similarity_score"]
                    or (
                        similarity_score == best_candidate["similarity_score"]
                        and status_priority[status] > status_priority[best_candidate["status"]]
                    )
                ):
                    best_candidate = candidate

            if best_candidate is None:
                match = results[0]
                payload = match.payload or {}
                semantic_score = float(match.score or 0.0)
                lexical_score = normalized_levenshtein(sentence, payload.get("text", ""))
                has_citation = citation_index.has_citation(idx, payload.get("paper_id"))
                status = SentenceStatus(
                    decision_tree_classification(semantic_score, lexical_score, has_citation, settings)
                )
                best_candidate = {
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "similarity_score": weighted_score(semantic_score, lexical_score),
                    "has_citation": has_citation,
                    "status": status,
                    "payload": payload,
                }

            sentence_reports.append(
                SentenceReport(
                    index=idx,
                    text=sentence,
                    status=best_candidate["status"],
                    similarity_score=best_candidate["similarity_score"],
                    semantic_score=best_candidate["semantic_score"],
                    lexical_score=best_candidate["lexical_score"],
                    has_citation=best_candidate["has_citation"],
                    matched_source=MatchedSource(
                        sentence=best_candidate["payload"].get("text", ""),
                        paper_id=best_candidate["payload"].get("paper_id"),
                        archive_path=best_candidate["payload"].get("archive_path"),
                        member=best_candidate["payload"].get("member"),
                        sent_index=best_candidate["payload"].get("sent_index"),
                        similarity_score=best_candidate["semantic_score"],
                        lexical_score=best_candidate["lexical_score"],
                    ),
                )
            )

        report = aggregate_sentence_reports(sentence_reports)
        return queue.set_completed(task.task_id, report=report)

    def process_benchmark_task(task, metadata: Dict[str, str]):
        data_path_raw = metadata.get("data_path")
        if not data_path_raw:
            raise ValueError("Benchmark task missing data_path")

        data_path = Path(data_path_raw)
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data_path not found: {data_path}")

        kind_value = metadata.get("benchmark_kind")
        if not kind_value:
            raise ValueError("Benchmark task missing benchmark_kind")

        try:
            kind = BenchmarkKind(kind_value)
        except ValueError as exc:
            raise ValueError(f"Unknown benchmark kind: {kind_value}") from exc

        cfg_kwargs: Dict[str, Any] = {}
        config_json = metadata.get("benchmark_config")
        if config_json:
            cfg_kwargs = json.loads(config_json)

        cfg = BenchConfig(**cfg_kwargs)

        if kind == BenchmarkKind.SENTENCES:
            samples = load_sentence_samples(data_path)
            result = run_sentence_benchmark(samples, cfg)
        else:
            docs = load_document_samples(data_path)
            result = run_document_benchmark(docs, cfg)

        payload = BenchmarkPayload(kind=kind, data_path=str(data_path), config=asdict(cfg), result=result)
        return queue.set_completed(task.task_id, benchmark=payload)

    async def worker_loop(worker_id: int, pipeline: RetrievalPipeline):
        while True:
            task = queue.fetch_next()
            if not task:
                await asyncio.sleep(0.1)
                continue

            try:
                metadata = task.metadata or {}
                task_type = metadata.get("task_type", "check")

                if task_type == "benchmark":
                    final_state = process_benchmark_task(task, metadata)
                else:
                    final_state = process_check_task(task, metadata, pipeline)

                callback_url = final_state.metadata.get("callback_url") if final_state.metadata else None
                if callback_url:
                    await post_callback(callback_url, final_state)

            except Exception as exc:
                logger.exception("Task %s failed", task.task_id)
                queue.set_failed(task.task_id, str(exc))

    await asyncio.gather(*[worker_loop(i, pipeline) for i, pipeline in enumerate(pipelines)])


async def post_callback(url: str, state):
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(url, json=state.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Callback to %s failed: %s", url, exc)