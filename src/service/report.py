from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .models import ReportPayload, ReportSummary, SentenceReport, SentenceStatus, SourceContribution
from .utils import overall_similarity_score


def aggregate_sentence_reports(sentence_reports: Sequence[SentenceReport]) -> ReportPayload:
    counts = Counter([sr.status for sr in sentence_reports])

    summary = ReportSummary(
        total_sentences=len(sentence_reports),
        identical_count=counts.get(SentenceStatus.IDENTICAL, 0),
        minor_changes_count=counts.get(SentenceStatus.MINOR_CHANGES, 0),
        paraphrased_count=counts.get(SentenceStatus.PARAPHRASED, 0),
        cited_count=counts.get(SentenceStatus.CITED, 0),
        original_count=counts.get(SentenceStatus.ORIGINAL, 0),
    )

    overall_score = overall_similarity_score(
        {
            "identical": summary.identical_count,
            "minor_changes": summary.minor_changes_count,
            "paraphrased": summary.paraphrased_count,
        },
        summary.total_sentences,
    )

    source_scores: Dict[str, float] = defaultdict(float)
    source_counts: Dict[str, int] = defaultdict(int)

    for sr in sentence_reports:
        if sr.matched_source and sr.matched_source.paper_id:
            paper_id = sr.matched_source.paper_id
            weight = 1.0
            if sr.status == SentenceStatus.IDENTICAL:
                weight = 1.0
            elif sr.status == SentenceStatus.MINOR_CHANGES:
                weight = 0.8
            elif sr.status == SentenceStatus.PARAPHRASED:
                weight = 0.6
            elif sr.status == SentenceStatus.CITED:
                weight = 0.4
            source_scores[paper_id] += weight
            source_counts[paper_id] += 1

    max_score = max(source_scores.values(), default=1.0)
    contributions = [
        SourceContribution(
            paper_id=paper_id,
            score=score,
            sentence_count=source_counts[paper_id],
            weight=score / max_score if max_score else 0.0,
            extra={},
        )
        for paper_id, score in sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return ReportPayload(
        overall_similarity_score=overall_score,
        summary=summary,
        top_sources=contributions[:5],
        sentence_details=list(sentence_reports),
    )
