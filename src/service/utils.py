from __future__ import annotations

import math
from typing import Sequence


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = previous_row[j] + 1
            delete_cost = current_row[j - 1] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    try:
        import rapidfuzz.distance as rf_distance

        dist = rf_distance.Levenshtein.distance(a, b)
    except Exception:
        try:
            import Levenshtein  # type: ignore

            dist = Levenshtein.distance(a, b)
        except Exception:
            dist = _levenshtein_distance(a, b)
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (dist / denom)


def decision_tree_classification(
    semantic_score: float,
    lexical_score: float,
    has_citation: bool,
    settings,
):
    if lexical_score >= settings.t_lev_high:
        status = "identical"
    elif lexical_score >= settings.t_lev_med and semantic_score >= settings.t_cos_high:
        status = "minor_changes"
    elif semantic_score >= settings.t_cos_mid:
        status = "paraphrased"
    else:
        status = "original"

    if has_citation and status in {"minor_changes", "paraphrased"}:
        return "cited"

    return status


def weighted_score(semantic_score: float, lexical_score: float, w_cos: float = 0.7, w_lev: float = 0.3) -> float:
    return semantic_score * w_cos + lexical_score * w_lev


def overall_similarity_score(counts, total_sentences: int) -> float:
    if total_sentences == 0:
        return 0.0
    return (
        counts.get("identical", 0) * 1.0
        + counts.get("minor_changes", 0) * 0.8
        + counts.get("paraphrased", 0) * 0.6
    ) / total_sentences


def softmax(scores: Sequence[float]) -> Sequence[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    if total == 0:
        return [0.0 for _ in scores]
    return [s / total for s in exp_scores]
