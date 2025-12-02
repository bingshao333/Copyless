from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def confusion_from_sets(pred: Iterable[Tuple], truth: Iterable[Tuple]) -> Dict[str, int]:
    p = set(pred)
    t = set(truth)
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    return {"tp": tp, "fp": fp, "fn": fn}


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def percentiles(values: Sequence[float], ps: Sequence[float] = (50, 95, 99)) -> Dict[str, float]:
    import numpy as np

    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}
    arr = np.asarray(values, dtype=float)
    out = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(arr, p))
    return out


def latency_summary(latencies_ms: Sequence[float]) -> Dict[str, float]:
    import numpy as np

    if not latencies_ms:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def throughput(n: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf") if n > 0 else 0.0
    return float(n) / float(seconds)
