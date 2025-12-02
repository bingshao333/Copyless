from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

_ARXIV_ID_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
_INLINE_LABEL_RE = re.compile(r"\[([^\[\]]+)\]")
_LABEL_ENTRY_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+)$")
_SPLIT_LABEL_RE = re.compile(r"[,;]")


def _normalize_arxiv_id(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if not text.lower().startswith("arxiv:"):
        return text
    core = text[len("arxiv:") :]
    core = core.split("v", 1)[0]
    return f"arXiv:{core}"


def _collect_reference_entries(lines: Sequence[str]) -> Mapping[str, Set[str]]:
    """Attempt to map reference labels (e.g. [1]) to arXiv ids."""
    entries: Dict[str, Set[str]] = {}
    for raw in lines:
        if not raw.strip():
            continue
        match = _LABEL_ENTRY_RE.match(raw)
        if not match:
            continue
        label, body = match.group(1).strip(), match.group(2)
        ids = {_normalize_arxiv_id(x) for x in _ARXIV_ID_RE.findall(body)}
        ids = {pid for pid in ids if pid}
        if not ids:
            continue
        entries[label] = ids
    return entries


def _split_reference_section(content: str) -> Sequence[str]:
    lines = content.splitlines()
    start: Optional[int] = None
    for idx, line in enumerate(lines):
        if _SECTION_RE.match(line):
            start = idx + 1
            break
    if start is None:
        return []
    block_lines: List[str] = []
    buffer: List[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if buffer:
                block_lines.append(" ".join(buffer))
                buffer.clear()
            continue
        buffer.append(stripped)
    if buffer:
        block_lines.append(" ".join(buffer))
    return block_lines


def _labels_from_inline(text: str) -> Iterable[str]:
    for match in _INLINE_LABEL_RE.finditer(text):
        chunk = match.group(1)
        for piece in _SPLIT_LABEL_RE.split(chunk):
            label = piece.strip()
            if label:
                yield label


@dataclass
class CitationIndex:
    sentence_ids: Mapping[int, Set[str]]

    @classmethod
    def build(cls, content: str, sentences: Sequence[str], window: int) -> "CitationIndex":
        if not content:
            return cls(sentence_ids={})
        reference_lines = _split_reference_section(content)
        label_map = _collect_reference_entries(reference_lines)
        sentence_citations: List[Set[str]] = [set() for _ in sentences]
        for idx, sentence in enumerate(sentences):
            direct_ids = {_normalize_arxiv_id(x) for x in _ARXIV_ID_RE.findall(sentence)}
            for label in _labels_from_inline(sentence):
                ids = label_map.get(label)
                if ids:
                    direct_ids.update(ids)
            normalized = {pid for pid in direct_ids if pid}
            if normalized:
                sentence_citations[idx].update(normalized)
        if window < 0:
            window = 0
        windowed: Dict[int, Set[str]] = {}
        n = len(sentence_citations)
        for idx, ids in enumerate(sentence_citations):
            if not ids:
                continue
            start = max(0, idx - window)
            end = min(n - 1, idx + window)
            for pos in range(start, end + 1):
                windowed.setdefault(pos, set()).update(ids)
        return cls(sentence_ids=windowed)

    def has_citation(self, sentence_index: int, paper_id: Optional[str]) -> bool:
        if paper_id is None:
            return False
        normalized = _normalize_arxiv_id(paper_id)
        if not normalized:
            return False
        cited = self.sentence_ids.get(sentence_index)
        if not cited:
            return False
        return normalized in cited
