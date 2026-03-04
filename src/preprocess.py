from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List

if TYPE_CHECKING:  # pragma: no cover - 仅类型提示
    import spacy

# 预编译一个正则：匹配一个或多个空白字符（空格、制表、换行等）。预编译有助于多次使用时的性能。
_WS_RE = re.compile(r"\s+")
_LIGATURE_TRANSLATIONS = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)

def clean_text(text: str) -> str:
    # Normalize typographic ligatures before any other cleanup.
    text = text.translate(_LIGATURE_TRANSLATIONS)
    # 将不间断空格（Unicode NBSP）替换为普通空格，避免分词/切句异常。
    text = text.replace("\u00A0", " ")
    # 将回车统一为换行，规范换行符，便于后续处理。
    text = text.replace("\r", "\n")
    # 用空格替换制表符、换页符、垂直制表符，消除不可见/控制类空白
    text = re.sub(r"[\t\f\v]", " ", text)
    # 将所有连续空白（含多空格/混合空白）压缩为单个空格，规范空白分布。
    text = _WS_RE.sub(" ", text)
    # 将多于一个的连续换行压缩为单个换行，避免过多空行
    text = re.sub(r"\n{2,}", "\n", text)
    # 去除首尾空白 返回最终清洗文本。
    return text.strip()


def sentences_nltk(text: str) -> List[str]:
    import nltk
    from nltk.tokenize import sent_tokenize

    # Only check/download models once (module-level flag)
    if not getattr(sentences_nltk, "_initialized", False):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab")
        sentences_nltk._initialized = True

    # Call sent_tokenize, strip whitespace per sentence, filter empty strings.
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


_SPACY_PIPELINES: Dict[str, "spacy.language.Language"] = {}


def sentences_spacy(text: str, model: str = "en_core_web_sm") -> List[str]:
    import spacy

    if model not in _SPACY_PIPELINES:
        try:
            _SPACY_PIPELINES[model] = spacy.load(model)
        except OSError as exc:  # pragma: no cover - 依赖外部模型
            raise RuntimeError(
                f"spaCy 模型 {model} 未安装。请先运行: python -m spacy download {model}"
            ) from exc

    nlp = _SPACY_PIPELINES[model]
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def chunk_iter(it: Iterable[str], size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def get_sentence_splitter(name: str) -> Callable[[str], List[str]]:
    """根据名称选择句子切分策略。

    支持：
        - nltk: 使用 NLTK Punkt 分句
        - spacy: 使用 spaCy pipeline (默认 `en_core_web_sm`)
        - mixed: 使用内置的中英混合启发式
    """

    key = name.strip().lower()
    if key == "nltk":
        return sentences_nltk
    if key == "spacy":
        return sentences_spacy
    if key == "mixed":
        return split_mixed_sentences
    raise ValueError(f"Unsupported sentence splitter '{name}'. 可选值: nltk | spacy | mixed")


def split_mixed_sentences(text: str) -> List[str]:
    """对混合中英文文本进行简单按句切分。

    规则：
    - 使用中文终止符：。！？
    - 使用英文终止符：.!?
    - 保留终止符在句尾
    - 去除首尾空白，过滤空串
    (该实现为启发式，若需更复杂语义切分可替换为 NLP 模型)
    """
    text = clean_text(text)
    # 捕获以终止符结束的最短片段，多行处理；中文英文终止符均支持。
    pattern = re.compile(r"[^。！？.!?]*[。！？.!?]", re.MULTILINE)
    sentences = pattern.findall(text)
    # 处理可能末尾没有终止符的剩余残片
    tail_start = sum(len(s) for s in sentences)
    if tail_start < len(text):
        tail = text[tail_start:].strip()
        if tail:
            sentences.append(tail)
    return [s.strip() for s in sentences if s.strip()]
