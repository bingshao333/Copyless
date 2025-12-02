from pathlib import Path
from typing import Optional

import re

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception:  # pragma: no cover
    LatexNodes2Text = None

# 定义PDF文本提取函数
def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) 未安装，无法解析 PDF")
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)

# 定义LaTeX文本提取函数
def extract_text_from_tex(path: Path) -> str:
    text = path.read_text(errors="ignore")
    if LatexNodes2Text is not None:
        return LatexNodes2Text().latex_to_text(text)
    # 简单回退：去除常见 LaTeX 命令
    text = re.sub(r"\\(section|subsection|textbf|emph|cite|ref|label)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"\{\}|\$|~|%", " ", text)
    return text

# 根据后缀选择解析方法
def extract_text(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in {".pdf"}:
        return extract_text_from_pdf(path)
    if suffix in {".tex"}:
        return extract_text_from_tex(path)
    return None
