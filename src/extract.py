from pathlib import Path
from typing import Optional

import logging
import re

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception:  # pragma: no cover
    LatexNodes2Text = None

# Define PDF text extraction function
def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed, cannot parse PDF")
    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", path, exc)
        raise RuntimeError(f"Failed to open PDF {path}: {exc}") from exc
    texts = []
    try:
        for page in doc:
            try:
                texts.append(page.get_text("text"))
            except Exception as exc:
                logger.warning("Failed to extract text from page %d of %s: %s", page.number, path, exc)
    finally:
        doc.close()
    return "\n".join(texts)

# Define LaTeX text extraction function
def extract_text_from_tex(path: Path) -> str:
    try:
        text = path.read_text(errors="ignore")
    except Exception as exc:
        logger.error("Failed to read TeX file %s: %s", path, exc)
        raise RuntimeError(f"Failed to read TeX file {path}: {exc}") from exc
    if LatexNodes2Text is not None:
        try:
            return LatexNodes2Text().latex_to_text(text)
        except Exception as exc:
            logger.warning("pylatexenc failed for %s, falling back to regex: %s", path, exc)
    # Fallback: strip common LaTeX commands
    text = re.sub(r"\\(section|subsection|textbf|emph|cite|ref|label)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"\{\}|\$|~|%", " ", text)
    return text

# Select extraction method by file suffix
def extract_text(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".pdf"}:
            return extract_text_from_pdf(path)
        if suffix in {".tex"}:
            return extract_text_from_tex(path)
    except Exception as exc:
        logger.error("Text extraction failed for %s: %s", path, exc)
        return None
    return None
