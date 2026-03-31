"""
Document parser using Docling for PDF, Word, and plain-text documents.
Falls back to plain text read if Docling is not installed.
"""
from pathlib import Path
from loguru import logger


def parse_document(file_path: str | Path) -> str:
    """
    Parse a document and return its text content.

    Supports: PDF, DOCX, TXT, MD, HTML (via Docling).
    Falls back to UTF-8 read for plain-text files when Docling is unavailable.

    Args:
        file_path: Path to the document.

    Returns:
        Extracted text as a single string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    logger.info(f"Parsing document: {path.name} (type={suffix})")

    try:
        from docling.document_converter import DocumentConverter 

        converter = DocumentConverter()
        result = converter.convert(str(path))
        text = result.document.export_to_markdown()
        logger.debug(f"Docling extracted {len(text)} chars from {path.name}")
        return text

    except ImportError:
        logger.warning("Docling not installed — falling back to plain-text read.")
    except Exception as exc:  
        logger.warning(f"Docling failed ({exc}) — falling back to plain-text read.")

    # Plain-text fallback
    if suffix in {".txt", ".md", ".csv", ".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="replace")

    raise RuntimeError(
        f"Cannot parse '{suffix}' files without Docling. "
        "Run: pip install docling"
    )
