"""Tests for document ingestion / parsing."""
from pathlib import Path
import pytest


def test_parse_txt_file(tmp_path: Path):
    """Plain text file should parse successfully without Docling."""
    from src.ingestion.parser import parse_document

    sample = tmp_path / "sample.txt"
    sample.write_text("Hello, this is a test invoice.", encoding="utf-8")

    text = parse_document(sample)
    assert "test invoice" in text


def test_parse_missing_file():
    """Missing file should raise FileNotFoundError."""
    from src.ingestion.parser import parse_document

    with pytest.raises(FileNotFoundError):
        parse_document("/nonexistent/path/doc.txt")


def test_parse_markdown_file(tmp_path: Path):
    """Markdown file should parse as plain text."""
    from src.ingestion.parser import parse_document

    md = tmp_path / "readme.md"
    md.write_text("# Contract\nParty A agrees with Party B.", encoding="utf-8")

    text = parse_document(md)
    assert "Party A" in text
