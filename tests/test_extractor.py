"""Tests for entity extractor (mocks the LLM call)."""
from unittest.mock import patch
import pytest


SAMPLE_INVOICE_ENTITIES = {
    "vendor": "Acme Corp",
    "buyer": "NexusIQ Ltd",
    "invoice_number": "INV-2024-001",
    "amount": "5000",
    "currency": "USD",
    "date": "2024-01-15",
    "due_date": "2024-02-15",
    "line_items": [],
    "payment_terms": "Net 30",
    "tax": "500",
}


def test_extract_entities_invoice():
    """Extractor should return dict with expected keys for invoice type."""
    from src.core.extractor import extract_entities

    with patch("src.core.extractor.call_llm_json", return_value=SAMPLE_INVOICE_ENTITIES):
        result = extract_entities("dummy content", "invoice")

    assert result["vendor"] == "Acme Corp"
    assert result["amount"] == "5000"
    assert "invoice_number" in result


def test_extract_entities_bad_json():
    """Extractor should return empty dict on LLM JSON failure."""
    from src.core.extractor import extract_entities

    with patch("src.core.extractor.call_llm_json", side_effect=ValueError("bad json")):
        result = extract_entities("dummy content", "invoice")

    assert result == {}
