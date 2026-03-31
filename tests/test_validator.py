"""Tests for business rule validator."""
import pytest
from src.core.validator import validate


def test_missing_required_fields_invoice():
    """Missing vendor and amount should produce high-severity issues."""
    issues = validate({}, "invoice")
    types = [i["type"] for i in issues]
    assert "missing_field" in types
    assert all(i["severity"] == "high" for i in issues if i["type"] == "missing_field")


def test_high_amount_triggers_suspicious():
    """Amount above threshold should trigger suspicious_value."""
    entities = {
        "vendor": "Acme",
        "amount": "999999",
        "date": "2024-01-01",
        "due_date": "2024-12-31",
        "invoice_number": "INV-001",
    }
    issues = validate(entities, "invoice")
    types = [i["type"] for i in issues]
    assert "suspicious_value" in types


def test_overdue_invoice():
    """Past due date should trigger overdue issue."""
    entities = {
        "vendor": "Acme",
        "amount": "1000",
        "date": "2023-01-01",
        "due_date": "2023-02-01",   # clearly in the past
        "invoice_number": "INV-001",
    }
    issues = validate(entities, "invoice")
    types = [i["type"] for i in issues]
    assert "overdue" in types


def test_valid_invoice_no_issues():
    """A well-formed future invoice should have no issues."""
    from datetime import date, timedelta

    today = date.today()
    future = (today + timedelta(days=30)).isoformat()
    entities = {
        "vendor": "Acme",
        "buyer": "Client Co",
        "amount": "500",
        "date": today.isoformat(),
        "due_date": future,
        "invoice_number": "INV-999",
    }
    issues = validate(entities, "invoice")
    assert issues == []


def test_contract_date_conflict():
    """Expiry before effective date should be a date_conflict issue."""
    entities = {
        "parties": ["A", "B"],
        "effective_date": "2025-06-01",
        "expiry_date": "2025-01-01",  # before effective
    }
    issues = validate(entities, "contract")
    types = [i["type"] for i in issues]
    assert "date_conflict" in types
