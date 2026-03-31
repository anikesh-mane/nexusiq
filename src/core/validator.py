"""
Validator — detects anomalies, missing fields, and business rule violations.
"""
from datetime import date, datetime
from typing import Any
from loguru import logger
from src.config import config


# Required fields per document type
REQUIRED_FIELDS: dict[str, list[str]] = {
    "invoice":  ["vendor", "amount", "date", "due_date", "invoice_number"],
    "contract": ["parties", "effective_date", "expiry_date"],
    "email":    ["sender", "subject", "date"],
    "report":   [],
    "other":    [],
}


def _parse_amount(value: Any) -> float | None:
    """Try to parse a numeric amount from various formats."""
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace("$", "").replace("€", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_date(value: Any) -> date | None:
    """Try to parse a date string."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def validate(entities: dict[str, Any], document_type: str) -> list[dict[str, str]]:
    """
    Run validation checks on extracted entities.

    Returns:
        List of issue dicts: [{"type": str, "message": str, "severity": str}]
    """
    issues: list[dict[str, str]] = []
    required = REQUIRED_FIELDS.get(document_type, [])

    # --- Missing field checks ---
    for field in required:
        val = entities.get(field)
        is_empty = val is None or val == "" or val == [] or val == {}
        if is_empty:
            issues.append({
                "type": "missing_field",
                "message": f"Required field '{field}' is missing or empty.",
                "severity": "high",
            })
            logger.debug(f"Validation: missing field '{field}'")

    # --- Invoice-specific checks ---
    if document_type == "invoice":
        amount = _parse_amount(entities.get("amount"))
        if amount is not None:
            if amount <= 0:
                issues.append({
                    "type": "invalid_value",
                    "message": f"Invoice amount ({amount}) must be positive.",
                    "severity": "high",
                })
            elif amount > config.HIGH_AMOUNT_THRESHOLD:
                issues.append({
                    "type": "suspicious_value",
                    "message": (
                        f"Invoice amount ({amount:,.2f}) exceeds threshold "
                        f"({config.HIGH_AMOUNT_THRESHOLD:,.0f}). Flag for review."
                    ),
                    "severity": "medium",
                })

        # Date vs due_date sanity
        invoice_date = _parse_date(entities.get("date"))
        due_date = _parse_date(entities.get("due_date"))
        today = date.today()

        if invoice_date and invoice_date > today:
            issues.append({
                "type": "suspicious_value",
                "message": f"Invoice date ({invoice_date}) is in the future.",
                "severity": "medium",
            })

        if invoice_date and due_date and due_date < invoice_date:
            issues.append({
                "type": "date_conflict",
                "message": "Due date is before the invoice date.",
                "severity": "high",
            })

        if due_date and due_date < today:
            issues.append({
                "type": "overdue",
                "message": f"Invoice is overdue (due: {due_date}).",
                "severity": "high",
            })

    # --- Contract-specific checks ---
    if document_type == "contract":
        effective = _parse_date(entities.get("effective_date"))
        expiry = _parse_date(entities.get("expiry_date"))
        today = date.today()

        if effective and expiry and expiry <= effective:
            issues.append({
                "type": "date_conflict",
                "message": "Contract expiry date is not after the effective date.",
                "severity": "high",
            })

        if expiry and expiry < today:
            issues.append({
                "type": "expired",
                "message": f"Contract has expired on {expiry}.",
                "severity": "high",
            })

    logger.info(f"Validation complete — {len(issues)} issue(s) found")
    return issues
