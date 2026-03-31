"""
Recommender — derives next-best-actions using the LLM given extraction + validation context.
"""
import json
from loguru import logger
from src.llm.client import recommend_actions
from src.llm.prompt_manager import prompt_manager
from src.utils.formatters import to_pretty_json


def recommend_actions(
    document_type: str,
    entities: dict,
    validation_issues: list[dict],
) -> list[dict]:
    """
    Generate a list of recommended next actions for this document.

    Returns:
        [{"action": str, "reason": str, "priority": str}]
    """
    issues_text = (
        to_pretty_json(validation_issues) if validation_issues else "None"
    )
    entities_text = to_pretty_json(entities)

    prompt = prompt_manager.render(
        "recommendation",
        document_type=document_type,
        extracted_data=entities_text,
        validation_issues=issues_text,
    )

    logger.info("Generating action recommendations...")
    try:
        recommendations = recommend_actions(prompt)
        if isinstance(recommendations, dict):
            # Sometimes the LLM wraps the array in a key
            recommendations = next(
                (v for v in recommendations.values() if isinstance(v, list)),
                [],
            )
    except ValueError:
        logger.warning("Recommendation returned bad JSON — falling back to heuristics.")
        recommendations = _heuristic_recommendations(document_type, validation_issues)

    logger.info(f"Generated {len(recommendations)} recommendation(s)")
    return recommendations


def _heuristic_recommendations(
    document_type: str, issues: list[dict]
) -> list[dict]:
    """Simple rule-based fallback recommendations."""
    recs = []
    severities = {i["type"] for i in issues}

    if "overdue" in severities:
        recs.append({
            "action": "Send payment reminder",
            "reason": "Invoice is past due date.",
            "priority": "high",
        })
    if "missing_field" in severities:
        recs.append({
            "action": "Request missing information",
            "reason": "Required fields are missing from the document.",
            "priority": "high",
        })
    if "suspicious_value" in severities:
        recs.append({
            "action": "Flag for manual review",
            "reason": "Document contains suspicious values.",
            "priority": "medium",
        })
    if not recs:
        recs.append({
            "action": "Approve and process document",
            "reason": "No issues detected.",
            "priority": "low",
        })
    return recs
