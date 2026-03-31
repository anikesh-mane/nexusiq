"""
RAGAS metric calculation — evaluates RAG quality against a reference answer.
Requires: pip install ragas datasets
"""
from typing import Any
from loguru import logger


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict[str, float | None]:
    """
    Compute RAGAS metrics for a single QA sample.

    Args:
        question:     The query / question asked.
        answer:       The generated answer from the LLM.
        contexts:     List of retrieved context strings.
        ground_truth: Optional reference answer for recall computation.

    Returns:
        Dict with metric names and scores (None if metric skipped).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
    except ImportError:
        logger.warning(
            "RAGAS / datasets not installed. Skipping metrics. "
            "Run: pip install ragas datasets"
        )
        return {
            "faithfulness": None,
            "answer_relevance": None,
            "context_recall": None,
            "context_precision": None,
        }

    sample = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    metrics_to_run = [faithfulness, answer_relevancy, context_precision]

    if ground_truth:
        sample["ground_truth"] = [ground_truth]
        metrics_to_run.append(context_recall)

    dataset = Dataset.from_dict(sample)

    try:
        scores = evaluate(dataset, metrics=metrics_to_run)
        result = {
            "faithfulness": scores.get("faithfulness"),
            "answer_relevance": scores.get("answer_relevancy"),
            "context_recall": scores.get("context_recall"),
            "context_precision": scores.get("context_precision"),
        }
        logger.info(f"RAGAS metrics: {result}")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error(f"RAGAS evaluation failed: {exc}")
        return {
            "faithfulness": None,
            "answer_relevance": None,
            "context_recall": None,
            "context_precision": None,
        }
