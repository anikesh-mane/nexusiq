"""
Classifier — determines the document type using the LLM.
"""
from loguru import logger
from src.llm.client import classifier
from src.llm.prompt_manager import prompt_manager


VALID_TYPES = {"invoice", "contract", "email", "report", "other"}


def classify_document(content: str) -> dict:
    """
    Classify a document and return its type and confidence.

    Returns:
        {
            "document_type": str,
            "confidence": float,
            "reasoning": str
        }
    """
    # Truncate to avoid token limits (first 4000 chars is sufficient for classification)
    snippet = content[:4000]
    prompt = prompt_manager.render("classification", content=snippet)

    logger.info("Classifying document...")
    result = classifier(prompt).model_dump()

    doc_type = result.get("document_type", "other").lower().strip()
    if doc_type not in VALID_TYPES:
        logger.warning(f"Unknown doc type '{doc_type}' — defaulting to 'other'")
        doc_type = "other"

    result["document_type"] = doc_type
    result["confidence"] = float(result.get("confidence", 0.5))
    logger.info(f"Classified as: {doc_type} (confidence={result['confidence']:.2f})")
    return result
