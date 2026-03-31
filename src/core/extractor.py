"""
Extractor — pulls structured entities from document text using the LLM.
"""
import json
from loguru import logger
from src.llm.client import extract_document
from src.llm.prompt_manager import prompt_manager


def extract_entities(content: str, document_type: str) -> dict:
    """
    Extract structured key entities from document content.

    Args:
        content: Full document text.
        document_type: One of invoice, contract, email, report, other.

    Returns:
        Dict of extracted fields (values may be None if not found).
    """
    # Send full content (the LLM handles token limits internally)
    prompt = prompt_manager.render(
        "extraction",
        document_type=document_type,
        content=content[:6000],  # safe cap
    )

    logger.info(f"Extracting entities for document type: {document_type}")
    try:
        entities = extract_document(prompt, document_type)
    except ValueError:
        logger.warning("Extraction returned bad JSON — returning empty dict")
        entities = {}

    logger.info(f"Extracted {len(entities)} top-level fields")
    return entities
