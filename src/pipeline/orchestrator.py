"""
Pipeline Orchestrator — ties all components together.
"""
import hashlib
import time
from pathlib import Path
from typing import Any
from loguru import logger

from src.config import config
from src.ingestion.parser import parse_document
from src.core.classifier import classify_document
from src.core.extractor import extract_entities
from src.core.validator import validate
from src.core.recommender import recommend_actions
from src.rag.vector_store import add_document, retrieve_similar
from src.utils.formatters import sanitize_output


def run_pipeline(file_path: str | Path) -> dict[str, Any]:
    """
    Full document intelligence pipeline.

    Steps:
      1. Parse document -> raw text
      2. Retrieve similar past docs from ChromaDB (RAG context)
      3. Classify document type
      4. Extract structured entities
      5. Validate / detect anomalies
      6. Generate action recommendations
      7. Index current document in vector store
      8. Return structured output

    Returns:
        Comprehensive result dict ready for JSON serialization.
    """
    path = Path(file_path)
    start = time.perf_counter()
    logger.info(f"=== Pipeline START: {path.name} ===")

    # ---------- 1. Parse ----------
    content = parse_document(path)
    doc_id = _make_doc_id(path, content)

    # ---------- 3. Classify ----------
    classification = classify_document(content)
    doc_type = classification["document_type"]

    # ---------- 2. Index ----------
    add_document(
        doc_id=doc_id,
        text=content[:2000],  # store excerpt
        metadata={
            "filename": path.name,
            "document_type": doc_type,
            "confidence": classification["confidence"],
        },
    )

    # ---------- 4. Extract ----------
    entities = extract_entities(content, doc_type)
    entities = sanitize_output(entities)

    # ---------- 5. Validate ----------
    issues = validate(entities, doc_type)

    # ---------- 6. Recommend ----------
    recommendations = recommend_actions(doc_type, entities, issues)

    # # ---------- 7. RAG context ----------
    # similar_docs = retrieve_similar(query=content[:500], n_results=3)
    # rag_context = [d["document"][:300] for d in similar_docs]
    # logger.info(f"RAG: retrieved {len(similar_docs)} similar document(s)")


    elapsed = time.perf_counter() - start
    logger.info(f"=== Pipeline END: {path.name} ({elapsed:.2f}s) ===")

    return {
        "document": path.name,
        "document_type": doc_type,
        "confidence_score": classification["confidence"],
        "classification_reasoning": classification.get("reasoning", ""),
        "key_entities": entities,
        "validation": {
            "issue_count": len(issues),
            "issues": issues,
        },
        "recommendations": recommendations,
        # "rag_context_used": len(similar_docs) > 0,
        # "similar_documents": [d["id"] for d in similar_docs],
        "processing_time_seconds": round(elapsed, 3),
        "raw_content": content,  # full parsed text — used by the RAG chatbot
    }


def _make_doc_id(path: Path, content: str) -> str:
    """Generate a stable ID from filename + content hash."""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{path.stem}_{content_hash}"
