"""
ChromaDB vector store — setup, indexing, and retrieval.
Uses Gemini embeddings via a custom embedding function.
"""
from typing import Any
from loguru import logger

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from src.config import config
from src.llm.client import embed_text


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function backed by Gemini."""

    def __call__(self, input: Documents) -> Embeddings:
        return [embed_text(doc) for doc in input]


def get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=GeminiEmbeddingFunction(),
    )
    logger.debug(
        f"ChromaDB collection '{config.CHROMA_COLLECTION}' ready "
        f"({collection.count()} documents)"
    )
    return collection


def add_document(doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
    """Embed and store a document chunk in ChromaDB."""
    collection = get_collection()
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata or {}],
    )
    logger.info(f"Indexed document chunk: {doc_id}")


def retrieve_similar(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    """
    Retrieve the top-n most similar documents for a query.

    Returns a list of dicts with keys: id, document, metadata, distance.
    """
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("Vector store is empty — no similar docs to retrieve.")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        output.append(
            {
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return output
