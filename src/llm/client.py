"""
Gemini API client — thin wrapper around google-genai.
"""

from typing import Any, Optional
from pydantic import BaseModel

from loguru import logger

from google import genai
from google.genai import types

from src.config import config


# Pydantic schemas

class DocumentClassification(BaseModel):
    document_type: str  # invoice | contract | email | report | other
    confidence: float
    reasoning: str


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceExtraction(BaseModel):
    vendor: Optional[str] = None
    buyer: Optional[str] = None
    invoice_number: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    line_items: list[LineItem] = []
    payment_terms: Optional[str] = None
    tax: Optional[float] = None


class ContractExtraction(BaseModel):
    parties: list[str] = []
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    value: Optional[float] = None
    jurisdiction: Optional[str] = None
    key_obligations: list[str] = []
    termination_clause: Optional[str] = None


class EmailExtraction(BaseModel):
    sender: Optional[str] = None
    recipients: list[str] = []
    subject: Optional[str] = None
    date: Optional[str] = None
    action_items: list[str] = []
    sentiment: Optional[str] = None

class Recommendation(BaseModel):
    action: str
    reasoning: str
    priority: str # "high|medium|low"


class GenericExtraction(BaseModel):
    fields: dict[str, Any] = {}


# Map document types to their extraction schema
EXTRACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "contract": ContractExtraction,
    "email": EmailExtraction,
    "report": GenericExtraction,
    "other": GenericExtraction,
}



# Client

def _init_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Check your .env file."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


_client: genai.Client | None = None


def get_model() -> genai.Client:
    global _client
    if _client is None:
        _client = _init_client()
    return _client



# Core LLM helpers

def call_llm(prompt: str) -> str:
    """Send a prompt to Gemini and return the raw text response."""
    client = get_model()
    logger.debug(f"Calling Gemini [{config.GEMINI_MODEL}] — prompt length={len(prompt)}")
    response = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
    text = response.text.strip()
    logger.debug(f"LLM response length: {len(text)}")
    return text


def call_llm_json(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """
    Call Gemini with a Pydantic schema enforced via response_schema.
    Returns a validated Pydantic model instance.
    """
    client = get_model()
    logger.debug(
        f"Calling Gemini [{config.GEMINI_MODEL}] with schema={schema.__name__}, "
        f"prompt length={len(prompt)}"
    )
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        ),
    )
    return schema.model_validate_json(response.text)



# Document pipeline

def classify_document(prompt) -> DocumentClassification:
    """Classify a document and return a validated DocumentClassification."""
    return call_llm_json(prompt, DocumentClassification)


def extract_document(prompt, document_type) -> BaseModel:
    """Extract structured fields using the schema matching document_type."""
    schema = EXTRACTION_SCHEMAS.get(document_type, GenericExtraction)
    return call_llm_json(prompt, schema)


def recommend_actions(prompt) -> Recommendation:
    """Generate a list of recommended next actions for this document."""
    return call_llm_json(prompt, Recommendation)


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for the given text using Gemini."""
    client = get_model()
    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values