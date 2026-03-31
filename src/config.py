import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Config:
    # --- Gemini ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw"
    CHROMA_DB_PATH: str = str(BASE_DIR / "data" / "chroma_db")
    METRICS_DB_PATH: str = str(BASE_DIR / "db" / "metrics.db")
    PROMPTS_PATH: Path = BASE_DIR / "prompts"

    # --- ChromaDB ---
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "nexusiq_docs")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "models/text-embedding-004"
    )

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Validation thresholds ---
    HIGH_AMOUNT_THRESHOLD: float = float(os.getenv("HIGH_AMOUNT_THRESHOLD", "100000"))


config = Config()
