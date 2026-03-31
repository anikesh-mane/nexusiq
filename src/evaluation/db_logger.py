"""
SQLite logger for RAGAS and pipeline metrics.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from loguru import logger
from src.config import config


DB_PATH = Path(config.METRICS_DB_PATH)


@contextmanager
def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename    TEXT,
                doc_type    TEXT,
                confidence  REAL,
                issue_count INTEGER,
                rec_count   INTEGER,
                elapsed_sec REAL
            );

            CREATE TABLE IF NOT EXISTS ragas_metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename        TEXT,
                faithfulness    REAL,
                answer_relevance REAL,
                context_recall  REAL,
                context_precision REAL
            );
            """
        )
    logger.debug(f"DB initialized: {DB_PATH}")


def log_pipeline_run(result: dict) -> int:
    """Insert a pipeline run record and return the row id."""
    init_db()
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO pipeline_runs
                (filename, doc_type, confidence, issue_count, rec_count, elapsed_sec)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("document"),
                result.get("document_type"),
                result.get("confidence_score"),
                result.get("validation", {}).get("issue_count", 0),
                len(result.get("recommendations", [])),
                result.get("processing_time_seconds"),
            ),
        )
    row_id = cur.lastrowid
    logger.info(f"Logged pipeline run to DB (row_id={row_id})")
    return row_id


def log_ragas_metrics(filename: str, metrics: dict) -> None:
    """Insert RAGAS evaluation metrics."""
    init_db()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ragas_metrics
                (filename, faithfulness, answer_relevance, context_recall, context_precision)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                metrics.get("faithfulness"),
                metrics.get("answer_relevance"),
                metrics.get("context_recall"),
                metrics.get("context_precision"),
            ),
        )
    logger.info(f"Logged RAGAS metrics for: {filename}")
