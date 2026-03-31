"""
RAG Chatbot — interactive Q&A grounded to the processed document and pipeline outputs.

Uses the Gemini native Chat Session API (client.chats.create) so the SDK manages
conversation history automatically. The pipeline context is injected once as the
system_instruction. Each user turn retrieves fresh ChromaDB context and sends an
augmented prompt to the stateful chat session.
"""
import json
from typing import Any

from loguru import logger
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt

from src.llm.client import get_model
from src.config import config as app_config
from src.rag.vector_store import retrieve_similar
from src.evaluation.ragas_metrics import compute_ragas_metrics
from src.evaluation.db_logger import log_ragas_metrics

console = Console()


# Context builder

def _build_system_instruction(result: dict[str, Any]) -> str:
    """
    Build the system instruction from the pipeline result.
    This is injected once into GenerateContentConfig and anchors the
    assistant's persona for the entire session.
    """
    doc_content     = result.get("raw_content", "")[:3000]
    doc_type        = result.get("document_type", "unknown")
    confidence      = result.get("confidence_score", 0)
    reasoning       = result.get("classification_reasoning", "")
    entities        = json.dumps(result.get("key_entities", {}), indent=2, default=str)
    issues          = result.get("validation", {}).get("issues", [])
    recommendations = result.get("recommendations", [])

    issues_text = "\n".join(
        f"  - [{i['severity'].upper()}] {i['type']}: {i['message']}"
        for i in issues
    ) or "  None detected."

    recs_text = "\n".join(
        f"  - [{r.get('priority','?').upper()}] {r.get('action','')}: {r.get('reason','')}"
        for r in recommendations
    ) or "  None."

    return f"""You are NexusIQ Assistant, a Smart Document Intelligence Assistant.
Your job is to answer questions based ONLY on the provided document context and pipeline outputs below.
If the answer is not contained in the provided context, state: "I cannot find the answer in the provided documents."
Do not use outside knowledge. Be concise, precise, and professional.

================================
DOCUMENT CONTENT (excerpt — first 3000 chars):
================================
{doc_content}

================================
PIPELINE OUTPUTS:
================================

[CLASSIFICATION]
  Type        : {doc_type}
  Confidence  : {confidence:.0%}
  Reasoning   : {reasoning}

[EXTRACTED ENTITIES]
{entities}

[VALIDATION ISSUES]
{issues_text}

[RECOMMENDATIONS]
{recs_text}
================================
"""


# ChromaDB retrieval

def _retrieve_context(query: str) -> str:
    """
    Query ChromaDB for the top-3 most relevant document chunks.
    Returns a single string joining the retrieved passages.
    Falls back to an empty string if the store is empty or retrieval fails.
    """
    try:
        similar_docs = retrieve_similar(query=query, n_results=3)
        if not similar_docs:
            return ""
        return "\n---\n".join(d["document"] for d in similar_docs)
    except Exception as exc:
        logger.warning(f"ChromaDB retrieval failed: {exc}")
        return ""


# Display helpers

def _print_welcome(result: dict[str, Any]) -> None:
    doc_name = result.get("document", "document")
    doc_type = result.get("document_type", "unknown").upper()
    n_issues = result.get("validation", {}).get("issue_count", 0)
    n_recs   = len(result.get("recommendations", []))

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]NexusIQ Chat[/bold cyan] — Ask anything about [green]{doc_name}[/green]\n"
            f"[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to end  •  "
            f"[bold]summary[/bold] to re-print results  •  "
            f"[bold]clear[/bold] to start a new session[/dim]",
            border_style="cyan",
        )
    )

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("K", style="bold cyan")
    t.add_column("V", style="white")
    t.add_row("Document Type",   doc_type)
    t.add_row("Confidence",      f"{result.get('confidence_score', 0):.0%}")
    t.add_row("Issues Found",    str(n_issues))
    t.add_row("Recommendations", str(n_recs))
    console.print(Panel(t, title="[bold]Session Context[/bold]", border_style="green"))
    console.print()


def _print_summary(result: dict[str, Any]) -> None:
    """Re-print a brief summary of the pipeline result."""
    issues = result.get("validation", {}).get("issues", [])
    recs   = result.get("recommendations", [])

    console.print()
    if issues:
        t = Table(title="Validation Issues", show_header=True, header_style="bold yellow")
        t.add_column("Severity", width=10)
        t.add_column("Type",     width=22)
        t.add_column("Message")
        for issue in issues:
            color = {"high": "red", "medium": "yellow", "low": "dim"}.get(issue["severity"], "white")
            t.add_row(
                f"[{color}]{issue['severity']}[/{color}]",
                issue["type"],
                issue["message"],
            )
        console.print(t)

    if recs:
        console.print()
        t = Table(title="Recommendations", show_header=True, header_style="bold green")
        t.add_column("Priority", width=10)
        t.add_column("Action",   width=30)
        t.add_column("Reason")
        for rec in recs:
            color = {"high": "red", "medium": "yellow", "low": "dim"}.get(
                rec.get("priority", "low"), "white"
            )
            t.add_row(
                f"[{color}]{rec.get('priority', '')}[/{color}]",
                rec.get("action", ""),
                rec.get("reason", ""),
            )
        console.print(t)

    console.print()


# Session factory

def _create_chat_session(system_instruction: str):
    """
    Create a stateful Gemini chat session with the pipeline context baked
    in as the system instruction. The SDK manages conversation history.
    """
    client = get_model()
    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,  # low temperature → factual, grounded answers
    )
    return client.chats.create(
        model=app_config.GEMINI_MODEL,
        config=chat_config,
    )


# Main entry point

def start_chat_session(result: dict[str, Any]) -> None:
    """
    Launch an interactive RAG chatbot grounded to the processed document.

    - Pipeline context is injected once as system_instruction.
    - ChromaDB is queried each turn for fresh relevant passages.
    - The augmented prompt (context + question) is sent to a stateful
      Gemini chat session; conversation history is managed by the SDK.
    - RAGAS metrics are computed and persisted after every answer.

    Args:
        result: The full dict returned by run_pipeline() — must include 'raw_content'.
    """
    system_instruction = _build_system_instruction(result)
    filename           = result.get("document", "unknown")

    _print_welcome(result)

    # Create the stateful chat session once
    chat = _create_chat_session(system_instruction)
    logger.info("Gemini chat session created for document: %s", filename)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # Built-in commands
        if cmd in ("exit", "quit"):
            console.print("[dim]Goodbye! Chat session ended.[/dim]")
            break

        if cmd == "summary":
            _print_summary(result)
            continue

        if cmd == "clear":
            # Create a fresh session (resets SDK-managed history)
            chat = _create_chat_session(system_instruction)
            console.print("[dim]New session started — conversation history cleared.[/dim]\n")
            continue

        if cmd in ("help", "?"):
            console.print(
                "[dim]Commands: [bold]exit[/bold] | [bold]quit[/bold] | "
                "[bold]summary[/bold] | [bold]clear[/bold] | [bold]help[/bold][/dim]\n"
            )
            continue

        # A. Retrieve relevant context from ChromaDB for this query
        retrieved_context = _retrieve_context(user_input)

        # B. Construct the augmented prompt for this turn
        augmented_prompt = f"""Context from documents:
{retrieved_context if retrieved_context else "(No additional context retrieved from vector store.)"}

User Question:
{user_input}"""

        # C. Send the augmented prompt to the stateful chat session
        try:
            with console.status("[dim]Thinking…[/dim]", spinner="dots"):
                response = chat.send_message(augmented_prompt)
            answer = response.text.strip()
        except Exception as exc:
            console.print(f"[red]LLM error:[/red] {exc}\n")
            logger.error(f"Chatbot LLM error: {exc}")
            continue

        # Render response
        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title="[bold green]NexusIQ Assistant[/bold green]",
                border_style="green",
            )
        )
        console.print()

        # D. RAGAS evaluation — compute + persist metrics silently
        try:
            with console.status("[dim]Evaluating response quality…[/dim]", spinner="dots"):
                rag_contexts = [retrieved_context or system_instruction]
                metrics = compute_ragas_metrics(
                    question=user_input,
                    answer=answer,
                    contexts=rag_contexts,
                )
            log_ragas_metrics(filename=filename, metrics=metrics)
            scores_line = "  ".join(
                f"[dim]{k}:[/dim] [cyan]{v:.2f}[/cyan]"
                if v is not None
                else f"[dim]{k}:[/dim] [yellow]n/a[/yellow]"
                for k, v in metrics.items()
            )
            console.print(f"[dim]📊 RAGAS:[/dim] {scores_line}\n")
        except Exception as exc:
            logger.warning(f"RAGAS evaluation skipped: {exc}")
