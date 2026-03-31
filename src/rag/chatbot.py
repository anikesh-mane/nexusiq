"""
RAG Chatbot — interactive Q&A grounded to the processed document and pipeline outputs.
"""
import json
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint

from src.llm.client import call_llm

console = Console()

#  Context builder

def _build_system_context(result: dict[str, Any]) -> str:
    """
    Serialise the pipeline result into a rich context block for the LLM.
    The raw document content is pulled from result['raw_content'].
    """
    doc_content   = result.get("raw_content", "")[:3000]
    doc_type      = result.get("document_type", "unknown")
    confidence    = result.get("confidence_score", 0)
    reasoning     = result.get("classification_reasoning", "")
    entities      = json.dumps(result.get("key_entities", {}), indent=2, default=str)
    issues        = result.get("validation", {}).get("issues", [])
    recommendations = result.get("recommendations", [])

    issues_text = "\n".join(
        f"  - [{i['severity'].upper()}] {i['type']}: {i['message']}"
        for i in issues
    ) or "  None detected."

    recs_text = "\n".join(
        f"  - [{r.get('priority','?').upper()}] {r.get('action','')}: {r.get('reason','')}"
        for r in recommendations
    ) or "  None."

    return f"""You are NexusIQ Assistant, an expert analyst. You help users understand a business document
that has already been processed through an AI pipeline.

IMPORTANT RULES:
1. Answer ONLY based on the context provided below. Do not use external knowledge.
2. If a question cannot be answered from this context, say so clearly.
3. Be concise, precise, and professional.

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


#  Prompt builder 

def _build_prompt(system_ctx: str, history: list[dict], user_msg: str) -> str:
    """Concatenate system context + conversation history + new user message."""
    turns = ""
    for turn in history:
        turns += f"\nUser: {turn['user']}\nAssistant: {turn['assistant']}\n"
    return f"{system_ctx}\n{turns}\nUser: {user_msg}\nAssistant:"


# Display helpers

def _print_welcome(result: dict[str, Any]) -> None:
    doc_name  = result.get("document", "document")
    doc_type  = result.get("document_type", "unknown").upper()
    n_issues  = result.get("validation", {}).get("issue_count", 0)
    n_recs    = len(result.get("recommendations", []))

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]NexusIQ Chat[/bold cyan] — Ask anything about [green]{doc_name}[/green]\n"
            f"[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to end  •  "
            f"[bold]summary[/bold] to re-print results  •  "
            f"[bold]clear[/bold] to reset history[/dim]",
            border_style="cyan",
        )
    )

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("K", style="bold cyan")
    t.add_column("V", style="white")
    t.add_row("Document Type",    doc_type)
    t.add_row("Confidence",       f"{result.get('confidence_score', 0):.0%}")
    t.add_row("Issues Found",     str(n_issues))
    t.add_row("Recommendations",  str(n_recs))
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


#  Main entry point 

def start_chat_session(result: dict[str, Any]) -> None:
    """
    Launch an interactive RAG chatbot REPL grounded to the processed document.

    Args:
        result: The full dict returned by run_pipeline() — must include 'raw_content'.
    """
    system_ctx: str = _build_system_context(result)
    history: list[dict] = []

    _print_welcome(result)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        #  Built-in commands 
        if cmd in ("exit", "quit"):
            console.print("[dim]Goodbye! Chat session ended.[/dim]")
            break

        if cmd == "summary":
            _print_summary(result)
            continue

        if cmd == "clear":
            history.clear()
            console.print("[dim]Conversation history cleared.[/dim]\n")
            continue

        if cmd in ("help", "?"):
            console.print(
                "[dim]Commands: [bold]exit[/bold] | [bold]quit[/bold] | "
                "[bold]summary[/bold] | [bold]clear[/bold] | [bold]help[/bold][/dim]\n"
            )
            continue

        # LLM call 
        prompt = _build_prompt(system_ctx, history, user_input)

        try:
            with console.status("[dim]Thinking…[/dim]", spinner="dots"):
                response = call_llm(prompt)
        except Exception as exc:
            console.print(f"[red]LLM error:[/red] {exc}\n")
            logger.error(f"Chatbot LLM error: {exc}")
            continue

        # Store in history
        history.append({"user": user_input, "assistant": response})

        # Render response as markdown for nice formatting
        console.print()
        console.print(
            Panel(
                Markdown(response),
                title="[bold green]NexusIQ Assistant[/bold green]",
                border_style="green",
            )
        )
        console.print()
