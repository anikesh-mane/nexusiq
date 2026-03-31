"""
CLI entry point — NexusIQ Document Intelligence System.
"""
from pathlib import Path
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

from src.utils.logger import setup_logger
from src.pipeline.orchestrator import run_pipeline
from src.evaluation.db_logger import log_pipeline_run

app = typer.Typer(
    name="nexusiq",
    help="Smart Document Intelligence & Action Recommendation System",
    add_completion=False,
)
console = Console()


@app.command()
def process(
    file_path: Path = typer.Argument(..., help="Path to the document to process."),
    output: Path = typer.Option(None, "--output", "-o", help="Save JSON output to file."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level."),
    save_metrics: bool = typer.Option(True, "--metrics/--no-metrics", help="Log run to SQLite."),
    pretty: bool = typer.Option(True, "--pretty/--plain", help="Rich formatted output."),
):
    """
    Process a business document and output structured insights + recommendations.
    """
    setup_logger(log_level)

    if not file_path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(code=1)

    if pretty:
        console.print(
            Panel.fit(
                f"[bold cyan]NexusIQ[/bold cyan] — Processing: [green]{file_path.name}[/green]",
                border_style="cyan",
            )
        )

    try:
        result = run_pipeline(file_path)
    except Exception as exc:
        console.print(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # --- Save metrics ---
    if save_metrics:
        try:
            log_pipeline_run(result)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not log metrics: {exc}")

    # --- Display output ---
    json_str = json.dumps(result, indent=2, default=str)

    if pretty:
        _display_pretty(result, json_str)
    else:
        print(json_str)

    # --- Optionally write to file ---
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_str, encoding="utf-8")
        console.print(f"\n[dim]Output saved to:[/dim] {output}")


def _display_pretty(result: dict, json_str: str) -> None:
    """Render a rich formatted summary to the terminal."""
    console.print()

    # Header info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold cyan")
    info_table.add_column("Value", style="white")
    info_table.add_row("Document Type", result["document_type"].upper())
    info_table.add_row("Confidence",    f"{result['confidence_score']:.0%}")
    info_table.add_row("Issues Found",  str(result["validation"]["issue_count"]))
    info_table.add_row("Recommendations", str(len(result["recommendations"])))
    info_table.add_row("Processing Time", f"{result['processing_time_seconds']}s")
    console.print(Panel(info_table, title="[bold]Summary[/bold]", border_style="green"))

    # Validation issues
    issues = result["validation"]["issues"]
    if issues:
        console.print()
        t = Table(title="Validation Issues", show_header=True, header_style="bold yellow")
        t.add_column("Severity", width=10)
        t.add_column("Type", width=20)
        t.add_column("Message")
        for issue in issues:
            color = {"high": "red", "medium": "yellow", "low": "dim"}.get(issue["severity"], "white")
            t.add_row(
                f"[{color}]{issue['severity']}[/{color}]",
                issue["type"],
                issue["message"],
            )
        console.print(t)

    # Recommendations
    recs = result["recommendations"]
    if recs:
        console.print()
        t = Table(title="Recommendations", show_header=True, header_style="bold green")
        t.add_column("Priority", width=10)
        t.add_column("Action", width=30)
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

    # Full JSON
    console.print()
    console.print(
        Panel(
            Syntax(json_str, "json", theme="monokai", line_numbers=False),
            title="[bold]Full JSON Output[/bold]",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    app()
