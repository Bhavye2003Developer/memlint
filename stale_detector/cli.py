import sys

import click
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from stale_detector.core import StaleDetector
from stale_detector.adapters.json_adapter import load_from_json
from stale_detector.adapters.mem0_adapter import load_from_mem0
from stale_detector.models import StalenessLevel

console = Console()

_LEVEL_STYLES = {
    StalenessLevel.FRESH:   "green",
    StalenessLevel.AGING:   "yellow",
    StalenessLevel.STALE:   "red",
    StalenessLevel.EXPIRED: "bold red",
}


@click.group()
def main():
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--only-flagged", is_flag=True, help="Show only STALE and EXPIRED facts.")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON to stdout.")
@click.option("--format", "fmt", default="default",
              type=click.Choice(["default", "mem0"]), help="Input format.")
def check(file: str, only_flagged: bool, output_json: bool, fmt: str):
    """Check memory facts for staleness."""
    facts = load_from_mem0(file) if fmt == "mem0" else load_from_json(file)

    if not facts:
        click.echo("No facts found in file.")
        sys.exit(0)

    report = StaleDetector().check(facts)

    if output_json:
        click.echo(report.model_dump_json(indent=2))
        return

    rows = report.flagged if only_flagged else report.results

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("ID",           style="dim",    max_width=12)
    table.add_column("Content",                      max_width=50)
    table.add_column("Category",                     max_width=12)
    table.add_column("Age",          justify="right", max_width=6)
    table.add_column("Score",        justify="right", max_width=6)
    table.add_column("Level",                        max_width=8)
    table.add_column("Action",                       max_width=8)

    for r in rows:
        content = r.content[:47] + "..." if len(r.content) > 50 else r.content
        table.add_row(
            r.fact_id,
            content,
            r.category.value,
            str(r.age_days),
            f"{r.staleness_score:.2f}",
            Text(r.staleness_level.value.upper(), style=_LEVEL_STYLES[r.staleness_level]),
            r.recommendation,
        )

    console.print(table)
    console.print(
        f"\n[dim]Checked {report.total_facts} facts: "
        f"[green]{report.fresh_count} fresh[/], "
        f"[yellow]{report.aging_count} aging[/], "
        f"[red]{report.stale_count} stale[/], "
        f"[bold red]{report.expired_count} expired[/][/dim]"
    )
