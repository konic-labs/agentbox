"""CLI entrypoint (minimal stub for packaging)."""

from __future__ import annotations

import typer

from agentbox.version import __version__

app = typer.Typer(name="agentbox", help="Agentic virtual env for trajectory building.")


@app.command()
def version() -> None:
    """Print AgentBox version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
