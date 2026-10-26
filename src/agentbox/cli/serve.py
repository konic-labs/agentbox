"""`agentbox serve`: run the HTTP environment server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentbox.cli._common import fail, state


def serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p")] = 8765,
    tasks: Annotated[Optional[Path], typer.Option("--tasks", help="Task source exposed by id")] = None,
    token_env: Annotated[
        str, typer.Option("--token-env", help="Env var holding the bearer token")
    ] = "AGENTBOX_SERVER_TOKEN",
    max_sessions: Annotated[int, typer.Option("--max-sessions")] = 64,
    idle_ttl: Annotated[float, typer.Option("--idle-ttl", help="Seconds before idle sessions are reaped")] = 1800.0,
) -> None:
    try:
        import uvicorn
    except ImportError:
        fail("the server needs: pip install 'agentbox-rl[server]'")
    from agentbox.server.app import create_app, settings_from_project

    st = state(ctx)
    token = os.environ.get(token_env)
    if host not in ("127.0.0.1", "localhost") and not token:
        typer.secho(
            f"warning: serving on {host} without ${token_env}; anyone who can reach it can run code",
            fg=typer.colors.YELLOW,
            err=True,
        )
    app = create_app(settings_from_project(
        st.project, tasks=tasks, token=token, max_sessions=max_sessions, idle_ttl_s=idle_ttl
    ))
    uvicorn.run(app, host=host, port=port, log_level="warning" if st.quiet else "info")
