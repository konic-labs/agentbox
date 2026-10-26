"""`agentbox sandbox …`: interactive sandboxes for debugging tasks (GH #89)."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentbox.cli._common import EXIT_FAILURES, emit, fail, state, table

sandbox_app = typer.Typer(help="Create, inspect and remove sandboxes.", no_args_is_help=True)


def _parse_age(value: str | None) -> float | None:
    if not value:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        if value[-1] in units:
            return float(value[:-1]) * units[value[-1]]
        return float(value)
    except ValueError:
        fail(f"invalid duration {value!r} (e.g. 30m, 2h, 1d)")


@sandbox_app.command("up")
def sandbox_up(
    ctx: typer.Context,
    task: Annotated[Path, typer.Argument(help="Task file or directory to seed")],
    network: Annotated[Optional[bool], typer.Option("--network/--no-network")] = None,
    image: Annotated[Optional[str], typer.Option("--image")] = None,
) -> None:
    """Seed a task into a sandbox and leave it running (prints how to attach)."""
    from agentbox.config import SandboxConfig
    from agentbox.runner.provision import manager_for, provision
    from agentbox.tasks.loader import load_task

    st = state(ctx)
    t = load_task(task)
    cfg = st.project.sandbox or SandboxConfig()
    upd: dict[str, object] = {}
    if image:
        upd["image"] = image
    if network is not None:
        upd["limits"] = cfg.limits.model_copy(update={"network_disabled": not network})
    cfg = cfg.model_copy(update=upd) if upd else cfg

    async def go() -> dict[str, str]:
        mgr = manager_for(t, cfg, None)
        prov = await provision(t, manager=mgr, run_id="interactive", rollout_id=uuid.uuid4().hex)
        mgr.keep(prov.sandbox)
        return {"name": prov.sandbox.name, "id": prov.sandbox.id, "image": prov.sandbox.image}

    info = asyncio.run(go())
    emit(st, info, lambda: (
        typer.secho(f"sandbox {info['name']} is up ({info['image']})", fg=typer.colors.GREEN),
        typer.echo(f"  agentbox sandbox shell {info['name']}"),
        typer.echo(f"  agentbox sandbox exec {info['name']} -- python -m pytest -q"),
        typer.echo(f"  agentbox sandbox rm {info['name']}"),
    ))


@sandbox_app.command("exec", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sandbox_exec(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Container name or id")],
    timeout: Annotated[float, typer.Option("--timeout")] = 300.0,
) -> None:
    """Run a command in a sandbox: agentbox sandbox exec NAME -- CMD ARGS..."""
    from agentbox.sandbox.backend import get_backend

    cmd = list(ctx.args)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        fail("no command given (use: agentbox sandbox exec NAME -- CMD)")
    backend = get_backend()
    res = backend.exec_sync(name, ["/bin/sh", "-c", " ".join(cmd)], timeout_s=timeout,
                            workdir="/workspace")
    typer.echo(res.stdout, nl=False)
    if res.stderr:
        typer.echo(res.stderr, err=True, nl=False)
    raise typer.Exit(res.exit_code if res.exit_code >= 0 else EXIT_FAILURES)


@sandbox_app.command("shell")
def sandbox_shell(name: Annotated[str, typer.Argument(help="Container name or id")]) -> None:
    """Open an interactive shell (docker exec -it) in a sandbox."""
    docker = shutil.which("docker")
    if docker is None:
        fail("the docker CLI is required for an interactive shell")
    os.execvp(docker, [docker, "exec", "-it", "-w", "/workspace", name, "sh", "-c",
                       "command -v bash >/dev/null && exec bash || exec sh"])


@sandbox_app.command("ls")
def sandbox_ls(
    ctx: typer.Context,
    all_: Annotated[bool, typer.Option("--all", "-a", help="Include builders/probes")] = False,
) -> None:
    """List AgentBox containers with owner, age and orphan status."""
    from agentbox.sandbox.backend import get_backend
    from agentbox.sandbox.prune import is_orphan

    st = state(ctx)
    rows = []
    now = time.time()
    for c in get_backend().list_containers_sync():
        labels = c.get("Labels") or {}
        role = labels.get("agentbox.role", "sandbox")
        if role != "sandbox" and not all_:
            continue
        created = float(labels.get("agentbox.created") or c.get("Created") or now)
        rows.append({
            "name": (c.get("Names") or ["?"])[0].lstrip("/"),
            "id": c["Id"][:12],
            "task": labels.get("agentbox.task_id", ""),
            "state": c.get("State"),
            "age_s": int(now - created),
            "owner": labels.get("agentbox.owner", "?"),
            "orphan": is_orphan(labels),
            "role": role,
        })
    emit(st, rows, lambda: table(
        ["name", "task", "state", "age", "owner", "orphan"],
        [[r["name"], r["task"], r["state"], f"{r['age_s'] // 60}m", r["owner"], r["orphan"]] for r in rows],
    ))


@sandbox_app.command("rm")
def sandbox_rm(
    ctx: typer.Context,
    names: Annotated[list[str], typer.Argument(help="Container names or ids")],
) -> None:
    """Remove sandboxes."""
    from agentbox.sandbox.backend import get_backend

    backend = get_backend()
    for n in names:
        backend.remove_container_sync(n)
    emit(state(ctx), {"removed": names}, lambda: typer.echo(f"removed {len(names)}"))


@sandbox_app.command("prune")
def sandbox_prune(
    ctx: typer.Context,
    all_: Annotated[bool, typer.Option("--all", help="Remove every AgentBox container")] = False,
    run_id: Annotated[Optional[str], typer.Option("--run-id")] = None,
    older_than: Annotated[Optional[str], typer.Option("--older-than", help="e.g. 2h")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Remove orphaned sandboxes (owner process gone), or all / by run / by age."""
    from agentbox.sandbox.prune import prune_agentbox_containers

    removed = prune_agentbox_containers(
        scope="all" if all_ else "orphans",
        run_id=run_id,
        older_than_s=_parse_age(older_than),
        dry_run=dry_run,
    )
    verb = "would remove" if dry_run else "removed"
    emit(state(ctx), {"removed": removed, "dry_run": dry_run},
         lambda: typer.echo(f"{verb} {len(removed)} container(s)"))
