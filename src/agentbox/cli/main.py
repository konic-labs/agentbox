"""CLI entrypoint."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from agentbox.version import __version__

app = typer.Typer(
    name="agentbox",
    help="Agentic virtual env for trajectory building.",
    no_args_is_help=True,
)

from agentbox.cli.bench import bench_app  # noqa: E402

app.add_typer(bench_app, name="bench")


@app.command()
def version() -> None:
    """Print AgentBox version."""
    typer.echo(__version__)


@app.command()
def doctor(
    prune: bool = typer.Option(False, "--prune", help="Remove orphan agentbox containers"),
) -> None:
    """Check Docker, Python, and sandbox image readiness."""
    import sys

    typer.echo(f"agentbox {__version__}")
    typer.echo(f"python {sys.version.split()[0]}")

    try:
        from agentbox.sandbox import docker_backend

        client = docker_backend.get_client()
        client.ping()
        typer.secho("docker: ok", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(f"docker: FAIL ({exc})", fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    image = "python:3.12-slim-bookworm"
    try:
        client.images.get(image)
        typer.secho(f"image {image}: present", fg=typer.colors.GREEN)
    except Exception:
        typer.secho(f"image {image}: missing (will auto-pull on first run)", fg=typer.colors.YELLOW)

    labeled = client.containers.list(all=True, filters={"label": "agentbox=1"})
    typer.echo(f"agentbox containers: {len(labeled)}")
    if prune and labeled:
        from agentbox.sandbox.prune import prune_agentbox_containers

        removed = prune_agentbox_containers(client)
        typer.echo(f"pruned: {len(removed)}")


@app.command("build-image")
def build_image(
    tag: str = typer.Option("agentbox/sandbox:latest", "--tag", "-t"),
    path: Path = typer.Option(
        Path("docker/python-sandbox"),
        "--path",
        help="Dockerfile directory",
    ),
) -> None:
    """Build the optional baked AgentBox sandbox image."""
    from agentbox.sandbox import docker_backend

    client = docker_backend.get_client()
    dockerfile_dir = path.resolve()
    if not (dockerfile_dir / "Dockerfile").exists():
        typer.secho(f"Dockerfile not found in {dockerfile_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Building {tag} from {dockerfile_dir} ...")
    image, logs = client.images.build(path=str(dockerfile_dir), tag=tag, rm=True)
    for line in logs:
        if isinstance(line, dict) and "stream" in line:
            typer.echo(line["stream"], nl=False)
    typer.secho(f"\nBuilt {tag} ({image.short_id})", fg=typer.colors.GREEN)


@app.command()
def run(
    task: Path = typer.Argument(..., help="Path to task.json or task directory"),
    model: str = typer.Option("mock", "--model", "-m", help="Model name or 'mock'"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    max_steps: int = typer.Option(40, "--max-steps"),
    tools: str = typer.Option("structured", "--tools", help="structured|shell|custom"),
    out: Path = typer.Option(Path("trajectories"), "--out", "-o"),
    image: Optional[str] = typer.Option(None, "--image"),
    network: bool = typer.Option(False, "--network", help="Enable container network"),
) -> None:
    """Run a single rollout."""
    from agentbox.config import AgentConfig, ModelConfig, ResourceLimits, SandboxConfig
    from agentbox.model.mock import MockModelClient
    from agentbox.model.base import ModelResponse
    from agentbox.runner.rollout import Rollout
    from agentbox.tasks.loader import load_task
    from agentbox.types import ToolMode

    task_obj = load_task(task)
    sandbox = SandboxConfig(
        image=image or SandboxConfig().image,
        limits=ResourceLimits(network_disabled=not network),
        ensure_pytest=True,
    )
    agent_cfg = AgentConfig(
        tools=ToolMode(tools) if tools in {m.value for m in ToolMode} else tools,
        max_steps=max_steps,
    )

    async def _run():
        if model == "mock":
            model_client = MockModelClient(
                [ModelResponse(content="(mock no-op done)", tool_calls=[])]
            )
        else:
            model_client = ModelConfig(
                model=model, base_url=base_url, api_key=api_key
            )
        return await Rollout.run(
            task_obj, model=model_client, agent=agent_cfg, sandbox=sandbox
        )

    traj = asyncio.run(_run())
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{task_obj.task_id}_{traj.run_id[:8]}.json"
    traj.save(path)
    typer.echo(f"status={traj.final_status.value} reward={traj.reward} steps={traj.metrics.steps}")
    typer.echo(f"saved {path}")
    if traj.final_status.value == "success":
        raise typer.Exit(0)
    if traj.final_status.value == "error":
        raise typer.Exit(2)
    raise typer.Exit(1)


@app.command("run-dir")
def run_dir(
    tasks_dir: Path = typer.Argument(..., help="Directory of task folders or JSON files"),
    model: str = typer.Option(..., "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    concurrency: int = typer.Option(8, "--concurrency", "-c"),
    n: int = typer.Option(1, "--n", help="Rollouts per task"),
    max_steps: int = typer.Option(40, "--max-steps"),
    out: Path = typer.Option(Path("trajectories"), "--out", "-o"),
    network: bool = typer.Option(False, "--network"),
) -> None:
    """Run all tasks in a directory in parallel."""
    from agentbox.config import AgentConfig, ModelConfig, ResourceLimits, RolloutConfig, SandboxConfig
    from agentbox.runner.parallel import ParallelRunner
    from agentbox.tasks.loader import load_task
    from agentbox.trajectory.formats.jsonl import export_jsonl

    paths: list[Path] = []
    for p in sorted(tasks_dir.iterdir()):
        if p.is_dir() and (p / "task.json").exists():
            paths.append(p)
        elif p.suffix == ".json":
            paths.append(p)
    if not paths:
        typer.secho(f"No tasks found in {tasks_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)

    tasks = [load_task(p) for p in paths]
    config = RolloutConfig(
        model=ModelConfig(model=model, base_url=base_url, api_key=api_key),
        agent=AgentConfig(max_steps=max_steps),
        sandbox=SandboxConfig(limits=ResourceLimits(network_disabled=not network)),
    )
    runner = ParallelRunner(concurrency=concurrency, config=config)

    async def _run():
        return await runner.run_tasks(tasks, n_per_task=n, progress=True)

    trajs = asyncio.run(_run())
    out.mkdir(parents=True, exist_ok=True)
    export_jsonl(trajs, out / "dataset.jsonl")
    for traj in trajs:
        traj.save(out / f"{traj.task_id}_{traj.run_id[:8]}.json")

    ok = sum(1 for t in trajs if t.final_status.value == "success")
    typer.echo(f"done: {ok}/{len(trajs)} success → {out}")
    raise typer.Exit(0 if ok == len(trajs) else 1)


@app.command()
def export(
    traj: Path = typer.Argument(..., help="Trajectory JSON file"),
    format: str = typer.Option("art", "--format", "-f", help="art|jsonl"),
    out: Path = typer.Option(..., "--out", "-o"),
) -> None:
    """Export a trajectory to another format."""
    from agentbox.trajectory.schema import Trajectory

    t = Trajectory.load(traj)
    if format == "art":
        out.write_text(json.dumps(t.to_art_dict(), indent=2), encoding="utf-8")
    elif format == "jsonl":
        out.write_text(t.model_dump_json() + "\n", encoding="utf-8")
    else:
        typer.secho(f"Unknown format: {format}", fg=typer.colors.RED)
        raise typer.Exit(2)
    typer.echo(f"wrote {out}")


@app.command()
def prune() -> None:
    """Remove all containers labeled agentbox=1."""
    from agentbox.sandbox.prune import prune_agentbox_containers

    removed = prune_agentbox_containers()
    typer.echo(f"removed {len(removed)} containers")


if __name__ == "__main__":
    app()
