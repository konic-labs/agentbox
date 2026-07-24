"""CLI: agentbox traj …"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

traj_app = typer.Typer(
    name="traj",
    help="Inspect and render trajectories.",
    no_args_is_help=True,
)


@traj_app.command("show")
def traj_show(
    path: Path = typer.Argument(..., help="Trajectory JSON file"),
) -> None:
    """Pretty-print one trajectory (status, reward, official verify)."""
    from agentbox.trajectory.schema import Trajectory

    t = Trajectory.load(path)
    meta = t.metadata or {}
    typer.echo(f"task_id={t.task_id} run_id={t.run_id}")
    typer.echo(f"status={t.final_status} reward={t.reward}")
    typer.echo(
        f"steps={t.metrics.steps} tool_calls={t.metrics.tool_calls} "
        f"duration_s={t.metrics.duration_s:.2f}"
    )
    typer.echo(f"verify_command={meta.get('verify_command')}")
    typer.echo(
        f"verify_exit_code={meta.get('verify_exit_code')} "
        f"verify_success={meta.get('verify_success')}"
    )
    if meta.get("verify_stdout"):
        typer.echo("--- verify_stdout (tail) ---")
        typer.echo(str(meta["verify_stdout"])[-1500:])
    if meta.get("verify_stderr"):
        typer.echo("--- verify_stderr (tail) ---")
        typer.echo(str(meta["verify_stderr"])[-800:])
    if t.error:
        typer.echo(f"error={t.error}")
    # last tools
    if t.tool_call_records:
        typer.echo("--- last tools ---")
        for r in t.tool_call_records[-5:]:
            typer.echo(f"  step={r.step} {r.name} err={r.is_error}")


@traj_app.command("render")
def traj_render(
    source: Path = typer.Argument(
        ..., help="Trajectory file, trajectories dir, or bench run dir"
    ),
    out: Path = typer.Option(Path("traj-dash.html"), "--out", "-o"),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", help="When source is a bench run, pick models/<id>/trajectories"
    ),
) -> None:
    """Render a self-contained HTML dashboard of trajectories."""
    from agentbox.trajectory.render import render_html, resolve_trajectory_paths
    from agentbox.trajectory.schema import Trajectory

    paths = resolve_trajectory_paths(source, model_id=model_id)
    if not paths:
        typer.secho(f"No trajectories found under {source}", fg=typer.colors.RED)
        raise typer.Exit(2)
    trajs = [Trajectory.load(p) for p in paths]
    report = None
    # attach report if present next to run dir
    for cand in (source / "report.json", source.parent / "report.json"):
        if cand.exists():
            try:
                report = json.loads(cand.read_text())
            except Exception:
                pass
            break
    path = render_html(trajs, out, title=f"AgentBox Trajectories ({len(trajs)})", report=report)
    typer.echo(f"wrote {path} ({path.stat().st_size // 1024} KB, {len(trajs)} trajs)")
