"""`agentbox traj …`: inspect, validate, curate, replay and export trajectories (GH #69)."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from agentbox.cli._common import EXIT_FAILURES, EXIT_OK, emit, fail, fmt_pct, state, table

traj_app = typer.Typer(help="Inspect, curate, replay and export trajectories.", no_args_is_help=True)


def _role(m: Any) -> str:
    return m.role.value if hasattr(m.role, "value") else str(m.role)


def conversation_errors(traj: Any) -> list[str]:
    """Structural problems that break chat templates / SFT / replay."""
    errors: list[str] = []
    msgs = traj.messages
    if msgs and _role(msgs[0]) not in ("system", "user"):
        errors.append("first message must be system or user")
    pending: dict[str, int] = {}
    for i, m in enumerate(msgs):
        role = _role(m)
        if role == "assistant":
            if pending:
                errors.append(f"message {i}: assistant turn before tool results for {sorted(pending)}")
                pending.clear()
            for tc in m.tool_calls or []:
                if tc.id in pending:
                    errors.append(f"message {i}: duplicate tool_call id {tc.id}")
                pending[tc.id] = i
        elif role == "tool":
            if m.tool_call_id not in pending:
                errors.append(f"message {i}: tool result for unknown call {m.tool_call_id}")
            pending.pop(m.tool_call_id or "", None)
    if pending:
        errors.append(f"dangling tool calls without results: {sorted(pending)}")
    return errors


@traj_app.command("show")
def traj_show(ctx: typer.Context, path: Annotated[Path, typer.Argument()]) -> None:
    """Summarise a trajectory (status, reward, verifier, last tool calls)."""
    from agentbox.trajectory.io import iter_trajectories

    st = state(ctx)
    for t in iter_trajectories(path):
        data = {
            "task_id": t.task_id,
            "rollout_id": t.rollout_id,
            "status": t.final_status.value,
            "failure_kind": str(getattr(t.failure_kind, "value", t.failure_kind)),
            "stop_reason": t.stop_reason,
            "reward": t.reward,
            "reward_breakdown": t.reward_breakdown,
            "steps": t.metrics.steps,
            "tool_calls": t.metrics.tool_calls,
            "duration_s": round(t.metrics.duration_s, 2),
            "verify": t.verify.model_dump(exclude={"stdout", "stderr"}) if t.verify else None,
            "error": t.error,
        }

        def human(t: Any = t, data: dict[str, Any] = data) -> None:
            typer.echo(f"{t.task_id} [{t.rollout_id[:8]}] {data['status']} reward={t.reward:.3f} "
                       f"steps={t.metrics.steps} tools={t.metrics.tool_calls} ({data['failure_kind']})")
            if t.verify:
                typer.echo(f"verify: {t.verify.command!r} exit={t.verify.exit_code} "
                           f"success={t.verify.success} tests={t.verify.tests_passed}/{t.verify.tests_total}")
                if t.verify.stdout:
                    typer.echo("--- verify stdout (tail) ---\n" + t.verify.stdout[-1200:])
            if t.error:
                typer.secho(f"error: {t.error}", fg=typer.colors.RED)
            for r in t.tool_call_records[-6:]:
                typer.echo(f"  step {r.step} {r.name}{' ERR' if r.is_error else ''} ({r.duration_s:.2f}s)")

        emit(st, data, human)
        break
    else:
        fail(f"no trajectory in {path}")


@traj_app.command("stats")
def traj_stats(
    ctx: typer.Context,
    sources: Annotated[list[Path], typer.Argument(help="Trajectory files/dirs/JSONL/bench runs")],
) -> None:
    """Aggregate metrics, tool usage and GRPO group statistics (streamed)."""
    from agentbox.metrics.aggregate import MetricsAccumulator
    from agentbox.rl.stats import group_stats
    from agentbox.trajectory.io import iter_trajectories

    acc = MetricsAccumulator()
    tools: Counter[str] = Counter()
    groups: dict[str, list[float]] = {}
    for t in iter_trajectories(sources):
        acc.add(t)
        tools.update(r.name for r in t.tool_call_records)
        if t.is_trainable:
            groups.setdefault(t.group_id or t.task_id, []).append(t.reward)
    m = acc.result()
    data = {**m.model_dump(exclude={"by_task"}), "tools": dict(tools.most_common()),
            "groups": group_stats(list(groups.values()))}
    st = state(ctx)

    def human() -> None:
        table(["n", "success", "mean reward", "reward std", "mean steps", "tool err rate", "infra"],
              [[m.n, fmt_pct(m.success_rate), f"{m.mean_reward:.3f}", f"{m.reward_std:.3f}",
                f"{m.mean_steps:.1f}", fmt_pct(m.tool_error_rate), m.infra_failures]])
        typer.echo(f"statuses: {m.by_status}")
        typer.echo(f"failure kinds: {m.by_failure_kind}")
        typer.echo(f"tools: {dict(tools.most_common(10))}")
        typer.echo(f"groups: {data['groups']}")

    emit(st, data, human)


@traj_app.command("filter")
def traj_filter(
    ctx: typer.Context,
    sources: Annotated[list[Path], typer.Argument()],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output .jsonl[.gz]")],
    status: Annotated[Optional[list[str]], typer.Option("--status")] = None,
    min_reward: Annotated[Optional[float], typer.Option("--min-reward")] = None,
    max_reward: Annotated[Optional[float], typer.Option("--max-reward")] = None,
    max_steps: Annotated[Optional[int], typer.Option("--max-steps")] = None,
    tool: Annotated[Optional[list[str]], typer.Option("--tool", help="Must use this tool")] = None,
    failure_kind: Annotated[Optional[list[str]], typer.Option("--failure-kind")] = None,
    grep: Annotated[Optional[str], typer.Option("--grep", help="Regex over message contents")] = None,
    trainable_only: Annotated[bool, typer.Option("--trainable-only")] = False,
) -> None:
    """Write the trajectories matching all filters to a JSONL file."""
    from agentbox.trajectory.io import iter_trajectories, write_jsonl

    rx = re.compile(grep) if grep else None
    kept = read = 0

    def keep() -> Any:
        nonlocal kept, read
        for t in iter_trajectories(sources):
            read += 1
            if status and t.final_status.value not in status:
                continue
            if min_reward is not None and t.reward < min_reward:
                continue
            if max_reward is not None and t.reward > max_reward:
                continue
            if max_steps is not None and t.metrics.steps > max_steps:
                continue
            if tool and not set(tool) <= {r.name for r in t.tool_call_records}:
                continue
            if failure_kind and str(getattr(t.failure_kind, "value", t.failure_kind)) not in failure_kind:
                continue
            if trainable_only and not t.is_trainable:
                continue
            if rx and not any(rx.search(m.content or "") for m in t.messages):
                continue
            kept += 1
            yield t

    write_jsonl(keep(), out)
    emit(state(ctx), {"read": read, "kept": kept, "out": str(out)},
         lambda: typer.echo(f"kept {kept}/{read} → {out}"))


@traj_app.command("validate")
def traj_validate(ctx: typer.Context, sources: Annotated[list[Path], typer.Argument()]) -> None:
    """Check schema + conversation validity (dangling tool calls, unknown ids, order)."""
    from agentbox.trajectory.io import iter_trajectories

    problems = {}
    n = 0
    for t in iter_trajectories(sources):
        n += 1
        errs = conversation_errors(t)
        if errs:
            problems[t.rollout_id or t.run_id] = errs
    emit(state(ctx), {"n": n, "invalid": len(problems), "problems": problems}, lambda: (
        typer.secho(f"{n - len(problems)}/{n} valid", fg=typer.colors.GREEN if not problems else typer.colors.RED),
        [typer.echo(f"  {k}: {v[0]}") for k, v in list(problems.items())[:20]],
    ))
    raise typer.Exit(EXIT_FAILURES if problems else EXIT_OK)


@traj_app.command("replay")
def traj_replay(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Trajectory to replay")],
    task: Annotated[Path, typer.Option("--task", help="Task the trajectory was collected on")],
) -> None:
    """Re-execute the recorded tool calls in a fresh sandbox and compare the reward."""
    from agentbox.config import AgentConfig, SandboxConfig
    from agentbox.model.replay import ReplayModelClient
    from agentbox.runner.rollout import Rollout
    from agentbox.tasks.loader import load_task
    from agentbox.trajectory.io import iter_trajectories

    st = state(ctx)
    original = next(iter(iter_trajectories(path)), None)
    if original is None:
        fail(f"no trajectory in {path}")
    t = load_task(task)
    agent = (st.project.agent or AgentConfig()).model_copy(
        update={"max_steps": max(1, original.metrics.steps + 1)}
    )
    if original.tool_mode and original.tool_mode in ("structured", "extended", "shell", "custom"):
        agent = agent.model_copy(update={"tools": original.tool_mode})
    replayed = asyncio.run(Rollout.run(
        t, model=ReplayModelClient(original), agent=agent, sandbox=st.project.sandbox or SandboxConfig()
    ))
    same = replayed.final_status == original.final_status and abs(replayed.reward - original.reward) < 1e-9
    data = {"original": {"status": original.final_status.value, "reward": original.reward},
            "replayed": {"status": replayed.final_status.value, "reward": replayed.reward},
            "reproduced": same}
    emit(st, data, lambda: typer.secho(
        f"original {data['original']} · replayed {data['replayed']} · "
        f"{'REPRODUCED' if same else 'DIFFERENT'}",
        fg=typer.colors.GREEN if same else typer.colors.YELLOW,
    ))
    raise typer.Exit(EXIT_OK if same else EXIT_FAILURES)


@traj_app.command("render")
def traj_render(
    ctx: typer.Context,
    source: Annotated[Path, typer.Argument(help="Trajectory file, dir, JSONL or bench run")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("traj-dash.html"),
    model_id: Annotated[Optional[str], typer.Option("--model-id")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max trajectories rendered")] = 500,
) -> None:
    """Render a self-contained HTML dashboard."""
    import itertools

    from agentbox.trajectory.io import iter_trajectories
    from agentbox.trajectory.render import render_html

    trajs = list(itertools.islice(iter_trajectories(source, model_id=model_id), limit))
    if not trajs:
        fail(f"no trajectories under {source}")
    report = None
    for cand in (source / "report.json", source.parent / "report.json"):
        if cand.is_file():
            try:
                report = json.loads(cand.read_text())
            except ValueError:
                report = None
            break
    path = render_html(trajs, out, title=f"AgentBox Trajectories ({len(trajs)})", report=report)
    emit(state(ctx), {"out": str(path), "n": len(trajs)},
         lambda: typer.echo(f"wrote {path} ({path.stat().st_size // 1024} KB, {len(trajs)} trajectories)"))


@traj_app.command("export")
def traj_export(
    ctx: typer.Context,
    source: Annotated[Path, typer.Argument(help="Trajectories (file/dir/JSONL/bench run)")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="chat|dpo|grpo|tokens|hf|sharegpt|art")] = "chat",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("export.jsonl"),
    successful_only: Annotated[bool, typer.Option("--successful-only")] = False,
    min_reward: Annotated[Optional[float], typer.Option("--min-reward")] = None,
    include_infra: Annotated[bool, typer.Option("--include-infra", help="Keep infra failures")] = False,
    per_turn: Annotated[bool, typer.Option("--per-turn", help="tokens: one row per turn")] = False,
    drop_zero_variance: Annotated[bool, typer.Option("--drop-zero-variance", help="grpo")] = False,
    min_margin: Annotated[float, typer.Option("--min-margin", help="dpo")] = 0.5,
    push_to_hub: Annotated[Optional[str], typer.Option("--push-to-hub", help="hf: repo id")] = None,
) -> None:
    """Export a training dataset in a trainer-friendly format."""
    from agentbox.export import ExportFilter, available_formats, export
    from agentbox.trajectory.io import iter_trajectories

    if fmt not in available_formats():
        fail(f"unknown format {fmt!r}; available: {available_formats()}")
    stats = export(
        iter_trajectories(source),
        fmt,
        out,
        filter=ExportFilter(
            trainable_only=not include_infra, successful_only=successful_only, min_reward=min_reward
        ),
        per_turn=per_turn,
        drop_zero_variance=drop_zero_variance,
        min_margin=min_margin,
        push_to_hub=push_to_hub,
    )
    emit(state(ctx), stats.model_dump(), lambda: (
        typer.echo(f"{stats.format}: {stats.rows} rows from {stats.kept}/{stats.read} trajectories → {out}"),
        [typer.echo(f"note: {n}") for n in stats.notes],
    ))


@traj_app.command("diff")
def traj_diff(
    ctx: typer.Context,
    a: Annotated[Path, typer.Argument()],
    b: Annotated[Path, typer.Argument()],
) -> None:
    """Compare two trajectories: outcome, tool sequence and first divergence."""
    from agentbox.trajectory.io import iter_trajectories

    ta = next(iter(iter_trajectories(a)), None)
    tb = next(iter(iter_trajectories(b)), None)
    if ta is None or tb is None:
        fail("both arguments must contain a trajectory")
    sa = [(r.name, json.dumps(r.arguments, sort_keys=True)) for r in ta.tool_call_records]
    sb = [(r.name, json.dumps(r.arguments, sort_keys=True)) for r in tb.tool_call_records]
    diverge = next((i for i, (x, y) in enumerate(zip(sa, sb, strict=False)) if x != y), None)
    if diverge is None and len(sa) != len(sb):
        diverge = min(len(sa), len(sb))
    data = {
        "a": {"status": ta.final_status.value, "reward": ta.reward, "steps": ta.metrics.steps},
        "b": {"status": tb.final_status.value, "reward": tb.reward, "steps": tb.metrics.steps},
        "same_task": ta.task_id == tb.task_id,
        "first_divergent_tool_call": diverge,
        "a_tools": [n for n, _ in sa],
        "b_tools": [n for n, _ in sb],
    }
    emit(state(ctx), data)
