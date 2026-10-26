"""`agentbox task …`: author, validate, lint and check tasks (GH #56)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from agentbox.cli._common import EXIT_FAILURES, EXIT_OK, emit, fail, state, table

if TYPE_CHECKING:
    from agentbox.tasks.schema import Task

task_app = typer.Typer(help="Author and validate tasks.", no_args_is_help=True)


@task_app.command("new")
def task_new(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Argument(help="Directory for the new task")],
    task_id: Annotated[str | None, typer.Option("--id", help="Task id (default: dir name)")] = None,
) -> None:
    """Scaffold task.json + files/ (starter code, tests) with tamper protection on."""
    from agentbox.tasks.schema import Task, VerifierSpec
    from agentbox.types import VerifierType

    if (directory / "task.json").exists():
        fail(f"{directory}/task.json already exists")
    tid = task_id or directory.name
    task = Task(
        task_id=tid,
        description=f"Describe what the agent must do for {tid}.",
        starter_files={
            "solution.py": "def solve(x):\n    raise NotImplementedError\n",
            "test_solution.py": (
                "from solution import solve\n\n\ndef test_solve():\n    assert solve(1) == 1\n"
            ),
        },
        verifier=VerifierSpec(
            type=VerifierType.PYTEST, command="python -m pytest -q", protected_paths=["test_*.py"]
        ),
        metadata={"difficulty": "easy", "tags": []},
    )
    task.save_dir(directory)
    emit(
        state(ctx),
        {"path": str(directory)},
        lambda: typer.echo(f"created {directory}/task.json + files/"),
    )


@task_app.command("validate")
def task_validate(
    ctx: typer.Context,
    sources: Annotated[list[Path], typer.Argument(help="Task files/dirs/datasets")],
) -> None:
    """Validate task schemas (strict: any invalid task fails)."""
    from agentbox.errors import TaskValidationError
    from agentbox.tasks.loader import iter_tasks

    st = state(ctx)
    try:
        tasks = list(iter_tasks(sources, strict=True))
    except TaskValidationError as exc:
        msg = str(exc)
        emit(st, {"ok": False, "error": msg}, lambda: typer.secho(msg, fg=typer.colors.RED))
        raise typer.Exit(EXIT_FAILURES) from exc
    emit(
        st,
        {"ok": True, "n": len(tasks)},
        lambda: typer.secho(f"ok: {len(tasks)} valid task(s)", fg=typer.colors.GREEN),
    )


@task_app.command("lint")
def task_lint(
    ctx: typer.Context,
    sources: Annotated[list[Path], typer.Argument()],
    min_asserts: Annotated[int, typer.Option("--min-asserts")] = 3,
    strict_paths: Annotated[bool, typer.Option("--strict-paths")] = False,
) -> None:
    """Static QC: stub-only starters, enough asserts, no spoilers, consistent paths."""
    from agentbox.tasks.generate.static_qc import validate_task_static
    from agentbox.tasks.loader import iter_tasks

    st = state(ctx)
    rows: list[dict[str, Any]] = []
    for t in iter_tasks(sources):
        rep = validate_task_static(t, min_asserts=min_asserts, strict_paths=strict_paths)
        rows.append(
            {"task_id": t.task_id, "ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings}
        )
    bad = [r for r in rows if not r["ok"]]
    emit(
        st,
        rows,
        lambda: table(
            ["task", "ok", "errors", "warnings"],
            [
                [r["task_id"], r["ok"], "; ".join(r["errors"])[:120], len(r["warnings"])]
                for r in rows
            ],
        ),
    )
    raise typer.Exit(EXIT_FAILURES if bad else EXIT_OK)


@task_app.command("check")
def task_check(
    ctx: typer.Context,
    sources: Annotated[list[Path], typer.Argument()],
    solution: Annotated[
        Path | None, typer.Option("--solution", help="Dir with golden files that must pass")
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 4,
) -> None:
    """Docker QC: the starter must FAIL the verifier and (with --solution) the golden must PASS."""
    from agentbox.config import SandboxConfig
    from agentbox.tasks.generate.validate import validate_task_live
    from agentbox.tasks.loader import iter_tasks

    st = state(ctx)
    tasks = list(iter_tasks(sources))
    golden = None
    if solution is not None:
        golden = {
            p.relative_to(solution).as_posix(): p.read_text(encoding="utf-8")
            for p in solution.rglob("*")
            if p.is_file()
        }
    cfg = st.project.sandbox or SandboxConfig()

    async def go() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)

        async def one(t: Task) -> dict[str, Any]:
            async with sem:
                rep = await validate_task_live(t, sandbox_config=cfg, solution_files=golden)
                return {"task_id": t.task_id, "ok": rep.ok, "errors": rep.errors}

        return list(await asyncio.gather(*(one(t) for t in tasks)))

    rows = asyncio.run(go())
    emit(
        st,
        rows,
        lambda: table(
            ["task", "ok", "errors"],
            [[r["task_id"], r["ok"], "; ".join(r["errors"])[:140]] for r in rows],
        ),
    )
    raise typer.Exit(EXIT_OK if all(r["ok"] for r in rows) else EXIT_FAILURES)


@task_app.command("show")
def task_show(ctx: typer.Context, path: Annotated[Path, typer.Argument()]) -> None:
    """Show a task (files listed, verifier, metadata)."""
    from agentbox.tasks.loader import load_task

    t = load_task(path)
    data = t.model_dump(mode="json", exclude={"starter_files", "binary_files"})
    data["files"] = sorted(t.starter_files) + sorted(t.binary_files)
    emit(state(ctx), data)


@task_app.command("ls")
def task_ls(ctx: typer.Context, sources: Annotated[list[Path], typer.Argument()]) -> None:
    """List tasks in files/dirs/datasets."""
    from agentbox.tasks.loader import iter_tasks

    rows = [
        {
            "task_id": t.task_id,
            "files": len(t.starter_files) + len(t.binary_files),
            "setup": len(t.setup_commands),
            "verifier": t.verifier.type.value,
            "difficulty": t.metadata.get("difficulty"),
        }
        for t in iter_tasks(sources)
    ]
    emit(
        state(ctx),
        rows,
        lambda: table(
            ["task", "files", "setup", "verifier", "difficulty"], [list(r.values()) for r in rows]
        ),
    )


@task_app.command("schema")
def task_schema() -> None:
    """Print the task JSON Schema (for editors / external validators)."""
    from agentbox.tasks.schema import Task

    typer.echo(json.dumps(Task.model_json_schema(), indent=2))
