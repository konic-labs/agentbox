"""CLI: agentbox bench …"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

import typer

from agentbox.version import __version__

bench_app = typer.Typer(
    name="bench",
    help="Create, freeze, and run real-rollout benchmark suites.",
    no_args_is_help=True,
)


def _interp_env(value: str | None) -> str | None:
    if value is None:
        return None

    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), "")

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, value)


@bench_app.command("create")
def bench_create(
    dest: Path = typer.Argument(..., help="Output suite directory"),
    from_tasks: Path = typer.Option(..., "--from-tasks", help="Tasks directory"),
    suite_id: str = typer.Option(..., "--suite-id"),
    name: str = typer.Option(..., "--name"),
    version: str = typer.Option("1.0.0", "--version"),
    description: str = typer.Option("", "--description"),
    concurrency: int = typer.Option(8, "--concurrency"),
    n: int = typer.Option(1, "--n", help="Rollouts per task"),
    network: bool = typer.Option(False, "--network"),
) -> None:
    """Create a suite pack from a tasks directory and freeze it."""
    from agentbox.benchmark.loader import create_suite_from_tasks
    from agentbox.config import ResourceLimits, SandboxConfig

    suite = create_suite_from_tasks(
        from_tasks,
        dest,
        suite_id=suite_id,
        name=name,
        version=version,
        description=description,
        concurrency=concurrency,
        n_per_task=n,
        sandbox=SandboxConfig(
            limits=ResourceLimits(network_disabled=not network),
        ),
        freeze=True,
    )
    typer.echo(
        f"created suite {suite.manifest.suite_id} v{suite.manifest.version} "
        f"tasks={len(suite.tasks)} hash={suite.manifest.content_hash}"
    )
    typer.echo(f"path={dest.resolve()}")


@bench_app.command("freeze")
def bench_freeze(suite_dir: Path = typer.Argument(...)) -> None:
    """Recompute and write content_hash for a suite."""
    from agentbox.benchmark.loader import load_suite, save_suite

    suite = load_suite(suite_dir).freeze()
    save_suite(suite, suite_dir)
    typer.echo(f"frozen hash={suite.manifest.content_hash}")


@bench_app.command("validate")
def bench_validate(
    suite_dir: Path = typer.Argument(...),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    """Validate suite structure and optional integrity hash."""
    from agentbox.benchmark.loader import load_suite
    from agentbox.errors import SuiteIntegrityError

    suite = load_suite(suite_dir)
    typer.echo(f"suite_id={suite.manifest.suite_id} tasks={len(suite.tasks)}")
    try:
        ok = suite.verify_integrity(strict=strict)
    except SuiteIntegrityError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    if ok:
        typer.secho("integrity: ok", fg=typer.colors.GREEN)
    else:
        typer.secho("integrity: FAIL", fg=typer.colors.RED)
        raise typer.Exit(2)


@bench_app.command("run")
def bench_run(
    suite_dir: Path = typer.Argument(...),
    out: Path = typer.Option(Path("bench-results/latest"), "--out", "-o"),
    model_id: Optional[str] = typer.Option(None, "--model-id"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    models_file: Optional[Path] = typer.Option(None, "--models-file"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c"),
    n: Optional[int] = typer.Option(None, "--n"),
    save_trajectories: bool = typer.Option(True, "--save-trajectories/--no-save-trajectories"),
    strict: bool = typer.Option(False, "--strict"),
    min_success_rate: Optional[float] = typer.Option(None, "--min-success-rate"),
    mock: bool = typer.Option(False, "--mock", help="Run with MockModelClient (no LLM)"),
) -> None:
    """Run a suite against one or more OpenAI-compatible models."""
    from agentbox.benchmark.loader import load_suite
    from agentbox.benchmark.runner import BenchmarkRunner
    from agentbox.benchmark.schema import BenchmarkRunConfig, ModelUnderTest
    from agentbox.config import ModelConfig
    from agentbox.model.base import ModelResponse
    from agentbox.model.mock import MockModelClient

    suite = load_suite(suite_dir)
    models: list[ModelUnderTest] = []

    if models_file:
        models.extend(_load_models_file(models_file))
    if model:
        models.append(
            ModelUnderTest(
                model_id=model_id or model,
                model=ModelConfig(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                ),
            )
        )
    if mock:
        # Special marker handled below
        models = [
            ModelUnderTest(
                model_id="mock",
                model=ModelConfig(model="mock", base_url="http://invalid", api_key="x"),
            )
        ]

    if not models:
        typer.secho("Provide --model / --models-file or --mock", fg=typer.colors.RED)
        raise typer.Exit(2)

    async def _run():
        # For --mock, inject via ParallelRunner is hard; use BenchmarkRunner with real
        # configs and replace model client by patching Run — simpler path: use mock
        # ModelConfig only for non-mock; for mock use dedicated path.
        if mock:
            return await _run_mock_suite(suite, out, concurrency, n, save_trajectories, strict)
        return await BenchmarkRunner(
            BenchmarkRunConfig(
                suite=suite,
                models=models,
                concurrency=concurrency,
                n_per_task=n,
                save_trajectories=save_trajectories,
                output_dir=out,
                strict_integrity=strict,
                tags={"agentbox": __version__},
            )
        ).run()

    report = asyncio.run(_run())
    typer.echo(f"report={out / 'report.json'}")
    for row in report.leaderboard:
        typer.echo(
            f"  {row['model_id']}: success_rate={row['success_rate']:.3f} "
            f"mean_steps={row['mean_steps']:.2f} n={row['n']}"
        )
    if min_success_rate is not None:
        best = max((r["success_rate"] for r in report.leaderboard), default=0.0)
        if best < min_success_rate:
            raise typer.Exit(1)
    raise typer.Exit(0)


async def _run_mock_suite(suite, out, concurrency, n, save_trajectories, strict):
    """Hermetic mock benchmark (no-op agent) for pipeline smoke tests."""
    import time
    import uuid
    from collections import defaultdict
    from datetime import datetime, timezone

    from agentbox.benchmark.report import build_leaderboard, build_model_report
    from agentbox.benchmark.schema import BenchmarkReport
    from agentbox.config import ModelConfig, RolloutConfig
    from agentbox.model.base import ModelResponse
    from agentbox.model.mock import MockModelClient
    from agentbox.runner.parallel import ParallelRunner

    n_per = n or suite.manifest.n_per_task
    mock = MockModelClient(
        [ModelResponse(content="(benchmark mock no-op)", tool_calls=[])]
        * max(8, len(suite.tasks) * n_per * 4)
    )
    agent = suite.manifest.agent.model_copy(update={"custom_tools": []})
    runner = ParallelRunner(
        concurrency=concurrency or suite.manifest.concurrency,
        config=RolloutConfig(
            sandbox=suite.manifest.sandbox,
            model=ModelConfig(model="mock"),
            agent=agent,
            seed=suite.manifest.seed,
        ),
        model=mock,
        agent=agent,
        sandbox=suite.manifest.sandbox,
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    traj_dir = out / "models" / "mock" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    path_map: dict[str, list[str]] = defaultdict(list)

    def on_t(traj) -> None:
        if save_trajectories:
            p = traj_dir / f"{traj.task_id}_{traj.run_id[:8]}.json"
            traj.save(p)
            path_map[traj.task_id].append(str(p))

    t0 = time.monotonic()
    trajs = await runner.run_tasks(
        suite.tasks,
        n_per_task=n_per,
        progress=True,
        on_trajectory=on_t,
        setup_checks=list(suite.manifest.scoring.setup_checks),
    )
    wall = time.monotonic() - t0
    mr = build_model_report(
        "mock",
        {"model": "mock"},
        trajs,
        wall_clock_s=wall,
        trajectory_paths=dict(path_map),
    )
    started = datetime.now(timezone.utc)
    report = BenchmarkReport(
        report_id=str(uuid.uuid4()),
        suite_id=suite.manifest.suite_id,
        suite_version=suite.manifest.version,
        suite_content_hash=suite.manifest.content_hash,
        agentbox_version=__version__,
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        models=[mr],
        leaderboard=build_leaderboard([mr]),
    )
    report.save_json(out / "report.json")
    report.save_markdown(out / "REPORT.md")
    return report


def _load_models_file(path: Path) -> list:
    from agentbox.benchmark.schema import ModelUnderTest
    from agentbox.config import ModelConfig

    text = path.read_text(encoding="utf-8")
    data: dict
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise typer.BadParameter(
                "PyYAML required for --models-file .yaml (pip install pyyaml)"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    models = []
    for item in data.get("models") or []:
        models.append(
            ModelUnderTest(
                model_id=item["model_id"],
                model=ModelConfig(
                    model=item["model"],
                    base_url=_interp_env(item.get("base_url")),
                    api_key=_interp_env(item.get("api_key")),
                    temperature=item.get("temperature", 0.7),
                ),
            )
        )
    return models


@bench_app.command("show")
def bench_show(report: Path = typer.Argument(...)) -> None:
    """Print leaderboard from a report.json."""
    from agentbox.benchmark.schema import BenchmarkReport

    r = BenchmarkReport.load(report)
    typer.echo(f"suite={r.suite_id} v{r.suite_version} hash={r.suite_content_hash}")
    for row in r.leaderboard:
        typer.echo(
            f"{row['model_id']}: success_rate={row['success_rate']:.3f} "
            f"mean_reward={row['mean_reward']:.3f} n={row['n']}"
        )


@bench_app.command("compare")
def bench_compare(
    a: Path = typer.Argument(..., help="report.json A"),
    b: Path = typer.Argument(..., help="report.json B"),
) -> None:
    """Compare two benchmark reports."""
    from agentbox.benchmark.compare import compare_reports
    from agentbox.benchmark.schema import BenchmarkReport

    result = compare_reports(BenchmarkReport.load(a), BenchmarkReport.load(b))
    typer.echo(f"suite_id={result.suite_id} hash_match={result.hash_match}")
    for w in result.warnings:
        typer.secho(f"warn: {w}", fg=typer.colors.YELLOW)
    for d in result.deltas:
        typer.echo(
            f"{d['model_id']}: Δsuccess={d['delta_success_rate']:+.3f} "
            f"Δsteps={d['delta_mean_steps']:+.2f}"
        )
