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
    student: Optional[list[str]] = typer.Option(
        None,
        "--student",
        help="Repeatable student spec: id=MODEL@BASE_URL (api key via env/OPENAI_API_KEY)",
    ),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c"),
    n: Optional[int] = typer.Option(None, "--n"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only first N tasks"),
    save_trajectories: bool = typer.Option(True, "--save-trajectories/--no-save-trajectories"),
    strict: bool = typer.Option(False, "--strict"),
    min_success_rate: Optional[float] = typer.Option(None, "--min-success-rate"),
    mock: bool = typer.Option(False, "--mock", help="Run with MockModelClient (no LLM)"),
    probe: bool = typer.Option(
        False, "--probe", help="Probe tool-calling on each model before the suite"
    ),
    disable_thinking: bool = typer.Option(
        False, "--disable-thinking", help="Send Qwen/GLM thinking-off extra_body"
    ),
) -> None:
    """Run a suite against one or more OpenAI-compatible models."""
    from agentbox.benchmark.loader import load_suite
    from agentbox.benchmark.runner import BenchmarkRunner
    from agentbox.benchmark.schema import BenchmarkRunConfig, ModelUnderTest
    from agentbox.config import ModelConfig
    from agentbox.model.base import ModelResponse
    from agentbox.model.mock import MockModelClient

    suite = load_suite(suite_dir)
    if limit and limit > 0:
        suite.tasks = suite.tasks[:limit]
        typer.echo(f"SUBSET: first {len(suite.tasks)} tasks")
    models: list[ModelUnderTest] = []

    if models_file:
        models.extend(_load_models_file(models_file))
    if student:
        models.extend(_parse_students(student, api_key=api_key, disable_thinking=disable_thinking))
    if model:
        extra: dict = {}
        if disable_thinking:
            extra = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        models.append(
            ModelUnderTest(
                model_id=model_id or model.replace("/", "-").replace(":", "-"),
                model=ModelConfig(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    extra_body=extra,
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

    if not models and not mock:
        # Fall back to agentbox.yaml students
        try:
            from agentbox.config_load import load_project_config

            pcfg = load_project_config()
            for s in pcfg.students:
                mid = s.id or s.model.replace("/", "-").replace(":", "-")
                models.append(
                    ModelUnderTest(
                        model_id=mid,
                        model=ModelConfig(
                            model=s.model,
                            base_url=s.base_url,
                            api_key=s.api_key or api_key or "EMPTY",
                            temperature=s.temperature,
                            max_tokens=s.max_tokens,
                            extra_body=dict(s.extra_body or {}),
                        ),
                    )
                )
        except Exception:
            pass

    if not models:
        typer.secho(
            "Provide --model / --student / --models-file / --mock "
            "or students in agentbox.yaml",
            fg=typer.colors.RED,
        )
        raise typer.Exit(2)

    if probe and not mock:
        from agentbox.model.probe import format_probe_results, probe_endpoint

        for mut in models:
            typer.echo(f"probe {mut.model_id}…")
            results = asyncio.run(probe_endpoint(mut.model, require_tools=True))
            typer.echo(format_probe_results(results))
            if any(not r.ok for r in results):
                typer.secho(f"probe failed for {mut.model_id}", fg=typer.colors.RED)
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


def _parse_students(
    specs: list[str],
    *,
    api_key: str | None,
    disable_thinking: bool,
) -> list:
    """Parse id=model@base_url student specs."""
    from agentbox.benchmark.schema import ModelUnderTest
    from agentbox.config import ModelConfig

    out = []
    for spec in specs:
        # id=model@http://host:port/v1
        if "=" not in spec or "@" not in spec:
            raise typer.BadParameter(
                f"Invalid --student {spec!r}; expected id=model@base_url"
            )
        mid, rest = spec.split("=", 1)
        model_name, burl = rest.rsplit("@", 1)
        extra: dict = {}
        if disable_thinking:
            extra = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        out.append(
            ModelUnderTest(
                model_id=mid,
                model=ModelConfig(
                    model=model_name,
                    base_url=burl,
                    api_key=api_key or "EMPTY",
                    extra_body=extra,
                ),
            )
        )
    return out


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
def bench_show(
    report: Path = typer.Argument(...),
    out: Optional[Path] = typer.Option(None, "--out", help="Write extended markdown"),
    traj_dir: Optional[Path] = typer.Option(
        None,
        "--traj-dir",
        help="Optional run dir to mine failure clusters from trajectories",
    ),
) -> None:
    """Print leaderboard and per-model status breakdown from a report.json."""
    from collections import Counter

    from agentbox.benchmark.schema import BenchmarkReport

    r = BenchmarkReport.load(report)
    typer.echo(f"suite={r.suite_id} v{r.suite_version} hash={r.suite_content_hash}")
    lines = [
        f"# Bench report: {r.suite_id}",
        "",
        f"- version: {r.suite_version}",
        f"- content_hash: `{r.suite_content_hash}`",
        "",
        "## Leaderboard",
        "",
    ]
    for row in r.leaderboard:
        typer.echo(
            f"{row['model_id']}: success_rate={row['success_rate']:.3f} "
            f"mean_reward={row['mean_reward']:.3f} mean_steps={row.get('mean_steps', 0):.2f} "
            f"n={row['n']}"
        )
        lines.append(
            f"- **{row['model_id']}**: success={row['success_rate']:.3f} "
            f"reward={row['mean_reward']:.3f} steps={row.get('mean_steps', 0):.2f} n={row['n']}"
        )

    lines.append("")
    lines.append("## Per-model status & tasks")
    lines.append("")
    # Cross-model task matrix when multiple models
    model_ids = [mr.model_id for mr in r.models]
    task_pass: dict[str, dict[str, float]] = {}
    for mr in r.models:
        typer.echo(f"\n[{mr.model_id}]")
        lines.append(f"### {mr.model_id}")
        agg = getattr(mr, "aggregate", None)
        if agg is not None:
            by_status = getattr(agg, "by_status", None) or {}
            if by_status:
                typer.echo(f"  statuses={by_status}")
                lines.append(f"- statuses: `{by_status}`")
            typer.echo(
                f"  success_rate={getattr(agg, 'success_rate', mr.success_rate if hasattr(mr, 'success_rate') else 'n/a')}"
            )
        # per-task rows
        task_rows = getattr(mr, "by_task", None) or []
        if task_rows:
            lines.append("")
            lines.append("| Task | successes | pass@1 | mean steps | statuses |")
            lines.append("| --- | --- | --- | --- | --- |")
            for tr in task_rows:
                if hasattr(tr, "model_dump"):
                    d = tr.model_dump()
                elif isinstance(tr, dict):
                    d = tr
                else:
                    continue
                tid = d.get("task_id") or "?"
                suc = d.get("successes", 0)
                p1 = d.get("pass_at_1", d.get("success_rate", 0))
                ms = d.get("mean_steps", 0)
                st = d.get("statuses") or {}
                task_pass.setdefault(tid, {})[mr.model_id] = float(p1 or 0)
                typer.echo(
                    f"  {tid}: successes={suc} pass@1={p1} steps={ms} statuses={st}"
                )
                lines.append(f"| {tid} | {suc} | {p1} | {ms} | `{st}` |")
        lines.append("")

    if len(model_ids) > 1 and task_pass:
        lines.append("## Task × model pass matrix")
        lines.append("")
        header = "| Task | " + " | ".join(model_ids) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(model_ids)) + " |"
        lines.append(header)
        lines.append(sep)
        for tid in sorted(task_pass):
            cells = [
                f"{task_pass[tid].get(mid, 0):.2f}" for mid in model_ids
            ]
            lines.append(f"| {tid} | " + " | ".join(cells) + " |")
        lines.append("")

    # Failure clusters from trajectories if available
    run_dir = traj_dir
    if run_dir is None:
        cand = report.parent if report.is_file() else report
        if (cand / "models").is_dir():
            run_dir = cand
    if run_dir and run_dir.is_dir():
        clusters = _failure_clusters(run_dir)
        if clusters:
            lines.append("## Top failure clusters")
            lines.append("")
            typer.echo("\n[failure clusters]")
            for fp, count, sample in clusters[:12]:
                typer.echo(f"  n={count}  {fp[:100]}")
                lines.append(f"- **n={count}** `{fp[:120]}`")
                if sample:
                    lines.append(f"  - sample task: `{sample}`")
            lines.append("")

    if out:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")


def _failure_clusters(run_dir: Path) -> list[tuple[str, int, str]]:
    """Fingerprint failed trajs by verify_exit + truncated verify_stdout head."""
    import hashlib
    import re
    from collections import defaultdict

    from agentbox.trajectory.schema import Trajectory

    buckets: dict[str, list[str]] = defaultdict(list)
    for path in run_dir.rglob("*.json"):
        if path.name == "report.json" or "suite" in path.parts:
            continue
        try:
            t = Trajectory.load(path)
        except Exception:
            continue
        status = t.final_status.value if hasattr(t.final_status, "value") else str(t.final_status)
        if status == "success":
            continue
        meta = t.metadata or {}
        exit_c = meta.get("verify_exit_code")
        stdout = str(meta.get("verify_stdout") or "")[:400]
        # normalize volatile bits
        stdout = re.sub(r"\d+\.\d+s", "Ts", stdout)
        stdout = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", stdout)
        key_src = f"exit={exit_c}|{stdout[:200]}"
        fp = hashlib.sha1(key_src.encode()).hexdigest()[:10] + " " + key_src.replace("\n", " ")[:80]
        buckets[fp].append(t.task_id)
    ranked = sorted(
        ((fp, len(ids), ids[0] if ids else "") for fp, ids in buckets.items()),
        key=lambda x: -x[1],
    )
    return ranked


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
