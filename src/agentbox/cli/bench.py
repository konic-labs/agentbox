"""`agentbox bench …`: frozen real-rollout benchmark suites."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from agentbox.cli._common import (
    EXIT_FAILURES,
    EXIT_INFRA,
    EXIT_OK,
    emit,
    fail,
    fmt_pct,
    state,
    table,
)

bench_app = typer.Typer(
    help="Create, freeze, run and compare benchmark suites.", no_args_is_help=True
)


@bench_app.command("create")
def bench_create(
    ctx: typer.Context,
    dest: Annotated[Path, typer.Argument(help="Output suite directory")],
    from_tasks: Annotated[Path, typer.Option("--from-tasks", help="Tasks directory or dataset")],
    suite_id: Annotated[str, typer.Option("--suite-id")],
    name: Annotated[str, typer.Option("--name")],
    version: Annotated[str, typer.Option("--version")] = "1.0.0",
    description: Annotated[str, typer.Option("--description")] = "",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 8,
    n: Annotated[int, typer.Option("--n", help="Rollouts per task")] = 1,
    network: Annotated[bool, typer.Option("--network/--no-network")] = False,
) -> None:
    """Create a suite pack from tasks and freeze its content hash."""
    from agentbox.benchmark.loader import create_suite_from_tasks
    from agentbox.config import ResourceLimits, SandboxConfig

    st = state(ctx)
    base = st.project.sandbox or SandboxConfig()
    suite = create_suite_from_tasks(
        from_tasks,
        dest,
        suite_id=suite_id,
        name=name,
        version=version,
        description=description,
        concurrency=concurrency,
        n_per_task=n,
        sandbox=base.model_copy(
            update={"limits": base.limits.model_copy(update={"network_disabled": not network})}
        )
        if base.limits
        else SandboxConfig(limits=ResourceLimits(network_disabled=not network)),
        freeze=True,
    )
    m = suite.manifest
    emit(
        st,
        {
            "suite_id": m.suite_id,
            "version": m.version,
            "tasks": len(suite.tasks),
            "hash": m.content_hash,
            "path": str(dest.resolve()),
        },
        lambda: typer.echo(
            f"created {m.suite_id} v{m.version} ({len(suite.tasks)} tasks) "
            f"hash={m.content_hash} → {dest.resolve()}"
        ),
    )


@bench_app.command("freeze")
def bench_freeze(ctx: typer.Context, suite_dir: Annotated[Path, typer.Argument()]) -> None:
    """Recompute and store the suite content hash."""
    from agentbox.benchmark.loader import load_suite, save_suite

    suite = load_suite(suite_dir).freeze()
    save_suite(suite, suite_dir)
    emit(
        state(ctx),
        {"hash": suite.manifest.content_hash},
        lambda: typer.echo(f"frozen hash={suite.manifest.content_hash}"),
    )


@bench_app.command("validate")
def bench_validate(
    ctx: typer.Context,
    suite_dir: Annotated[Path, typer.Argument()],
    strict: Annotated[bool, typer.Option("--strict")] = False,
) -> None:
    """Validate suite structure and integrity hash."""
    from agentbox.benchmark.loader import load_suite
    from agentbox.errors import SuiteIntegrityError

    st = state(ctx)
    suite = load_suite(suite_dir)
    try:
        ok = suite.verify_integrity(strict=strict)
    except SuiteIntegrityError as exc:
        fail(str(exc), code=EXIT_FAILURES)
    emit(
        st,
        {"suite_id": suite.manifest.suite_id, "tasks": len(suite.tasks), "integrity": ok},
        lambda: typer.secho(
            f"{suite.manifest.suite_id}: {len(suite.tasks)} tasks · integrity "
            f"{'ok' if ok else 'MISMATCH'}",
            fg=typer.colors.GREEN if ok else typer.colors.RED,
        ),
    )
    raise typer.Exit(EXIT_OK if ok else EXIT_FAILURES)


def _students(specs: list[str], api_key_env: str | None) -> list[Any]:
    import os

    from agentbox.benchmark.schema import ModelUnderTest
    from agentbox.config import ModelConfig

    out = []
    for spec in specs:
        if "=" not in spec or "@" not in spec:
            raise typer.BadParameter(f"invalid --student {spec!r}; expected id=model@base_url")
        mid, rest = spec.split("=", 1)
        model_name, burl = rest.rsplit("@", 1)
        out.append(
            ModelUnderTest(
                model_id=mid,
                model=ModelConfig(
                    model=model_name,
                    base_url=burl,
                    api_key_env=api_key_env,
                    api_key=os.environ.get(api_key_env) if api_key_env else None,
                ),
            )
        )
    return out


def _models_file(path: Path) -> list[Any]:
    from agentbox.benchmark.schema import ModelUnderTest
    from agentbox.config_load import EndpointConfig, _walk_interp

    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    items = data.get("models") or []
    if isinstance(items, dict):
        items = [{"model_id": k, **v} for k, v in items.items()]
    out = []
    for item in _walk_interp(items):
        mid = item.pop("model_id", None) or item.pop("id", None) or item["model"]
        out.append(ModelUnderTest(model_id=mid, model=EndpointConfig(**item).to_model_config()))
    return out


@bench_app.command("run")
def bench_run(
    ctx: typer.Context,
    suite_dir: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("bench-results/latest"),
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    api_key_env: Annotated[str | None, typer.Option("--api-key-env")] = None,
    profile: Annotated[
        list[str] | None, typer.Option("--profile", help="Model profile(s) to run")
    ] = None,
    student: Annotated[
        list[str] | None, typer.Option("--student", help="id=MODEL@BASE_URL")
    ] = None,
    models_file: Annotated[Path | None, typer.Option("--models-file")] = None,
    mock: Annotated[bool, typer.Option("--mock", help="No-op model (pipeline smoke test)")] = False,
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c")] = None,
    n: Annotated[int | None, typer.Option("--n")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    strict: Annotated[bool, typer.Option("--strict", help="Fail on integrity mismatch")] = False,
    probe: Annotated[bool, typer.Option("--probe", help="Probe tool calling first")] = False,
    save_trajectories: Annotated[
        bool, typer.Option("--save-trajectories/--no-save-trajectories")
    ] = True,
    min_success_rate: Annotated[float | None, typer.Option("--min-success-rate")] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
) -> None:
    """Run a suite against one or more OpenAI-compatible models."""
    from agentbox.benchmark.loader import load_suite
    from agentbox.benchmark.runner import BenchmarkRunner
    from agentbox.benchmark.schema import BenchmarkRunConfig, ModelUnderTest
    from agentbox.cli._common import resolve_model
    from agentbox.config import ModelConfig
    from agentbox.model.base import ModelResponse
    from agentbox.model.mock import MockModelClient
    from agentbox.version import __version__

    st = state(ctx)
    suite = load_suite(suite_dir)
    if limit:
        suite.tasks = suite.tasks[:limit]
    models: list[ModelUnderTest] = []
    clients: dict[str, Any] = {}
    if mock:
        models = [ModelUnderTest(model_id="mock", model=ModelConfig(model="mock"))]
        clients["mock"] = MockModelClient([ModelResponse(content="(benchmark mock no-op)")])
    else:
        if models_file:
            models.extend(_models_file(models_file))
        if student:
            models.extend(_students(student, api_key_env))
        for p in profile or []:
            models.append(
                ModelUnderTest(
                    model_id=p,
                    model=resolve_model(st, model=None, base_url=None, api_key_env=None, profile=p),
                )
            )
        if model:
            cfg = resolve_model(st, model=model, base_url=base_url, api_key_env=api_key_env)
            models.append(
                ModelUnderTest(
                    model_id=model_id or model.replace("/", "-").replace(":", "-"), model=cfg
                )
            )
        if not models and st.project.all_models():
            models = [
                ModelUnderTest(model_id=k, model=v.to_model_config())
                for k, v in st.project.all_models().items()
                if k != "teacher"
            ]
    if not models:
        fail(
            "provide --model / --profile / --student / --models-file / --mock "
            "or models in agentbox.yaml"
        )

    if probe and not mock:
        from agentbox.model.probe import format_probe_results, probe_endpoint

        for mut in models:
            results = asyncio.run(probe_endpoint(mut.model, require_tools=True))
            typer.echo(f"probe {mut.model_id}:\n{format_probe_results(results)}")
            if any(not r.ok for r in results):
                fail(f"probe failed for {mut.model_id}", code=EXIT_INFRA)

    report = asyncio.run(
        BenchmarkRunner(
            BenchmarkRunConfig(
                suite=suite,
                models=models,
                concurrency=concurrency,
                n_per_task=n,
                save_trajectories=save_trajectories,
                output_dir=out,
                strict_integrity=strict,
                fail_fast=fail_fast,
                resume=resume,
                progress=progress and not st.json and not st.quiet,
                tags={"agentbox": __version__},
            ),
            model_clients=clients,
        ).run()
    )

    def human() -> None:
        ks = sorted({k for r in report.leaderboard for k in r if k.startswith("pass@")})
        table(
            ["model", "success", *ks, "mean reward", "mean steps", "infra", "n"],
            [
                [
                    r["model_id"],
                    fmt_pct(r["success_rate"]),
                    *[fmt_pct(r.get(k)) for k in ks],
                    f"{r['mean_reward']:.3f}",
                    f"{r['mean_steps']:.1f}",
                    r.get("infra_failures", 0),
                    r["n"],
                ]
                for r in report.leaderboard
            ],
            title=f"{report.suite_id} → {out / 'report.json'}",
        )

    emit(st, {"report": str(out / "report.json"), "leaderboard": report.leaderboard}, human)
    if min_success_rate is not None:
        best = max((r["success_rate"] for r in report.leaderboard), default=0.0)
        if best < min_success_rate:
            raise typer.Exit(EXIT_FAILURES)


@bench_app.command("show")
def bench_show(
    ctx: typer.Context,
    report: Annotated[Path, typer.Argument(help="report.json or run directory")],
    out: Annotated[Path | None, typer.Option("--out", help="Write extended markdown")] = None,
) -> None:
    """Leaderboard, per-task matrix and failure clusters of a report."""
    import hashlib
    import re
    from collections import defaultdict

    from agentbox.benchmark.report import render_markdown
    from agentbox.benchmark.schema import BenchmarkReport
    from agentbox.trajectory.io import iter_trajectories

    path = report / "report.json" if report.is_dir() else report
    r = BenchmarkReport.load(path)
    lines = [render_markdown(r)]
    model_ids = [m.model_id for m in r.models]
    if len(model_ids) > 1:
        lines += [
            "",
            "## Task × model pass@1",
            "",
            "| Task | " + " | ".join(model_ids) + " |",
            "| --- |" + " --- |" * len(model_ids),
        ]
        per: dict[str, dict[str, float]] = defaultdict(dict)
        for m in r.models:
            for t in m.by_task:
                per[t.task_id][m.model_id] = t.pass_at_1
        for tid in sorted(per):
            lines.append(
                f"| {tid} | "
                + " | ".join(f"{per[tid].get(mid, 0):.2f}" for mid in model_ids)
                + " |"
            )
    clusters: dict[str, list[str]] = defaultdict(list)
    for tr in iter_trajectories(path.parent):
        if tr.success:
            continue
        v = tr.verify
        head = re.sub(r"\d+\.\d+s|0x[0-9a-f]+", "#", (v.stdout if v else tr.error or "")[:200])
        key = f"{tr.failure_kind}|exit={v.exit_code if v else None}|{head}"
        clusters[
            hashlib.sha1(key.encode()).hexdigest()[:10] + " " + key.replace("\n", " ")[:90]
        ].append(tr.task_id)
    if clusters:
        lines += ["", "## Top failure clusters", ""]
        for fp, ids in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:12]:
            lines.append(f"- **n={len(ids)}** `{fp}` (e.g. `{ids[0]}`)")
    text = "\n".join(lines) + "\n"
    if out:
        out.write_text(text, encoding="utf-8")
    emit(
        state(ctx),
        {"report": r.model_dump(mode="json"), "clusters": {k: len(v) for k, v in clusters.items()}},
        lambda: typer.echo(text if not out else f"wrote {out}"),
    )


@bench_app.command("compare")
def bench_compare(
    ctx: typer.Context,
    a: Annotated[Path, typer.Argument(help="Baseline report.json")],
    b: Annotated[Path, typer.Argument(help="Candidate report.json")],
    fail_on_regression: Annotated[
        float | None,
        typer.Option("--fail-on-regression", help="Exit 1 if success drops by more than X"),
    ] = None,
) -> None:
    """Paired per-task comparison with bootstrap CI and sign test."""
    from agentbox.benchmark.compare import compare_reports
    from agentbox.benchmark.schema import BenchmarkReport

    res = compare_reports(BenchmarkReport.load(a), BenchmarkReport.load(b))

    def human() -> None:
        for w in res.warnings:
            typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
        table(
            ["model", "success A", "success B", "Δ", "95% CI (paired)", "W/L/T", "sign p"],
            [
                [
                    d["model_id"],
                    fmt_pct(d["success_rate_a"]),
                    fmt_pct(d["success_rate_b"]),
                    f"{d['delta_success_rate']:+.3f}",
                    "[{:+.3f}, {:+.3f}]".format(*d["paired"].get("ci95", [0, 0])),
                    "/".join(str(d["paired"].get(k, 0)) for k in ("wins", "losses", "ties")),
                    f"{d['paired'].get('sign_test_p', 1):.3f}",
                ]
                for d in res.deltas
            ],
        )

    emit(state(ctx), res.model_dump(), human)
    if fail_on_regression is not None and any(
        d["delta_success_rate"] < -fail_on_regression for d in res.deltas
    ):
        raise typer.Exit(EXIT_FAILURES)
