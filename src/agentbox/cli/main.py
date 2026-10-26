"""AgentBox command-line interface (Typer)."""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentbox.cli._common import (
    EXIT_INFRA,
    EXIT_OK,
    EXIT_USAGE,
    State,
    emit,
    fail,
    resolve_model,
    setup_logging,
    state,
    table,
)
from agentbox.version import __version__

app = typer.Typer(
    name="agentbox",
    help=(
        "Docker sandboxes, tools, rewards and trajectories for RL post-training of LLM "
        "agents — trainer-agnostic (TRL, verl, OpenRLHF, SkyRL, ART)."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _root(
    ctx: typer.Context,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-C", envvar="AGENTBOX_CONFIG", help="Project config file."),
    ] = None,
    profile: Annotated[
        Optional[str],
        typer.Option("--profile", "-P", envvar="AGENTBOX_PROFILE", help="Model profile from the config."),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="-v info, -vv debug.")] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only errors.")] = False,
    log_format: Annotated[str, typer.Option("--log-format", help="text | json")] = "text",
) -> None:
    ctx.obj = State(config_path=config, profile=profile, json=json_out, verbose=verbose, quiet=quiet)
    setup_logging(verbose, quiet, log_format)


# ------------------------------------------------------------------ sub-apps
from agentbox.cli import bench as _bench  # noqa: E402
from agentbox.cli import cache as _cache  # noqa: E402
from agentbox.cli import generate as _generate  # noqa: E402
from agentbox.cli import run as _run  # noqa: E402
from agentbox.cli import sandbox as _sandbox  # noqa: E402
from agentbox.cli import serve as _serve  # noqa: E402
from agentbox.cli import task as _task  # noqa: E402
from agentbox.cli import train as _train  # noqa: E402
from agentbox.cli import traj as _traj  # noqa: E402

app.command("run", help="Run rollouts for tasks (files, dirs, JSONL) and stream trajectories.")(_run.run)
app.add_typer(_sandbox.sandbox_app, name="sandbox")
app.add_typer(_cache.cache_app, name="cache")
app.add_typer(_cache.image_app, name="image")
app.add_typer(_task.task_app, name="task")
app.add_typer(_traj.traj_app, name="traj")
app.add_typer(_bench.bench_app, name="bench")
app.add_typer(_generate.generate_app, name="generate")
app.add_typer(_train.train_app, name="train")
app.command("serve", help="Serve AgentBox environments over HTTP (remote sandboxes).")(_serve.serve)

config_app = typer.Typer(help="Inspect the project configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@app.command()
def version(ctx: typer.Context) -> None:
    """Print the AgentBox version."""
    st = state(ctx)
    emit(st, {"version": __version__}, lambda: typer.echo(__version__))


@app.command()
def doctor(
    ctx: typer.Context,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Probe this model id.")] = None,
    base_url: Annotated[Optional[str], typer.Option("--base-url")] = None,
    api_key_env: Annotated[Optional[str], typer.Option("--api-key-env")] = None,
    image: Annotated[Optional[str], typer.Option("--image", help="Image to check.")] = None,
    skip_tools: Annotated[bool, typer.Option("--skip-tools", help="Only probe plain chat.")] = False,
    probe: Annotated[bool, typer.Option("--probe/--no-probe", help="Probe the model endpoint.")] = True,
) -> None:
    """Check Python, Docker, sandbox image, caches, integrations and (optionally) a model."""
    st = state(ctx)
    report: dict[str, object] = {
        "agentbox": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    code = EXIT_OK
    try:
        from agentbox.sandbox.backend import get_backend
        from agentbox.sandbox.prune import select_containers

        backend = get_backend()
        info = backend.client.info()
        report["docker"] = {
            "ok": True,
            "server": info.get("ServerVersion"),
            "cgroup": info.get("CgroupVersion"),
            "storage": info.get("Driver"),
            "root": info.get("DockerRootDir"),
            "cpus": info.get("NCPU"),
            "mem_gb": round((info.get("MemTotal") or 0) / 2**30, 1),
        }
        root = info.get("DockerRootDir")
        if root and Path(root).exists():
            report["docker"]["free_gb"] = round(shutil.disk_usage(root).free / 2**30, 1)  # type: ignore[index]
        cfg_image = image or (st.project.sandbox.image if st.project.sandbox else None)
        from agentbox.config import SandboxConfig

        img = cfg_image or SandboxConfig().image
        present = backend.image_exists_sync(img)
        img_report: dict[str, object] = {"image": img, "present": present}
        if present:
            prof = backend.image_profile_sync(img)
            img_report.update(bash=prof.has_bash, python=prof.has_python, pytest=prof.has_pytest)
        report["image"] = img_report
        containers = backend.list_containers_sync()
        report["containers"] = {
            "total": len(containers),
            "orphans": len(select_containers(containers, scope="orphans")),
        }
        report["cache_images"] = len(backend.list_cache_images_sync())
    except Exception as exc:
        report["docker"] = {"ok": False, "error": str(exc)}
        code = EXIT_INFRA

    from agentbox.integrations import available_integrations

    report["integrations"] = {r["name"]: r["version"] for r in available_integrations()}

    if probe and (model or st.profile or st.project.default_model):
        from agentbox.model.probe import probe_endpoint

        cfg = resolve_model(
            st,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            overrides={"temperature": 0.0, "max_tokens": 64, "timeout_s": 60.0},
        )
        results = asyncio.run(probe_endpoint(cfg, require_tools=not skip_tools))
        report["model"] = {
            "model": cfg.model,
            "base_url": cfg.base_url,
            "checks": [r.model_dump() for r in results],
        }
        if any(not r.ok for r in results):
            code = EXIT_INFRA

    def human() -> None:
        d = report.get("docker", {})
        ok = isinstance(d, dict) and d.get("ok")
        typer.echo(f"agentbox {__version__} · python {report['python']}")
        if ok and isinstance(d, dict):
            typer.secho(
                f"docker: ok (server {d['server']}, cgroup v{d['cgroup']}, {d['storage']}, "
                f"{d['cpus']} cpu, {d['mem_gb']} GB, free {d.get('free_gb', '?')} GB)",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(f"docker: FAIL ({d.get('error') if isinstance(d, dict) else d})", fg=typer.colors.RED)
        if "image" in report:
            typer.echo(f"image: {report['image']}")
            typer.echo(f"containers: {report['containers']} · cache images: {report['cache_images']}")
        installed = {k: v for k, v in report["integrations"].items() if v}  # type: ignore[union-attr]
        typer.echo(f"integrations installed: {installed or 'none'}")
        if "model" in report:
            from agentbox.model.probe import ProbeResult, format_probe_results

            m = report["model"]
            checks = [ProbeResult.model_validate(c) for c in m["checks"]]  # type: ignore[index]
            typer.echo(f"model probe: {m['model']} @ {m['base_url'] or 'default'}")  # type: ignore[index]
            typer.echo(format_probe_results(checks))

    emit(st, report, human)
    raise typer.Exit(code)


@app.command()
def integrations(ctx: typer.Context) -> None:
    """List trainer integrations, their entry points and installed versions."""
    from agentbox.integrations import available_integrations

    rows = available_integrations()
    st = state(ctx)
    emit(
        st,
        rows,
        lambda: table(
            ["name", "installed", "entry point", "install", "what"],
            [[r["name"], r["version"] or "—", r["entry_point"], r["install"], r["description"]] for r in rows],
            title="AgentBox integrations",
        ),
    )


_INIT_CONFIG = """\
# AgentBox project config — see `agentbox config show`
default_model: local
models:
  local:
    model: Qwen/Qwen3-1.7B
    base_url: http://localhost:8000/v1
    api_key: ${VLLM_API_KEY:-EMPTY}
    extra_body:
      chat_template_kwargs: {enable_thinking: false}
sandbox:
  image: python:3.12-slim-bookworm
  limits: {cpu_count: 1.0, memory_mb: 1024, network_disabled: true}
agent:
  tools: structured
  max_steps: 30
rewards:
  - {name: verifier}
"""

_EXAMPLE_TASK = {
    "task_id": "example_add",
    "description": "Fix add() in calc.py so that the tests in test_calc.py pass.",
    "starter_files": {
        "calc.py": "def add(a, b):\n    return a - b\n",
        "test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    },
    "verifier": {
        "type": "pytest",
        "command": "python -m pytest -q",
        "protected_paths": ["test_*.py"],
    },
    "metadata": {"difficulty": "easy", "tags": ["python"]},
}


@app.command()
def init(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Argument(help="Project directory")] = Path("."),
    trainer: Annotated[
        Optional[str], typer.Option("--trainer", help="Also scaffold a trainer script: trl")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Scaffold agentbox.yaml, an example task and (optionally) a trainer script."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    targets: dict[Path, str] = {
        directory / "agentbox.yaml": _INIT_CONFIG,
        directory / "tasks" / "example_add" / "task.json": json.dumps(_EXAMPLE_TASK, indent=2) + "\n",
    }
    if trainer == "trl":
        from agentbox.cli.train import TRL_TEMPLATE

        targets[directory / "train_trl.py"] = TRL_TEMPLATE
    elif trainer:
        fail(f"no template for trainer {trainer!r} (available: trl)")
    for path, content in targets.items():
        if path.exists() and not force:
            typer.echo(f"skip {path} (exists; --force to overwrite)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    emit(state(ctx), {"written": written}, lambda: [typer.echo(f"wrote {p}") for p in written])


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the resolved project configuration (API keys redacted)."""
    st = state(ctx)
    cfg = st.project
    data = cfg.model_dump(mode="json", exclude={"path"})
    for prof in data.get("models", {}).values():
        if prof.get("api_key"):
            prof["api_key"] = "***"
    data["path"] = str(cfg.path) if cfg.path else None
    emit(st, data)


@config_app.command("validate")
def config_validate(ctx: typer.Context) -> None:
    """Validate the project configuration (exit 2 on errors)."""
    st = state(ctx)
    from agentbox.config_load import load_project_config

    try:
        cfg = load_project_config(st.config_path, required=True)
    except Exception as exc:
        fail(str(exc), code=EXIT_USAGE)
    emit(
        st,
        {"ok": True, "path": str(cfg.path), "profiles": sorted(cfg.all_models())},
        lambda: typer.secho(f"ok: {cfg.path} ({len(cfg.all_models())} model profiles)", fg=typer.colors.GREEN),
    )


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Print the config file in use."""
    st = state(ctx)
    p = st.project.path
    emit(st, {"path": str(p) if p else None}, lambda: typer.echo(str(p) if p else "(none)"))


# --------------------------------------------------------------- legacy aliases
@app.command("run-dir", hidden=True)
def _legacy_run_dir(
    ctx: typer.Context,
    tasks_dir: Path,
    model: Annotated[str, typer.Option("--model", "-m")],
    base_url: Annotated[Optional[str], typer.Option("--base-url")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 8,
    n: Annotated[int, typer.Option("--n")] = 1,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("trajectories"),
    network: Annotated[bool, typer.Option("--network")] = False,
) -> None:
    """Deprecated: use `agentbox run DIR`."""
    ctx.invoke(
        _run.run, ctx=ctx, tasks=[tasks_dir], model=model, base_url=base_url,
        concurrency=concurrency, n=n, out=out, network=network,
    )


@app.command("export", hidden=True)
def _legacy_export(
    ctx: typer.Context,
    traj: Path,
    format: Annotated[str, typer.Option("--format", "-f")] = "art",  # noqa: A002
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("out.jsonl"),
) -> None:
    """Deprecated: use `agentbox traj export`."""
    ctx.invoke(_traj.traj_export, ctx=ctx, source=traj, fmt=format, out=out)


@app.command("prune", hidden=True)
def _legacy_prune(ctx: typer.Context) -> None:
    """Deprecated: use `agentbox sandbox prune`."""
    ctx.invoke(_sandbox.sandbox_prune, ctx=ctx)


@app.command("build-image", hidden=True)
def _legacy_build_image(
    ctx: typer.Context, tag: Annotated[str, typer.Option("--tag", "-t")] = "agentbox/sandbox:latest"
) -> None:
    """Deprecated: use `agentbox image build`."""
    ctx.invoke(_cache.image_build, ctx=ctx, tag=tag)


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
