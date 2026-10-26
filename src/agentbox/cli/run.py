"""`agentbox run`: stream rollouts into a run directory (resumable)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from agentbox.cli._common import (
    EXIT_FAILURES,
    EXIT_OK,
    EXIT_USAGE,
    emit,
    fail,
    fmt_pct,
    parse_kv,
    resolve_model,
    state,
    table,
)


def _reward_specs(values: list[str] | None) -> Any:
    """``--reward name[:weight][,k=v...]`` e.g. ``step_penalty:1,per_step=0.01``."""
    from agentbox.rewards import RewardSpec

    if not values:
        return None
    specs = []
    for raw in values:
        head, *params = raw.split(",")
        name, _, weight = head.partition(":")
        kv = parse_kv(params, what="reward param")
        specs.append(
            RewardSpec(
                name=name,
                weight=float(weight) if weight else 1.0,
                params={k: _coerce(v) for k, v in kv.items()},
            )
        )
    return specs


def _coerce(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def run(
    ctx: typer.Context,
    tasks: Annotated[list[Path], typer.Argument(help="Task files, task dirs, datasets (*.jsonl[.gz], *.parquet)")],
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model id (or use --profile)")] = None,
    base_url: Annotated[Optional[str], typer.Option("--base-url", help="OpenAI-compatible endpoint")] = None,
    api_key_env: Annotated[Optional[str], typer.Option("--api-key-env", help="Env var holding the API key")] = None,
    mock: Annotated[bool, typer.Option("--mock", help="No-op model (pipeline smoke tests)")] = False,
    tito: Annotated[
        Optional[str],
        typer.Option("--tito", help="Token-in/token-out via /v1/completions with this HF tokenizer"),
    ] = None,
    n: Annotated[int, typer.Option("--n", "-n", help="Rollouts per task (GRPO group size)")] = 1,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 8,
    model_concurrency: Annotated[Optional[int], typer.Option("--model-concurrency")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Run directory")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Skip rollouts already in --out")] = False,
    force: Annotated[bool, typer.Option("--force", help="Resume even if the config changed")] = False,
    compress: Annotated[bool, typer.Option("--compress/--no-compress", help="Write .jsonl.gz")] = False,
    limit: Annotated[Optional[int], typer.Option("--limit", help="Only the first N tasks")] = None,
    tools: Annotated[Optional[str], typer.Option("--tools", help="structured|extended|shell|custom")] = None,
    max_steps: Annotated[Optional[int], typer.Option("--max-steps")] = None,
    temperature: Annotated[Optional[float], typer.Option("--temperature")] = None,
    max_tokens: Annotated[Optional[int], typer.Option("--max-tokens")] = None,
    capture_tokens: Annotated[bool, typer.Option("--capture-tokens", help="Record token ids + logprobs")] = False,
    image: Annotated[Optional[str], typer.Option("--image")] = None,
    network: Annotated[Optional[bool], typer.Option("--network/--no-network")] = None,
    reward: Annotated[Optional[list[str]], typer.Option("--reward", help="name[:weight][,k=v] (repeat)")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    retries: Annotated[int, typer.Option("--retries", help="Retries for infra failures")] = 1,
    keep_failed: Annotated[bool, typer.Option("--keep-failed", help="Keep sandboxes of failed rollouts")] = False,
    min_success_rate: Annotated[Optional[float], typer.Option("--min-success-rate")] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
) -> None:
    from agentbox.config import AgentConfig, ModelConfig, ResourceLimits, RolloutConfig, SandboxConfig
    from agentbox.errors import TaskValidationError
    from agentbox.model.base import ModelResponse
    from agentbox.model.mock import MockModelClient
    from agentbox.runner.parallel import ParallelRunner
    from agentbox.tasks.loader import iter_tasks
    from agentbox.trajectory.io import JsonlSink
    from agentbox.version import __version__

    st = state(ctx)
    proj = st.project
    try:
        task_list = list(iter_tasks(tasks))
    except TaskValidationError as exc:
        fail(str(exc))
    if limit:
        task_list = task_list[:limit]
    if not task_list:
        fail("no tasks found")

    sandbox = proj.sandbox or SandboxConfig()
    sb_update: dict[str, Any] = {}
    if image:
        sb_update["image"] = image
    if network is not None:
        sb_update["limits"] = sandbox.limits.model_copy(update={"network_disabled": not network})
    if keep_failed:
        sb_update["keep_on_failure"] = True
    sandbox = sandbox.model_copy(update=sb_update) if sb_update else sandbox
    agent = proj.agent or AgentConfig()
    ag_update: dict[str, Any] = {}
    if tools:
        ag_update["tools"] = tools
    if max_steps:
        ag_update["max_steps"] = max_steps
    agent = AgentConfig.model_validate({**agent.model_dump(), **ag_update}) if ag_update else agent

    model_client: Any
    if mock:
        model_cfg = ModelConfig(model="mock")
        model_client = MockModelClient([ModelResponse(content="(mock) done", finish_reason="stop")])
    else:
        model_cfg = resolve_model(
            st,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            overrides={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "capture_tokens": True if capture_tokens else None,
                "capture_logprobs": True if capture_tokens else None,
            },
        )
        model_client = model_cfg
        if tito:
            from agentbox.tito import ChatTokenizer, CompletionsGenerator, TitoModelClient

            kwargs = (model_cfg.extra_body or {}).get("chat_template_kwargs")
            chat = ChatTokenizer.from_pretrained(tito, chat_template_kwargs=kwargs)
            gen = CompletionsGenerator(
                model=model_cfg.model,
                base_url=model_cfg.base_url or "http://localhost:8000/v1",
                api_key=model_cfg.api_key or "EMPTY",
            )
            model_client = TitoModelClient(
                chat,
                gen,
                model=model_cfg.model,
                max_tokens=model_cfg.max_tokens or 1024,
                temperature=model_cfg.temperature if model_cfg.temperature is not None else 1.0,
            )

    config = RolloutConfig(
        sandbox=sandbox,
        model=model_cfg,
        agent=agent,
        rewards=_reward_specs(reward) or proj.rewards,
        seed=seed,
    )
    out = out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run.json"
    fingerprint = {
        "model": model_cfg.model_dump(mode="json", exclude={"api_key"}),
        "agent": agent.model_dump(mode="json", exclude={"custom_tools"}),
        "sandbox": sandbox.model_dump(mode="json"),
        "rewards": [r.model_dump() for r in config.rewards] if config.rewards else None,
        "seed": seed,
        "tito": tito,
    }
    # n is deliberately not fingerprinted: resuming with a larger n adds samples.
    config_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("config_hash") != config_hash and not force:
            fail(
                f"config changed since this run started ({previous.get('config_hash')} != "
                f"{config_hash}); pass --force to resume anyway",
                code=EXIT_USAGE,
            )
    manifest = {
        "agentbox": __version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "tasks": [str(p) for p in tasks],
        "n_tasks": len(task_list),
        "n": n,
        **fingerprint,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    traj_path = out / ("trajectories.jsonl.gz" if compress else "trajectories.jsonl")
    runner = ParallelRunner(
        concurrency=concurrency,
        config=config,
        model=model_client,
        max_retries=retries,
        model_concurrency=model_concurrency,
    )
    t0 = time.monotonic()
    summary = asyncio.run(
        runner.run_to_sink(
            task_list,
            JsonlSink(traj_path, append=resume),
            n_per_task=n,
            resume=resume,
            progress=progress and not st.json and not st.quiet,
        )
    )
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    m = summary.metrics

    def human() -> None:
        typer.echo(f"run {summary.run_id[:8]} → {traj_path}")
        table(
            ["rollouts", "skipped", "success", "mean reward", "mean steps", "p90 dur (s)", "infra", "rollouts/min"],
            [[summary.n, summary.skipped, fmt_pct(m.success_rate), f"{m.mean_reward:.3f}",
              f"{m.mean_steps:.1f}", f"{m.p90_duration_s:.1f}", m.infra_failures,
              f"{summary.rollouts_per_min:.1f}"]],
        )
        if m.by_failure_kind:
            typer.echo(f"failure kinds: {m.by_failure_kind}")
        typer.echo(f"wall clock {time.monotonic() - t0:.1f}s")

    emit(st, {"out": str(out), "trajectories": str(traj_path), **summary.model_dump(mode="json")}, human)
    if min_success_rate is not None and m.success_rate < min_success_rate:
        raise typer.Exit(EXIT_FAILURES)
    raise typer.Exit(EXIT_OK)
