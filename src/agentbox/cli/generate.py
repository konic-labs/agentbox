"""CLI: agentbox generate …"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

generate_app = typer.Typer(
    name="generate",
    help="Generate and QC coding tasks via OpenAI-compatible teachers.",
    no_args_is_help=True,
)


def _apply_project_defaults(
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    use_dspy: Optional[bool],
    llm_judge: Optional[bool],
    min_score: Optional[float],
) -> tuple:
    """Fill missing flags from agentbox.yaml when present."""
    try:
        from agentbox.config_load import load_project_config

        cfg = load_project_config()
    except Exception:
        cfg = None
    t = cfg.teacher if cfg else None
    g = cfg.generate if cfg else None
    model = model or (t.model if t else None)
    base_url = base_url if base_url is not None else (t.base_url if t else None)
    api_key = api_key if api_key is not None else (t.api_key if t else None)
    max_tokens = max_tokens if max_tokens is not None else (
        (t.max_tokens if t and t.max_tokens else None) or (g.max_tokens if g else 8192)
    )
    use_dspy = use_dspy if use_dspy is not None else (g.use_dspy if g else False)
    llm_judge = llm_judge if llm_judge is not None else (g.llm_judge if g else True)
    min_score = min_score if min_score is not None else (g.min_score if g else 0.65)
    return model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score


def _gen_config(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: int,
    use_dspy: bool,
    llm_judge: bool,
    docker_qc: bool,
    min_score: float,
    *,
    two_stage: bool = False,
):
    from agentbox.tasks.generate import GenerateConfig

    return GenerateConfig(
        model=model,
        base_url=base_url,
        api_key=api_key or "EMPTY",
        max_tokens=max_tokens,
        temperature=0.6,
        validate_in_docker=docker_qc,
        validate_with_llm=llm_judge,
        llm_judge_min_score=min_score,
        use_dspy=use_dspy,
        disable_thinking=True,
        max_retries=2,
        two_stage=two_stage,
    )


@generate_app.command("one")
def generate_one(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    difficulty: str = typer.Option("easy", "--difficulty"),
    domain: str = typer.Option("python", "--domain"),
    constraints: str = typer.Option("", "--constraints"),
    out: Path = typer.Option(Path("generated/tasks"), "--out", "-o"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
    use_dspy: bool = typer.Option(False, "--dspy/--no-dspy"),
    docker_qc: bool = typer.Option(True, "--docker-qc/--no-docker-qc"),
    llm_judge: bool = typer.Option(True, "--llm-judge/--no-llm-judge"),
    min_score: Optional[float] = typer.Option(None, "--min-score"),
    static_qc: bool = typer.Option(True, "--static-qc/--no-static-qc"),
    two_stage: bool = typer.Option(
        False, "--two-stage/--no-two-stage", help="Golden solution + AST stub strip"
    ),
) -> None:
    """Generate a single QC-passed coding task."""
    from agentbox.tasks.generate import TaskGenerator
    from agentbox.tasks.generate.static_qc import validate_task_static

    model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score = (
        _apply_project_defaults(
            model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score
        )
    )
    if not model:
        typer.secho("Provide --model or set teacher.model in agentbox.yaml", fg=typer.colors.RED)
        raise typer.Exit(2)

    gen = TaskGenerator(
        _gen_config(
            model,
            base_url,
            api_key,
            int(max_tokens or 8192),
            bool(use_dspy),
            bool(llm_judge),
            docker_qc,
            float(min_score or 0.65),
            two_stage=two_stage,
        )
    )

    async def _run():
        task = await gen.generate(
            difficulty=difficulty,  # type: ignore[arg-type]
            domain=domain,
            constraints=constraints or None,
        )
        if static_qc:
            rep = validate_task_static(task)
            if not rep.ok:
                raise RuntimeError("static QC failed: " + "; ".join(rep.errors))
        return task

    task = asyncio.run(_run())
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{task.task_id}.json"
    task.save_json(path)
    typer.echo(f"wrote {path} files={list(task.starter_files.keys())}")
    if task.metadata.get("llm_judge_score") is not None:
        typer.echo(f"llm_judge_score={task.metadata['llm_judge_score']}")


@generate_app.command("batch")
def generate_batch(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    target: Optional[int] = typer.Option(None, "--target", "-n"),
    out: Path = typer.Option(Path("generated/tasks"), "--out", "-o"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c"),
    docker_concurrency: Optional[int] = typer.Option(None, "--docker-concurrency"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
    use_dspy: bool = typer.Option(False, "--dspy/--no-dspy"),
    docker_qc: bool = typer.Option(True, "--docker-qc/--no-docker-qc"),
    llm_judge: bool = typer.Option(True, "--llm-judge/--no-llm-judge"),
    static_qc: bool = typer.Option(True, "--static-qc/--no-static-qc"),
    min_score: Optional[float] = typer.Option(None, "--min-score"),
    constraints_file: Optional[Path] = typer.Option(None, "--constraints-file"),
    two_stage: bool = typer.Option(False, "--two-stage/--no-two-stage"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup"),
    strict_paths: bool = typer.Option(False, "--strict-paths"),
) -> None:
    """Generate N QC-passed tasks with resume-safe concurrency."""
    from agentbox.tasks.generate.batch import BatchGenerateConfig, batch_generate

    try:
        from agentbox.config_load import load_project_config

        pcfg = load_project_config()
    except Exception:
        pcfg = None
    g = pcfg.generate if pcfg else None

    model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score = (
        _apply_project_defaults(
            model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score
        )
    )
    if not model:
        typer.secho("Provide --model or set teacher.model in agentbox.yaml", fg=typer.colors.RED)
        raise typer.Exit(2)
    target = target if target is not None else (g.target if g else 20)
    concurrency = concurrency if concurrency is not None else (g.concurrency if g else 8)
    docker_concurrency = (
        docker_concurrency
        if docker_concurrency is not None
        else (g.docker_concurrency if g else 8)
    )

    gen_cfg = _gen_config(
        model,
        base_url,
        api_key,
        int(max_tokens or 8192),
        bool(use_dspy),
        False,
        False,
        float(min_score or 0.65),
        two_stage=two_stage,
    )
    gen_cfg.llm_judge_min_score = float(min_score or 0.65)
    batch = BatchGenerateConfig(
        target=int(target),
        out_dir=out,
        gen_concurrency=int(concurrency),
        docker_concurrency=int(docker_concurrency),
        enable_docker_qc=docker_qc,
        enable_llm_qc=llm_judge,
        enable_static_qc=static_qc,
        constraints_file=constraints_file,
        two_stage=two_stage,
        dedup=dedup,
        strict_paths=strict_paths,
    )
    tasks = asyncio.run(
        batch_generate(gen_cfg, batch, progress=lambda m: typer.echo(m))
    )
    typer.echo(f"generated {len(tasks)}/{target} → {out}")
    if len(tasks) < target:
        raise typer.Exit(1)


@generate_app.command("validate-llm")
def generate_validate_llm(
    tasks_dir: Path = typer.Argument(..., help="Directory of task JSON files"),
    model: str = typer.Option(..., "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    min_score: float = typer.Option(0.65, "--min-score"),
    out: Path = typer.Option(Path("generated/llm_audit.jsonl"), "--out", "-o"),
    use_dspy: bool = typer.Option(True, "--dspy/--no-dspy"),
) -> None:
    """Re-audit tasks with the LLM judge only."""
    from agentbox import Task
    from agentbox.tasks.generate import TaskGenerator

    gen = TaskGenerator(
        _gen_config(model, base_url, api_key, 2048, use_dspy, True, False, min_score)
    )
    paths = sorted(tasks_dir.glob("*.json"))
    if not paths:
        typer.secho(f"No tasks in {tasks_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0

    async def _run() -> None:
        nonlocal accepted, rejected
        with out.open("w", encoding="utf-8") as f:
            for path in paths:
                task = Task.model_validate(json.loads(path.read_text()))
                rep = await gen.validate_task_llm_only(task)
                row = {
                    "path": str(path),
                    "task_id": task.task_id,
                    "ok": rep.ok,
                    "llm_score": rep.llm_score,
                    "errors": rep.errors,
                    "reasons": rep.llm_reasons,
                }
                f.write(json.dumps(row) + "\n")
                status = "OK" if rep.ok else "REJECT"
                if rep.ok:
                    accepted += 1
                else:
                    rejected += 1
                typer.echo(f"  {status} {task.task_id} score={rep.llm_score}")

    asyncio.run(_run())
    typer.echo(f"accept={accepted} reject={rejected} → {out}")
    if rejected:
        raise typer.Exit(1)


@generate_app.command("validate-docker")
def generate_validate_docker(
    tasks_dir: Path = typer.Argument(...),
    network: bool = typer.Option(True, "--network/--no-network"),
) -> None:
    """Docker fail-on-starter QC for all tasks in a directory."""
    from agentbox import Task
    from agentbox.config import ResourceLimits, SandboxConfig
    from agentbox.tasks.generate import validate_task_live

    paths = sorted(tasks_dir.glob("*.json"))
    if not paths:
        raise typer.Exit(2)
    sandbox = SandboxConfig(
        limits=ResourceLimits(network_disabled=not network),
        ensure_pytest=True,
    )
    ok_n = fail_n = 0

    async def _run() -> None:
        nonlocal ok_n, fail_n
        for path in paths:
            task = Task.model_validate(json.loads(path.read_text()))
            rep = await validate_task_live(
                task, sandbox_config=sandbox, expect_fail_on_starter=True
            )
            if rep.ok:
                ok_n += 1
                typer.echo(f"  OK {task.task_id}")
            else:
                fail_n += 1
                typer.echo(f"  FAIL {task.task_id}: {rep.errors}")

    asyncio.run(_run())
    typer.echo(f"ok={ok_n} fail={fail_n}")
    if fail_n:
        raise typer.Exit(1)
