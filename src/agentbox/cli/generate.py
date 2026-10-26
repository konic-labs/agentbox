"""`agentbox generate …`: generate and QC coding tasks with a teacher model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer

from agentbox.cli._common import EXIT_FAILURES, EXIT_OK, State, emit, fail, state

generate_app = typer.Typer(
    help="Generate and QC coding tasks via OpenAI-compatible teachers.", no_args_is_help=True
)


def _teacher(
    st: State,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int | None,
    use_dspy: bool | None,
    llm_judge: bool | None,
    min_score: float | None,
) -> tuple[str, str | None, str | None, int, bool, bool, float]:
    """Flags first, then ``teacher`` / ``generate`` sections (config errors are fatal)."""
    cfg = st.project
    t, g = cfg.teacher, cfg.generate
    model = model or (t.model if t else None)
    if not model:
        fail("provide --model or set teacher.model in agentbox.yaml")
    return (
        model,
        base_url if base_url is not None else (t.base_url if t else None),
        api_key if api_key is not None else (t.api_key if t else None),
        int(
            max_tokens
            if max_tokens is not None
            else ((t.max_tokens if t else None) or g.max_tokens)
        ),
        bool(use_dspy if use_dspy is not None else g.use_dspy),
        bool(llm_judge if llm_judge is not None else g.llm_judge),
        float(min_score if min_score is not None else g.min_score),
    )


def _gen_config(
    model: str,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    use_dspy: bool,
    llm_judge: bool,
    docker_qc: bool,
    min_score: float,
    *,
    two_stage: bool = False,
) -> Any:
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
    ctx: typer.Context,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", envvar="OPENAI_API_KEY")] = None,
    difficulty: Annotated[str, typer.Option("--difficulty")] = "easy",
    domain: Annotated[str, typer.Option("--domain")] = "python",
    constraints: Annotated[str, typer.Option("--constraints")] = "",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("generated/tasks"),
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    use_dspy: Annotated[bool | None, typer.Option("--dspy/--no-dspy")] = None,
    docker_qc: Annotated[bool, typer.Option("--docker-qc/--no-docker-qc")] = True,
    llm_judge: Annotated[bool | None, typer.Option("--llm-judge/--no-llm-judge")] = None,
    min_score: Annotated[float | None, typer.Option("--min-score")] = None,
    static_qc: Annotated[bool, typer.Option("--static-qc/--no-static-qc")] = True,
    two_stage: Annotated[bool, typer.Option("--two-stage/--no-two-stage")] = False,
) -> None:
    """Generate a single QC-passed coding task."""
    from agentbox.tasks.generate import TaskGenerator
    from agentbox.tasks.generate.static_qc import validate_task_static

    st = state(ctx)
    m, b, k, mt, dspy, judge, score = _teacher(
        st, model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score
    )
    gen = TaskGenerator(
        _gen_config(m, b, k, mt, dspy, judge, docker_qc, score, two_stage=two_stage)
    )

    async def go() -> Any:
        try:
            task = await gen.generate(
                difficulty=cast(Literal["easy", "medium", "hard"], difficulty),
                domain=domain,
                constraints=constraints or None,
            )
        finally:
            await gen.aclose()
        if static_qc:
            rep = validate_task_static(task)
            if not rep.ok:
                fail("static QC failed: " + "; ".join(rep.errors), code=EXIT_FAILURES)
        return task

    task = asyncio.run(go())
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{task.task_id}.json"
    task.save_json(path)
    emit(
        st,
        {
            "path": str(path),
            "files": sorted(task.starter_files),
            "llm_judge_score": task.metadata.get("llm_judge_score"),
        },
        lambda: typer.echo(f"wrote {path} files={sorted(task.starter_files)}"),
    )


@generate_app.command("batch")
def generate_batch(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", envvar="OPENAI_API_KEY")] = None,
    target: Annotated[int | None, typer.Option("--target", "-n")] = None,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("generated/tasks"),
    concurrency: Annotated[int | None, typer.Option("--concurrency", "-c")] = None,
    docker_concurrency: Annotated[int | None, typer.Option("--docker-concurrency")] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    use_dspy: Annotated[bool | None, typer.Option("--dspy/--no-dspy")] = None,
    docker_qc: Annotated[bool, typer.Option("--docker-qc/--no-docker-qc")] = True,
    llm_judge: Annotated[bool | None, typer.Option("--llm-judge/--no-llm-judge")] = None,
    static_qc: Annotated[bool, typer.Option("--static-qc/--no-static-qc")] = True,
    min_score: Annotated[float | None, typer.Option("--min-score")] = None,
    constraints_file: Annotated[Path | None, typer.Option("--constraints-file")] = None,
    two_stage: Annotated[bool, typer.Option("--two-stage/--no-two-stage")] = False,
    dedup: Annotated[bool, typer.Option("--dedup/--no-dedup")] = True,
    strict_paths: Annotated[bool, typer.Option("--strict-paths")] = False,
) -> None:
    """Generate N QC-passed tasks with resume-safe concurrency."""
    from agentbox.tasks.generate.batch import BatchGenerateConfig, batch_generate

    st = state(ctx)
    g = st.project.generate
    m, b, k, mt, dspy, judge, score = _teacher(
        st, model, base_url, api_key, max_tokens, use_dspy, llm_judge, min_score
    )
    gen_cfg = _gen_config(m, b, k, mt, dspy, False, False, score, two_stage=two_stage)
    n = int(target if target is not None else g.target)
    batch = BatchGenerateConfig(
        target=n,
        out_dir=out,
        gen_concurrency=int(concurrency if concurrency is not None else g.concurrency),
        docker_concurrency=int(
            docker_concurrency if docker_concurrency is not None else g.docker_concurrency
        ),
        enable_docker_qc=docker_qc,
        enable_llm_qc=judge,
        enable_static_qc=static_qc,
        constraints_file=constraints_file,
        two_stage=two_stage,
        dedup=dedup,
        strict_paths=strict_paths,
    )
    log = (lambda msg: None) if st.json else (lambda msg: typer.echo(msg))
    tasks = asyncio.run(batch_generate(gen_cfg, batch, progress=log))
    emit(
        st,
        {"generated": len(tasks), "target": n, "out": str(out)},
        lambda: typer.echo(f"generated {len(tasks)}/{n} → {out}"),
    )
    raise typer.Exit(EXIT_OK if len(tasks) >= n else EXIT_FAILURES)


@generate_app.command("validate-llm")
def generate_validate_llm(
    ctx: typer.Context,
    tasks_dir: Annotated[Path, typer.Argument(help="Directory of task JSON files")],
    model: Annotated[str, typer.Option("--model", "-m")],
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", envvar="OPENAI_API_KEY")] = None,
    min_score: Annotated[float, typer.Option("--min-score")] = 0.65,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("generated/llm_audit.jsonl"),
    use_dspy: Annotated[bool, typer.Option("--dspy/--no-dspy")] = True,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 8,
) -> None:
    """Re-audit tasks with the LLM judge only (concurrent)."""
    from agentbox.tasks.generate import TaskGenerator
    from agentbox.tasks.loader import iter_tasks

    st = state(ctx)
    tasks = list(iter_tasks(tasks_dir))
    if not tasks:
        fail(f"no tasks in {tasks_dir}")
    gen = TaskGenerator(
        _gen_config(model, base_url, api_key, 2048, use_dspy, True, False, min_score)
    )

    async def go() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)

        async def one(t: Any) -> dict[str, Any]:
            async with sem:
                rep = await gen.validate_task_llm_only(t)
                return {
                    "task_id": t.task_id,
                    "ok": rep.ok,
                    "llm_score": rep.llm_score,
                    "errors": rep.errors,
                    "reasons": rep.llm_reasons,
                }

        try:
            return await asyncio.gather(*(one(t) for t in tasks))
        finally:
            await gen.aclose()

    rows = asyncio.run(go())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rejected = sum(1 for r in rows if not r["ok"])
    emit(
        st,
        {"accepted": len(rows) - rejected, "rejected": rejected, "out": str(out)},
        lambda: typer.echo(f"accept={len(rows) - rejected} reject={rejected} → {out}"),
    )
    raise typer.Exit(EXIT_FAILURES if rejected else EXIT_OK)


@generate_app.command("validate-docker")
def generate_validate_docker(
    ctx: typer.Context,
    tasks_dir: Annotated[Path, typer.Argument()],
    network: Annotated[bool, typer.Option("--network/--no-network")] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 4,
) -> None:
    """Docker fail-on-starter QC for all tasks in a directory (concurrent)."""
    from agentbox.cli.task import task_check

    ctx.invoke(task_check, ctx=ctx, sources=[tasks_dir], solution=None, concurrency=concurrency)
