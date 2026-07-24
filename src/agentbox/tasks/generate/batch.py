"""Batch task generation with resume, concurrency, and multi-stage QC."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from agentbox.tasks.generate.constraints import constraint_for_index, load_constraint_families
from agentbox.tasks.generate.dedup import is_near_duplicate, task_signature_hash
from agentbox.tasks.generate.generator import GenerateConfig, TaskGenerator
from agentbox.tasks.generate.review import export_review_queue, queue_item_from_task
from agentbox.tasks.generate.static_qc import validate_task_static
from agentbox.tasks.generate.validate import validate_task_live
from agentbox.tasks.schema import Task

logger = logging.getLogger("agentbox.generate.batch")


class BatchGenerateConfig(BaseModel):
    target: int = 20
    out_dir: Path = Path("generated/tasks")
    state_path: Path | None = None
    gen_concurrency: int = 8
    docker_concurrency: int = 8
    oversample: float = 1.25
    max_rounds: int = 12
    max_retries_per_slot: int = 4
    enable_docker_qc: bool = True
    enable_llm_qc: bool = True
    enable_static_qc: bool = True
    min_asserts: int = 3
    constraints_file: Path | None = None
    domain: str = "python"
    two_stage: bool = False
    dedup: bool = True
    jaccard_threshold: float = 0.85
    review_queue_path: Path | None = None
    strict_paths: bool = False


class BatchState(BaseModel):
    done_ids: list[str] = Field(default_factory=list)
    attempts: int = 0
    next_i: int = 0
    failures: int = 0


def _load_state(path: Path) -> BatchState:
    if path.exists():
        return BatchState.model_validate(json.loads(path.read_text()))
    return BatchState()


def _save_state(path: Path, state: BatchState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _scan_done(out_dir: Path) -> set[str]:
    done: set[str] = set()
    if not out_dir.exists():
        return done
    for p in out_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            if "task_id" in data:
                done.add(data["task_id"])
        except Exception:
            pass
    return done


async def batch_generate(
    gen_config: GenerateConfig,
    batch: BatchGenerateConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[Task]:
    """Generate until ``batch.target`` QC-passed tasks exist on disk."""
    log = progress or (lambda m: logger.info("%s", m))
    out = batch.out_dir
    out.mkdir(parents=True, exist_ok=True)
    state_path = batch.state_path or (out.parent / "state.json")
    state = _load_state(state_path)
    done = _scan_done(out) | set(state.done_ids)
    state.done_ids = sorted(done)
    _save_state(state_path, state)

    # Draft without embedded QC; we orchestrate stages for concurrency control
    draft_cfg = gen_config.model_copy(
        update={
            "validate_in_docker": False,
            "validate_with_llm": False,
            "validate_static": False,
            "two_stage": batch.two_stage or gen_config.two_stage,
            "static_strict_paths": batch.strict_paths,
        }
    )
    gen = TaskGenerator(draft_cfg)
    families = load_constraint_families(batch.constraints_file)
    gen_sem = asyncio.Semaphore(batch.gen_concurrency)
    docker_sem = asyncio.Semaphore(batch.docker_concurrency)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    accepted: list[Task] = []
    review_items: list = []

    # load existing accepted tasks
    for p in sorted(out.glob("*.json")):
        try:
            accepted.append(Task.model_validate(json.loads(p.read_text())))
        except Exception:
            pass

    if len(done) >= batch.target:
        log(f"already have {len(done)}/{batch.target}")
        return accepted[: batch.target]

    async def one_slot(index: int) -> Task | None:
        if stop.is_set():
            return None
        difficulty, constraints = constraint_for_index(index, families=families)
        last_err = ""
        for attempt in range(batch.max_retries_per_slot):
            if stop.is_set():
                return None
            async with lock:
                if len(done) >= batch.target:
                    stop.set()
                    return None
            try:
                t0 = time.perf_counter()
                async with gen_sem:
                    task = await gen.generate(
                        difficulty=difficulty,  # type: ignore[arg-type]
                        domain=batch.domain,
                        constraints=constraints,
                    )
                gen_s = time.perf_counter() - t0

                if batch.enable_static_qc:
                    sq = validate_task_static(
                        task,
                        min_asserts=batch.min_asserts,
                        strict_paths=batch.strict_paths,
                    )
                    if not sq.ok:
                        last_err = "; ".join(sq.errors)
                        log(f"  [{index:02d}] static fail: {last_err[:120]}")
                        async with lock:
                            state.attempts += 1
                            state.failures += 1
                            _save_state(state_path, state)
                            review_items.append(
                                queue_item_from_task(
                                    task, errors=sq.errors, warnings=sq.warnings
                                )
                            )
                        continue

                if batch.dedup:
                    async with lock:
                        dup, reason = is_near_duplicate(
                            task,
                            accepted,
                            jaccard_threshold=batch.jaccard_threshold,
                        )
                    if dup:
                        last_err = f"dedup: {reason}"
                        log(f"  [{index:02d}] {last_err[:120]}")
                        async with lock:
                            state.attempts += 1
                            state.failures += 1
                            _save_state(state_path, state)
                        continue

                solution_files = None
                if task.metadata.get("solution_files"):
                    solution_files = {
                        str(k): str(v)
                        for k, v in task.metadata["solution_files"].items()
                    }

                if batch.enable_docker_qc:
                    async with docker_sem:
                        drep = await validate_task_live(
                            task,
                            sandbox_config=gen.config.sandbox,
                            expect_fail_on_starter=True,
                            solution_files=solution_files,
                        )
                    if not drep.ok:
                        last_err = "; ".join(drep.errors)
                        log(f"  [{index:02d}] docker fail: {last_err[:120]}")
                        async with lock:
                            state.attempts += 1
                            state.failures += 1
                            _save_state(state_path, state)
                            review_items.append(
                                queue_item_from_task(task, errors=drep.errors)
                            )
                        continue

                if batch.enable_llm_qc:
                    async with gen_sem:
                        jrep = await gen.validate_task_llm_only(task, domain=batch.domain)
                    if not jrep.ok:
                        last_err = "; ".join(jrep.errors)
                        log(f"  [{index:02d}] llm fail: {last_err[:120]}")
                        async with lock:
                            state.attempts += 1
                            state.failures += 1
                            _save_state(state_path, state)
                            review_items.append(
                                queue_item_from_task(
                                    task,
                                    score=jrep.llm_score,
                                    reasons=jrep.llm_reasons,
                                    errors=jrep.errors,
                                )
                            )
                        continue
                    task.metadata["llm_judge_score"] = jrep.llm_score
                    task.metadata["llm_judge_accept"] = jrep.llm_accept

                # Drop golden solution from disk unless kept
                if solution_files and not gen_config.keep_solution_in_metadata:
                    task.metadata.pop("solution_files", None)
                    task.metadata.pop("_solution_ephemeral", None)
                task.metadata.setdefault("signature_hash", task_signature_hash(task))

                async with lock:
                    if len(done) >= batch.target:
                        stop.set()
                        return None
                    tid = task.task_id
                    if tid in done:
                        tid = f"{task.task_id}_{uuid.uuid4().hex[:6]}"
                        task = task.model_copy(update={"task_id": tid})
                    path = out / f"{tid}.json"
                    task.save_json(path)
                    done.add(tid)
                    accepted.append(task)
                    state.done_ids = sorted(done)
                    state.attempts += 1
                    state.next_i = max(state.next_i, index + 1)
                    _save_state(state_path, state)
                    review_items.append(
                        queue_item_from_task(
                            task,
                            path=str(path),
                            score=task.metadata.get("llm_judge_score"),
                        )
                    )
                    n = len(done)
                    if n >= batch.target:
                        stop.set()
                log(
                    f"  OK [{len(done)}/{batch.target}] i={index} {task.task_id} "
                    f"({gen_s:.1f}s gen) files={list(task.starter_files.keys())}"
                )
                return task
            except Exception as exc:
                last_err = str(exc)
                log(f"  [{index:02d}] error attempt {attempt+1}: {last_err[:160]}")
                async with lock:
                    state.attempts += 1
                    state.failures += 1
                    _save_state(state_path, state)
                await asyncio.sleep(min(2 ** attempt, 8))
        log(f"  FAIL i={index}: {last_err[:160]}")
        return None

    next_i = state.next_i
    for round_n in range(1, batch.max_rounds + 1):
        remaining = batch.target - len(done)
        if remaining <= 0 or stop.is_set():
            break
        n_slots = max(
            remaining,
            min(int(remaining * batch.oversample + 0.999), remaining + 12),
        )
        indices = list(range(next_i, next_i + n_slots))
        next_i = indices[-1] + 1
        state.next_i = next_i
        _save_state(state_path, state)
        log(
            f"=== round {round_n}: need {remaining}, slots {n_slots} "
            f"(i={indices[0]}..{indices[-1]}) ==="
        )
        await asyncio.gather(*[one_slot(i) for i in indices])

    rq = batch.review_queue_path or (out.parent / "review_queue.jsonl")
    try:
        export_review_queue(review_items, rq)
        log(f"review_queue → {rq} ({len(review_items)} rows)")
    except Exception as exc:
        log(f"review_queue write failed: {exc}")

    log(f"Done: {len(done)}/{batch.target} in {out}")
    return accepted[: batch.target]
