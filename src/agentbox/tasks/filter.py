"""Task filtering and curriculum helpers."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from agentbox.tasks.schema import Task

if TYPE_CHECKING:
    from agentbox.trajectory.schema import Trajectory

Difficulty = Literal["easy", "medium", "hard"]


def filter_tasks(
    tasks: Sequence[Task],
    *,
    difficulty: Difficulty | Sequence[Difficulty] | None = None,
    tags: Sequence[str] | None = None,
    require_all_tags: bool = False,
    domain: str | None = None,
    language: str | None = None,
    min_estimated_steps: int | None = None,
    max_estimated_steps: int | None = None,
    predicate: Any | None = None,
) -> list[Task]:
    """Filter tasks by metadata fields."""
    diffs: set[str] | None = None
    if difficulty is not None:
        if isinstance(difficulty, str):
            diffs = {difficulty}
        else:
            diffs = set(difficulty)

    tag_set = set(tags) if tags else None
    out: list[Task] = []
    for task in tasks:
        meta = task.metadata or {}
        if diffs is not None and meta.get("difficulty") not in diffs:
            continue
        if domain is not None and meta.get("domain") != domain:
            continue
        if language is not None and meta.get("language") != language:
            continue
        est = meta.get("estimated_steps")
        if min_estimated_steps is not None and (est is None or int(est) < min_estimated_steps):
            continue
        if max_estimated_steps is not None and (est is None or int(est) > max_estimated_steps):
            continue
        if tag_set is not None:
            task_tags = set(meta.get("tags") or [])
            if require_all_tags:
                if not tag_set.issubset(task_tags):
                    continue
            elif not tag_set.intersection(task_tags):
                continue
        if predicate is not None and not predicate(task):
            continue
        out.append(task)
    return out


def sample_curriculum(
    tasks: Sequence[Task],
    *,
    n: int,
    order: Sequence[Difficulty] = ("easy", "medium", "hard"),
    rng: random.Random | None = None,
) -> list[Task]:
    """Sample up to n tasks, preferring easier difficulties first (curriculum)."""
    rng = rng or random.Random()
    remaining = n
    selected: list[Task] = []
    for diff in order:
        if remaining <= 0:
            break
        pool = filter_tasks(tasks, difficulty=diff)
        rng.shuffle(pool)
        take = pool[:remaining]
        selected.extend(take)
        remaining -= len(take)
    if remaining > 0:
        chosen = {id(t) for t in selected}  # identity, not deep pydantic equality (O(n))
        leftover = [t for t in tasks if id(t) not in chosen]
        rng.shuffle(leftover)
        selected.extend(leftover[:remaining])
    return selected


def group_by_difficulty(tasks: Iterable[Task]) -> dict[str, list[Task]]:
    groups: dict[str, list[Task]] = {"easy": [], "medium": [], "hard": [], "unknown": []}
    for task in tasks:
        d = (task.metadata or {}).get("difficulty") or "unknown"
        if d not in groups:
            groups[str(d)] = []
        groups[str(d)].append(task)
    return groups


def pass_rates(trajectories: Iterable[Trajectory]) -> dict[str, tuple[int, int]]:
    """``task_id -> (successes, samples)`` over trainable rollouts (infra failures ignored)."""
    out: dict[str, tuple[int, int]] = {}
    for t in trajectories:
        if not t.is_trainable:
            continue
        ok, n = out.get(t.task_id, (0, 0))
        out[t.task_id] = (ok + int(t.success), n + 1)
    return out


def filter_by_pass_rate(
    tasks: Iterable[Task],
    rates: dict[str, tuple[int, int]],
    *,
    min_rate: float = 0.0,
    max_rate: float = 1.0,
    min_samples: int = 1,
    keep_unseen: bool = True,
) -> list[Task]:
    """Keep tasks whose observed pass rate lies in ``[min_rate, max_rate]``.

    The usual GRPO selection is the learnable band, e.g. ``min_rate=0.01,
    max_rate=0.99``: tasks the policy always or never solves give zero-variance
    groups. Tasks with fewer than ``min_samples`` observations are kept when
    ``keep_unseen``.
    """
    selected: list[Task] = []
    for task in tasks:
        ok, n = rates.get(task.task_id, (0, 0))
        if n < min_samples:
            if keep_unseen:
                selected.append(task)
            continue
        if min_rate <= ok / n <= max_rate:
            selected.append(task)
    return selected
