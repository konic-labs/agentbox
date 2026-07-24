"""Light human-review queue export/import for generated tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from agentbox.tasks.schema import Task


class ReviewQueueItem(BaseModel):
    task_id: str
    path: str | None = None
    score: float | None = None
    reasons: str | None = None
    accept_hint: bool | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewDecision(BaseModel):
    task_id: str
    decision: Literal["accept", "reject"]
    note: str | None = None


def export_review_queue(
    items: Iterable[ReviewQueueItem | dict[str, Any]],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, ReviewQueueItem):
                row = item.model_dump(mode="json")
            else:
                row = dict(item)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_review_decisions(path: Path) -> dict[str, ReviewDecision]:
    path = Path(path)
    out: dict[str, ReviewDecision] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = ReviewDecision.model_validate(json.loads(line))
        out[d.task_id] = d
    return out


def filter_tasks_by_decisions(
    tasks: Iterable[Task],
    decisions: dict[str, ReviewDecision],
    *,
    default: Literal["accept", "reject"] = "accept",
) -> list[Task]:
    kept: list[Task] = []
    for t in tasks:
        dec = decisions.get(t.task_id)
        if dec is None:
            if default == "accept":
                kept.append(t)
            continue
        if dec.decision == "accept":
            kept.append(t)
    return kept


def queue_item_from_task(
    task: Task,
    *,
    path: str | None = None,
    score: float | None = None,
    reasons: str | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ReviewQueueItem:
    return ReviewQueueItem(
        task_id=task.task_id,
        path=path,
        score=score if score is not None else task.metadata.get("llm_judge_score"),
        reasons=reasons or task.metadata.get("llm_judge_reasons"),
        accept_hint=task.metadata.get("llm_judge_accept"),
        errors=list(errors or []),
        warnings=list(warnings or []),
        files=sorted((task.starter_files or {}).keys()),
        metadata={
            k: v
            for k, v in (task.metadata or {}).items()
            if k
            in {
                "difficulty",
                "tags",
                "language",
                "generator_model",
                "heuristic_label",
                "signature_hash",
            }
        },
    )
