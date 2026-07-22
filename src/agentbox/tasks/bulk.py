"""Bulk task dataset helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from agentbox.tasks.loader import load_task
from agentbox.tasks.schema import Task


def save_task_dataset(tasks: Sequence[Task], directory: str | Path) -> list[Path]:
    """Write each task as ``{task_id}.json`` under directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for task in tasks:
        path = directory / f"{task.task_id}.json"
        task.save_json(path)
        paths.append(path)
    return paths


def load_task_dataset(directory: str | Path) -> list[Task]:
    """Load all task.json files and task directories under a path."""
    directory = Path(directory)
    tasks: list[Task] = []
    if not directory.exists():
        return tasks
    for path in sorted(directory.rglob("task.json")):
        tasks.append(load_task(path.parent))
    for path in sorted(directory.glob("*.json")):
        if path.name == "task.json":
            continue
        try:
            tasks.append(Task.from_json(path))
        except Exception:
            continue
    # de-dupe by task_id preserving order
    seen: set[str] = set()
    unique: list[Task] = []
    for t in tasks:
        if t.task_id in seen:
            continue
        seen.add(t.task_id)
        unique.append(t)
    return unique


def export_tasks_jsonl(tasks: Iterable[Task], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json())
            f.write("\n")


def import_tasks_jsonl(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tasks.append(Task.model_validate(json.loads(line)))
    return tasks
