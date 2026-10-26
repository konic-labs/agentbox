"""Bulk task dataset helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from agentbox.tasks.loader import iter_tasks
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


def load_task_dataset(directory: str | Path, *, strict: bool = False) -> list[Task]:
    """Load all tasks under a path (see :func:`iter_tasks` for formats)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return list(iter_tasks(directory, strict=strict))


def export_tasks_jsonl(tasks: Iterable[Task], path: str | Path) -> None:
    import gzip
    import io

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.open(path, "wb") if path.suffix == ".gz" else path.open("wb")
    with io.TextIOWrapper(raw, encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json(exclude_defaults=True))
            f.write("\n")


def iter_tasks_jsonl(path: str | Path) -> Iterator[Task]:
    yield from iter_tasks(path)


def import_tasks_jsonl(path: str | Path) -> list[Task]:
    return list(iter_tasks(path))
