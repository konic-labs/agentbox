"""Task loading helpers."""

from __future__ import annotations

from pathlib import Path

from agentbox.tasks.schema import Task


def load_task(path: str | Path) -> Task:
    path = Path(path)
    if path.is_dir() or (path.parent / "files").exists():
        if path.is_dir():
            return Task.from_dir(path)
    return Task.from_json(path)
