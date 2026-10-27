"""Task loading helpers (single tasks and lazy datasets)."""

from __future__ import annotations

import gzip
import io
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from agentbox.errors import TaskValidationError
from agentbox.tasks.schema import Task

logger = logging.getLogger("agentbox.tasks")


def load_task(path: str | Path) -> Task:
    """Load a task from a directory, a ``task.json`` (with optional sibling ``files/``) or JSON."""
    path = Path(path)
    if path.is_dir() or (path.name == "task.json" and (path.parent / "files").is_dir()):
        return Task.from_dir(path)
    return Task.from_json(path)


def _open(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: Path, strict: bool) -> Iterator[Task]:
    with _open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Task.model_validate(json.loads(line))
            except Exception as exc:
                if strict:
                    raise TaskValidationError(f"{path}:{lineno}: {exc}") from exc
                logger.warning("skipping invalid task %s:%d: %s", path, lineno, exc)


def _iter_parquet(path: Path, strict: bool) -> Iterator[Task]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TaskValidationError("reading parquet needs pyarrow (pip install pyarrow)") from exc
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            data = row.get("task") if isinstance(row.get("task"), (str, dict)) else row
            if isinstance(data, str):
                data = json.loads(data)
            try:
                yield Task.model_validate(data)
            except Exception as exc:
                if strict:
                    raise TaskValidationError(f"{path}: {exc}") from exc
                logger.warning("skipping invalid task row in %s: %s", path, exc)


def iter_tasks(
    source: str | Path | Iterable[str | Path],
    *,
    strict: bool = False,
    dedupe: bool = True,
    split: str | None = None,
) -> Iterator[Task]:
    """Lazily yield tasks from dirs, ``*.json``, ``*.jsonl(.gz)`` or ``*.parquet``.

    Directories are scanned for task directories (``task.json`` + ``files/``)
    and loose task JSON files. Invalid entries raise with ``strict=True`` and
    are logged otherwise; duplicate ``task_id`` values are reported. ``split``
    keeps only tasks whose ``split`` field matches.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    seen: set[str] = set()

    def emit(task: Task) -> Iterator[Task]:
        if split is not None and task.split != split:
            return
        if dedupe and task.task_id in seen:
            logger.warning("duplicate task_id %r skipped", task.task_id)
            return
        seen.add(task.task_id)
        yield task

    for src in sources:
        path = Path(src)
        if not path.exists():
            raise TaskValidationError(f"task source not found: {path}")
        if path.is_file():
            if path.name.endswith((".jsonl", ".jsonl.gz")):
                for t in _iter_jsonl(path, strict):
                    yield from emit(t)
            elif path.suffix == ".parquet":
                for t in _iter_parquet(path, strict):
                    yield from emit(t)
            else:
                yield from emit(load_task(path))
            continue
        if (path / "task.json").exists():
            yield from emit(Task.from_dir(path))
            continue
        task_dirs = {p.parent for p in path.rglob("task.json")}
        for d in sorted(task_dirs):
            try:
                yield from emit(Task.from_dir(d))
            except TaskValidationError as exc:
                if strict:
                    raise
                logger.warning("skipping %s: %s", d, exc)
        for p in sorted(path.rglob("*")):
            if not p.is_file() or any(parent in task_dirs for parent in p.parents):
                continue
            if p.name.endswith((".jsonl", ".jsonl.gz")):
                for t in _iter_jsonl(p, strict):
                    yield from emit(t)
            elif p.suffix == ".json" and p.name not in {"suite.json", "state.json"}:
                try:
                    yield from emit(Task.from_json(p))
                except TaskValidationError as exc:
                    if strict:
                        raise
                    logger.debug("not a task: %s (%s)", p, exc)
