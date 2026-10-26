"""Streaming trajectory storage: sinks for writers, iterators for readers (GH #39, #65).

Runners hand every finished trajectory to a :class:`TrajectorySink` instead of
keeping it in memory; serialisation and file I/O happen off the event loop.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

from agentbox.trajectory.schema import Trajectory

logger = logging.getLogger("agentbox.trajectory")

_SKIP_NAMES = {"report.json", "model_report.json", "run.json", "suite.json", "task.json"}


@runtime_checkable
class TrajectorySink(Protocol):
    async def write(self, traj: Trajectory) -> None: ...

    async def close(self) -> None: ...


def _open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, mode.replace("t", "") + "b"), encoding="utf-8")
    return path.open(mode, encoding="utf-8")


class JsonlSink:
    """Append trajectories to a JSONL (optionally ``.jsonl.gz``) file.

    Appending is resume-friendly: :meth:`completed` lists the
    ``(task_id, sample_index)`` pairs already on disk.
    """

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh: IO[str] | None = _open_text(self.path, "at" if append else "wt")
        self.count = 0

    def completed(self) -> set[tuple[str, int]]:
        return completed_samples(self.path)

    def _write_line(self, line: str) -> None:
        with self._lock:
            if self._fh is None:
                raise RuntimeError("sink is closed")
            self._fh.write(line)
            self._fh.write("\n")
            self._fh.flush()
            self.count += 1

    async def write(self, traj: Trajectory) -> None:
        line = await asyncio.to_thread(traj.to_json)
        await asyncio.to_thread(self._write_line, line)

    async def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class DirectorySink:
    """One JSON file per trajectory: ``<dir>/<task_id>_<rollout_id[:8]>.json``."""

    def __init__(self, directory: str | Path, *, indent: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.indent = indent
        self.paths: dict[str, list[str]] = {}

    def path_for(self, traj: Trajectory) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in traj.task_id)
        rid = (traj.rollout_id or traj.run_id).replace("-", "")[:12]
        return self.directory / f"{safe}_{traj.sample_index}_{rid}.json"

    async def write(self, traj: Trajectory) -> None:
        path = self.path_for(traj)
        await asyncio.to_thread(traj.save, path, indent=self.indent)
        self.paths.setdefault(traj.task_id, []).append(str(path))

    async def close(self) -> None:
        return None


class MemorySink:
    """Keeps trajectories in a list (small runs and tests only)."""

    def __init__(self) -> None:
        self.trajectories: list[Trajectory] = []

    async def write(self, traj: Trajectory) -> None:
        self.trajectories.append(traj)

    async def close(self) -> None:
        return None


class CallbackSink:
    """Invoke ``fn(traj)`` (sync or async) for every trajectory."""

    def __init__(self, fn: Callable[[Trajectory], Any]) -> None:
        self.fn = fn

    async def write(self, traj: Trajectory) -> None:
        result = self.fn(traj)
        if asyncio.iscoroutine(result):
            await result

    async def close(self) -> None:
        return None


class MultiSink:
    def __init__(self, *sinks: TrajectorySink) -> None:
        self.sinks = [s for s in sinks if s is not None]

    async def write(self, traj: Trajectory) -> None:
        for s in self.sinks:
            await s.write(traj)

    async def close(self) -> None:
        for s in self.sinks:
            await s.close()


# ---------------------------------------------------------------------- readers
def _iter_jsonl(path: Path) -> Iterator[Trajectory]:
    with _open_text(path, "rt") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Trajectory.model_validate_json(line)
            except ValueError as exc:
                logger.warning("skipping invalid trajectory %s:%d: %s", path, lineno, exc)


def _iter_json(path: Path) -> Iterator[Trajectory]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("skipping unreadable %s: %s", path, exc)
        return
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, dict) and "messages" in item and "task_id" in item:
            try:
                yield Trajectory.model_validate(item)
            except ValueError as exc:
                logger.warning("skipping invalid trajectory in %s: %s", path, exc)


def trajectory_files(source: str | Path, *, model_id: str | None = None) -> list[Path]:
    """Trajectory files under a file, directory or benchmark run directory."""
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []
    models = source / "models"
    if models.is_dir():
        root = models / model_id if model_id else models
        source = root
    files = []
    for p in sorted(source.rglob("*")):
        if not p.is_file() or p.name in _SKIP_NAMES:
            continue
        if "suite_snapshot" in p.parts or "tasks" in p.relative_to(source).parts[:1]:
            continue
        if p.name.endswith((".jsonl", ".jsonl.gz", ".json")):
            files.append(p)
    return files


def iter_trajectories(
    source: str | Path | Iterable[str | Path], *, model_id: str | None = None
) -> Iterator[Trajectory]:
    """Stream trajectories from files/dirs (``.json``, ``.jsonl``, ``.jsonl.gz``)."""
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    for src in sources:
        for path in trajectory_files(src, model_id=model_id):
            if path.name.endswith((".jsonl", ".jsonl.gz")):
                yield from _iter_jsonl(path)
            else:
                yield from _iter_json(path)


def completed_samples(path: str | Path) -> set[tuple[str, int]]:
    """``(task_id, sample_index)`` pairs present in a JSONL file (for resume)."""
    path = Path(path)
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with _open_text(path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # torn last line after a crash
            done.add((str(row.get("task_id")), int(row.get("sample_index", 0))))
    return done


def write_jsonl(trajs: Iterable[Trajectory], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "wt") as fh:
        for t in trajs:
            fh.write(t.to_json())
            fh.write("\n")
    return path
