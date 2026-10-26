"""JSONL export for trajectory datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from agentbox.trajectory.io import write_jsonl
from agentbox.trajectory.schema import Trajectory


def export_jsonl(trajs: Iterable[Trajectory], path: str | Path) -> None:
    """Write trajectories as JSON lines (``.gz`` suffix compresses)."""
    write_jsonl(trajs, path)
