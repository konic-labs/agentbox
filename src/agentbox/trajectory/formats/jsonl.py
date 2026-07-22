"""JSONL export for trajectory datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from agentbox.trajectory.schema import Trajectory


def export_jsonl(trajs: Iterable[Trajectory], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for traj in trajs:
            f.write(traj.model_dump_json())
            f.write("\n")
