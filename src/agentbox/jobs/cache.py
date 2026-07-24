"""Content-addressed cache for task validation reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentbox.tasks.schema import Task


def task_content_hash(task: Task) -> str:
    payload = task.model_dump(mode="json")
    # stable json
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ValidationCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, task: Task) -> dict[str, Any] | None:
        p = self._path(task_content_hash(task))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, task: Task, report: dict[str, Any]) -> None:
        p = self._path(task_content_hash(task))
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
