"""Task definition schema (generator ↔ seeder ↔ agent contract)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentbox.errors import TaskValidationError
from agentbox.types import VerifierType


class VerifierSpec(BaseModel):
    type: VerifierType = VerifierType.COMMAND
    command: str | None = None
    path: str | None = None
    success_exit_code: int = 0
    timeout_s: float = 60.0
    reward_success: float = 1.0
    reward_failure: float = 0.0
    parse_stdout: bool = False

    @model_validator(mode="after")
    def _check_command(self) -> VerifierSpec:
        if self.type == VerifierType.COMMAND and not self.command:
            raise ValueError("verifier.command is required when type is 'command'")
        return self


class Task(BaseModel):
    """Complete task definition for sandbox seeding and agent rollout."""

    task_id: str
    description: str
    starter_files: dict[str, str] = Field(default_factory=dict)
    setup_commands: list[str] = Field(default_factory=list)
    verifier: VerifierSpec
    metadata: dict[str, Any] = Field(default_factory=dict)
    system_prompt_extra: str | None = None
    allowed_tools: list[str] | None = None
    max_steps: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> Task:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskValidationError(f"Failed to load task from {path}: {exc}") from exc
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise TaskValidationError(f"Invalid task in {path}: {exc}") from exc

    @classmethod
    def from_dir(cls, path: str | Path) -> Task:
        """Load task.json and optional files/ directory as starter_files."""
        path = Path(path)
        task_path = path / "task.json" if path.is_dir() else path
        if not task_path.exists():
            raise TaskValidationError(f"task.json not found under {path}")
        task = cls.from_json(task_path)
        files_dir = task_path.parent / "files"
        if files_dir.is_dir():
            extra: dict[str, str] = {}
            for file_path in files_dir.rglob("*"):
                if file_path.is_file():
                    rel = str(file_path.relative_to(files_dir)).replace("\\", "/")
                    extra[rel] = file_path.read_text(encoding="utf-8")
            # files/ overrides inline starter_files on conflict
            task.starter_files = {**task.starter_files, **extra}
        return task

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
