"""Task definition schema (generator ↔ seeder ↔ agent ↔ verifier contract)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentbox.errors import TaskValidationError
from agentbox.types import VerifierType

# Files a policy can create to subvert pytest (collection hooks, addopts, site hooks).
PYTEST_TAMPER_FILES: tuple[str, ...] = (
    "conftest.py",
    "*/conftest.py",
    "sitecustomize.py",
    "usercustomize.py",
    "pytest.ini",
    "*/pytest.ini",
    "tox.ini",
    "setup.cfg",
)

RewardMode = Literal["binary", "pytest_fraction", "stdout", "custom"]


class VerifierSpec(BaseModel):
    """How the outcome of an episode is measured (ground truth for reward)."""

    type: VerifierType = Field(
        default=VerifierType.COMMAND, description="pytest | command | custom (test/shell aliases)."
    )
    command: str | None = Field(
        default=None, description="Shell command run in the workspace (pytest: default pytest -q)."
    )
    path: str | None = Field(
        default=None, description="pytest only: test path when no command is given."
    )
    success_exit_code: int = Field(default=0, description="Exit code that counts as success.")
    timeout_s: float = Field(default=60.0, description="Verifier command timeout.")
    reward_success: float = Field(default=1.0, description="Reward on success (binary mode).")
    reward_failure: float = Field(default=0.0, description="Reward on failure (binary mode).")
    reward_mode: RewardMode = Field(
        default="binary",
        description=(
            "binary: success/failure reward; pytest_fraction: interpolate by passed/total "
            "tests (JUnit XML); stdout: last stdout line is a float or JSON {'reward': x}; "
            "custom: reward_fn (dotted path) computes the reward"
        ),
    )
    parse_stdout: bool = Field(default=False, description="Deprecated alias of reward_mode=stdout")
    reward_fn: str | None = Field(
        default=None, description="'package.module:function' for reward_mode=custom"
    )
    success_threshold: float | None = Field(
        default=None, description="If set, success means reward >= threshold"
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Extra environment for the verifier command."
    )
    workdir: str | None = Field(
        default=None, description="cwd for the verifier (default: workspace)."
    )
    pre_commands: list[str] = Field(
        default_factory=list, description="Run before the verifier command (not scored)"
    )
    # --- reward-hacking protection (GH #49) ---
    hidden_files: dict[str, str] = Field(
        default_factory=dict,
        description="Never shown to the agent; written right before verification",
    )
    protected_paths: list[str] = Field(
        default_factory=list,
        description="Globs of starter files restored before verification; edits are flagged",
    )
    forbid_new_files: list[str] = Field(
        default_factory=list,
        description="Globs (e.g. PYTEST_TAMPER_FILES) the agent may not create; deleted + flagged",
    )
    on_tamper: Literal["fail", "flag"] = Field(
        default="fail",
        description="fail: tampering scores tamper_reward and is not a success; flag: record only.",
    )
    tamper_reward: float | None = Field(
        default=None, description="Reward when tampering is detected (default reward_failure)"
    )
    hardened: bool = Field(
        default=True,
        description="Clean pytest env (PYTEST_ADDOPTS, user site) and disable cache provider",
    )

    @model_validator(mode="after")
    def _normalise(self) -> VerifierSpec:
        if self.type == VerifierType.SHELL:
            self.type = VerifierType.COMMAND
        elif self.type == VerifierType.TEST:
            self.type = VerifierType.PYTEST
        if self.parse_stdout and self.reward_mode == "binary":
            self.reward_mode = "stdout"
        if self.type == VerifierType.COMMAND and not self.command:
            raise ValueError("verifier.command is required when type is 'command'")
        if self.type == VerifierType.CUSTOM and not (self.command or self.reward_fn):
            raise ValueError("verifier type 'custom' needs a command or reward_fn")
        if self.reward_mode == "custom" and not self.reward_fn:
            raise ValueError("reward_mode 'custom' requires reward_fn")
        return self

    @property
    def tamper_protection(self) -> bool:
        return bool(self.protected_paths or self.forbid_new_files or self.hidden_files)


class EnvironmentSpec(BaseModel):
    """Per-task sandbox overrides merged over the run's SandboxConfig (GH #25)."""

    image: str | None = Field(default=None, description="Image for this task's sandboxes.")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment merged over SandboxConfig.env."
    )
    network: bool | None = Field(default=None, description="Allow network for this task.")
    cpu_count: float | None = Field(default=None, description="CPU limit override.")
    memory_mb: int | None = Field(default=None, description="Memory limit override (MiB).")
    working_dir: str | None = Field(
        default=None, description="Workspace and cwd for this task's sandboxes."
    )
    ensure_pytest: bool | None = Field(default=None, description="Override ensure_pytest.")


class Task(BaseModel):
    """Complete task definition for sandbox seeding and agent rollout."""

    task_id: str = Field(description="Unique id (used in file and container names).")
    description: str = Field(description="What the agent must do (the default user prompt).")
    prompt: str | None = Field(
        default=None,
        description="User prompt template; {description} and {files} are substituted",
    )
    starter_files: dict[str, str] = Field(
        default_factory=dict, description="Workspace-relative path -> text content."
    )
    binary_files: dict[str, str] = Field(
        default_factory=dict, description="path -> base64 content for non-text starter files"
    )
    setup_commands: list[str] = Field(
        default_factory=list,
        description="Run after seeding, before the agent (snapshotted into cached images).",
    )
    verifier: VerifierSpec = Field(description="How the outcome is measured.")
    environment: EnvironmentSpec | None = Field(
        default=None, description="Per-task sandbox overrides."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form; copied into trajectories."
    )
    system_prompt_extra: str | None = Field(
        default=None, description="Appended to the system prompt."
    )
    allowed_tools: list[str] | None = Field(
        default=None, description="Restrict the tool set for this task."
    )
    max_steps: int | None = Field(default=None, description="Override AgentConfig.max_steps.")
    episode_timeout_s: float | None = Field(
        default=None, description="Override AgentConfig.episode_timeout_s."
    )
    split: Literal["train", "eval", "test"] | None = Field(
        default=None, description="Dataset split (iter_tasks(split=...), `run --split`)."
    )
    group_key: str | None = Field(
        default=None,
        description="Label shared by variants of one problem; copied into trajectories.",
    )
    version: str | None = Field(default=None, description="Task revision (part of task_hash).")

    def prompt_text(self) -> str:
        """The user message shown to the agent."""
        if not self.prompt:
            return self.description
        files = "\n".join(sorted(self.starter_files))
        return self.prompt.replace("{description}", self.description).replace("{files}", files)

    def seed_files(self) -> dict[str, bytes]:
        """Starter + binary files as bytes, ready for the sandbox."""
        out = {p: c.encode("utf-8") for p, c in self.starter_files.items()}
        for p, b64 in self.binary_files.items():
            out[p] = base64.b64decode(b64)
        return out

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
        """Load task.json plus an optional files/ directory (text or binary)."""
        path = Path(path)
        task_path = path / "task.json" if path.is_dir() else path
        if not task_path.exists():
            raise TaskValidationError(f"task.json not found under {path}")
        task = cls.from_json(task_path)
        files_dir = task_path.parent / "files"
        if files_dir.is_dir():
            text: dict[str, str] = {}
            binary: dict[str, str] = {}
            for file_path in sorted(files_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(files_dir).as_posix()
                raw = file_path.read_bytes()
                try:
                    if b"\x00" in raw:
                        raise UnicodeDecodeError("utf-8", raw, 0, 1, "nul byte")
                    text[rel] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    binary[rel] = base64.b64encode(raw).decode("ascii")
            # files/ overrides inline files on conflict
            task.starter_files = {**task.starter_files, **text}
            task.binary_files = {**task.binary_files, **binary}
        return task

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        from agentbox.utils.io import atomic_write_text

        atomic_write_text(path, self.model_dump_json(indent=indent, exclude_defaults=False))

    def save_dir(self, directory: str | Path) -> Path:
        """Write ``task.json`` + ``files/`` (starter files as real files)."""
        directory = Path(directory)
        files_dir = directory / "files"
        for rel, content in self.starter_files.items():
            fp = files_dir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
        for rel, b64 in self.binary_files.items():
            fp = files_dir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(base64.b64decode(b64))
        slim = self.model_copy(update={"starter_files": {}, "binary_files": {}})
        slim.save_json(directory / "task.json")
        return directory
