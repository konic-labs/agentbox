"""Hugging Face TRL integration (GH #71).

TRL's ``GRPOTrainer(environment_factory=...)`` creates one environment per
rollout, calls ``reset(**row)``, exposes the instance's public methods as tools
(schemas from type hints + Google docstrings) and calls ``get_reward()`` when the
rollout ends. :func:`make_environment_factory` builds such a class backed by
AgentBox sandboxes::

    from agentbox.integrations.trl import make_environment_factory, tasks_to_dataset

    EnvCls = make_environment_factory(tasks, sandbox=SandboxConfig(...))
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        train_dataset=tasks_to_dataset(tasks),
        environment_factory=EnvCls,
        args=GRPOConfig(...),
    )
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbox.agent.prompts import render_system_prompt
from agentbox.config import AgentConfig, SandboxConfig
from agentbox.env.env import SyncAgentBoxEnv
from agentbox.rewards import RewardSpec
from agentbox.runner.provision import ProvisionError
from agentbox.tasks.loader import iter_tasks
from agentbox.tasks.schema import Task
from agentbox.trajectory.schema import Trajectory
from agentbox.types import ToolMode
from agentbox.utils.aio import BackgroundLoop

logger = logging.getLogger("agentbox.integrations.trl")


# ---------------------------------------------------------------- tool methods
def list_files(self, path: str = ".", recursive: bool = False) -> str:
    """List files and directories in the workspace (directories end with '/').

    Args:
        path: Directory relative to the workspace root.
        recursive: Whether to list the directory tree recursively.

    Returns:
        One entry per line.
    """
    return self._tool("list_files", path=path, recursive=recursive)


def read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
    """Read a text file from the workspace.

    Args:
        path: File path relative to the workspace root.
        offset: First line to return (1-based).
        limit: Maximum number of lines to return.

    Returns:
        The requested lines of the file.
    """
    return self._tool("read_file", path=path, offset=offset, limit=limit)


def write_file(self, path: str, content: str) -> str:
    """Create or overwrite a file in the workspace.

    Args:
        path: File path relative to the workspace root.
        content: Full new content of the file.

    Returns:
        A confirmation message.
    """
    return self._tool("write_file", path=path, content=content)


def edit_file(self, path: str, old_string: str, new_string: str) -> str:
    """Replace exactly one occurrence of old_string with new_string in a file.

    Args:
        path: File path relative to the workspace root.
        old_string: Exact text to replace (must occur exactly once).
        new_string: Replacement text.

    Returns:
        A confirmation or error message.
    """
    return self._tool("edit_file", path=path, old_string=old_string, new_string=new_string)


def run_command(self, command: str, timeout_s: float = 60.0) -> str:
    """Run a shell command in the workspace.

    Args:
        command: The shell command to execute.
        timeout_s: Timeout in seconds.

    Returns:
        Exit code, stdout and stderr.
    """
    return self._tool("run_command", command=command, timeout_s=timeout_s)


def run_tests(self, command: str = "python -m pytest -q") -> str:
    """Run the test suite.

    Args:
        command: Test command to run.

    Returns:
        Exit code and test output.
    """
    return self._tool("run_tests", command=command)


def search_files(self, pattern: str, path: str = ".", glob: str = "") -> str:
    """Search file contents with an extended regular expression.

    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search.
        glob: Only search files matching this glob, e.g. '*.py'.

    Returns:
        Matching lines as path:line:text.
    """
    return self._tool("search_files", pattern=pattern, path=path, glob=glob or None)


def apply_patch(self, patch: str) -> str:
    """Apply a unified diff to workspace files.

    Args:
        patch: Unified diff text (as produced by diff -u or git diff).

    Returns:
        The list of modified files or an error message.
    """
    return self._tool("apply_patch", patch=patch)


def submit(self, answer: str = "") -> str:
    """Finish the task; call once when the work is complete.

    Args:
        answer: Optional summary of the changes.

    Returns:
        A confirmation message.
    """
    return self._tool("submit", answer=answer)


TOOL_METHODS: dict[str, Callable[..., str]] = {
    f.__name__: f
    for f in (
        list_files,
        read_file,
        write_file,
        edit_file,
        run_command,
        run_tests,
        search_files,
        apply_patch,
        submit,
    )
}
_MODE_TOOLS = {
    ToolMode.STRUCTURED: [
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "run_tests",
    ],
    ToolMode.EXTENDED: list(TOOL_METHODS),
    ToolMode.SHELL: ["run_command"],
}


class _TRLEnvBase:
    """Shared machinery for generated TRL environments."""

    _tasks: dict[str, Task] = {}
    _order: list[str] = []
    _sandbox: SandboxConfig | None = None
    _agent: AgentConfig | None = None
    _rewards: list[RewardSpec] | None = None
    _loop: BackgroundLoop | None = None
    _cycle: Any = None
    _provision_retries: int = 1

    def __init__(self) -> None:
        self._env = SyncAgentBoxEnv(
            sandbox=self._sandbox,
            agent=self._agent,
            rewards=self._rewards,
            loop=type(self)._loop,
        )
        self._active = False
        self._infra_error: str | None = None
        self.reward: float | None = 0.0
        self.last_trajectory: Trajectory | None = None

    def _tool(self, name: str, **kwargs: Any) -> str:
        if self._infra_error is not None:
            return f"ERROR: the environment is unavailable ({self._infra_error})"
        if not self._active:
            return "ERROR: environment not reset"
        args = {k: v for k, v in kwargs.items() if v is not None}
        outcome = self._env.call_tool(name, args)
        return outcome.message.content or ""

    def reset(self, task_id: str | None = None, **kwargs: Any) -> str | None:
        """Start an episode for ``task_id`` (dataset column) or the next task.

        TRL passes the whole dataset row and appends the return value to the
        row's last prompt message, so the task prompt is returned only when the
        row carries no ``prompt`` (environment-owned data); rows built by
        :func:`tasks_to_rows` already contain it.
        """
        if self._active:
            self._env.close()
            self._active = False
        cls = type(self)
        if task_id is None:
            task_id = next(cls._cycle)
        task = cls._tasks[str(task_id)]
        self._infra_error = None
        self.reward = 0.0
        for attempt in range(cls._provision_retries + 1):
            try:
                self._env.reset(task)
                break
            except ProvisionError as exc:  # infrastructure, not the policy's fault
                logger.warning("provisioning %s failed (attempt %d): %s", task_id, attempt + 1, exc)
                self._infra_error = f"{exc.kind.value}: {exc}"
        else:
            self.reward = None  # unscorable: TRL leaves it out of the group baseline
            return None if kwargs.get("prompt") else task.prompt_text()
        self._infra_error = None
        self._active = True
        return None if kwargs.get("prompt") else task.prompt_text()

    def get_reward(self) -> float | None:
        """Verify the sandbox state, compute the reward and release the sandbox.

        Infrastructure failures (sandbox, seeding, verifier crash) return
        ``None``: TRL then treats the completion as unscorable (excluded from the
        group baseline, zero advantage) instead of training on a fake 0.
        """
        if not self._active:
            return self.reward
        try:
            result = self._env.evaluate()
            self.last_trajectory = self._env.trajectory()
            if result.failure_kind.is_infra:
                logger.warning("unscorable episode (%s): %s", result.failure_kind, result.error)
                self.reward = None
            else:
                self.reward = float(result.reward)
        finally:
            self._env.close()
            self._active = False
        return self.reward

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        if getattr(self, "_active", False):
            try:
                self._env.close()
            except Exception:
                pass


def _task_map(tasks: Iterable[Task] | Mapping[str, Task] | str | Path) -> dict[str, Task]:
    if isinstance(tasks, (str, Path)):
        return {t.task_id: t for t in iter_tasks(tasks)}
    if isinstance(tasks, Mapping):
        return dict(tasks)
    return {t.task_id: t for t in tasks}


def make_environment_factory(
    tasks: Iterable[Task] | Mapping[str, Task] | str | Path,
    *,
    sandbox: SandboxConfig | None = None,
    agent: AgentConfig | None = None,
    rewards: list[RewardSpec] | None = None,
    tools: Sequence[str] | None = None,
    name: str = "AgentBoxEnv",
) -> type:
    """Build a TRL ``environment_factory`` class over ``tasks``."""
    task_map = _task_map(tasks)
    if not task_map:
        raise ValueError("no tasks given")
    agent = agent or AgentConfig()
    if tools is None:
        mode = agent.tools
        if isinstance(mode, list):
            tools = mode
        else:
            tools = _MODE_TOOLS.get(ToolMode(mode), _MODE_TOOLS[ToolMode.STRUCTURED])
    unknown = [t for t in tools if t not in TOOL_METHODS]
    if unknown:
        raise ValueError(f"unsupported tools for TRL: {unknown}; choose from {list(TOOL_METHODS)}")
    order = sorted(task_map)
    attrs: dict[str, Any] = {
        "_tasks": task_map,
        "_order": order,
        "_sandbox": sandbox,
        "_agent": agent.model_copy(update={"tools": list(tools)}),
        "_rewards": rewards,
        "_loop": BackgroundLoop(name="agentbox-trl"),
        "_cycle": itertools.cycle(order),
    }
    attrs.update({t: TOOL_METHODS[t] for t in tools})
    return type(name, (_TRLEnvBase,), attrs)


def tasks_to_rows(
    tasks: Iterable[Task],
    *,
    agent: AgentConfig | None = None,
    workspace_dir: str = "/workspace",
    system_prompt: bool = True,
) -> list[dict[str, Any]]:
    """Dataset rows: conversational ``prompt`` + ``task_id`` (passed to ``reset``)."""
    agent = agent or AgentConfig()
    rows = []
    for t in tasks:
        prompt: list[dict[str, str]] = []
        if system_prompt:
            system = render_system_prompt(
                workspace_dir=workspace_dir,
                mode=agent.tools,
                extra=t.system_prompt_extra,
                override=agent.system_prompt,
            )
            prompt.append({"role": "system", "content": system})
        prompt.append({"role": "user", "content": t.prompt_text()})
        rows.append({"prompt": prompt, "task_id": t.task_id})
    return rows


def tasks_to_dataset(tasks: Iterable[Task], **kwargs: Any) -> Any:
    """A ``datasets.Dataset`` built from :func:`tasks_to_rows`."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pip install 'agentbox-rl[trl]' (needs datasets)") from exc
    return Dataset.from_list(tasks_to_rows(tasks, **kwargs))
