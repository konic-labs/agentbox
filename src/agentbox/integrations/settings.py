"""Settings shared by the trainer adapters (verl, OpenRLHF, SkyRL; GH #98).

Each adapter reads one YAML or JSON document from an environment variable
(``$AGENTBOX_VERL_CONFIG``, ``$AGENTBOX_OPENRLHF_CONFIG``, ``$AGENTBOX_SKYRL_CONFIG``)
with the sections ``sandbox``, ``agent``, ``rewards`` and ``remote``. The
variable holds a path to the document, or the document itself: on a cluster
without a shared filesystem, pass the text through Ray's
``runtime_env.env_vars`` so every worker on every node sees the same settings.

With a ``remote`` section, episodes run in sandboxes hosted by ``agentbox serve``
(GPU nodes need no Docker); without one, every worker uses its local Docker.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from agentbox.config import AgentConfig, RemoteConfig, SandboxConfig
from agentbox.rewards import RewardSpec
from agentbox.tasks.loader import iter_tasks
from agentbox.tasks.schema import Task


class Settings:
    def __init__(
        self,
        sandbox: SandboxConfig | None = None,
        agent: AgentConfig | None = None,
        rewards: list[RewardSpec] | None = None,
        remote: RemoteConfig | None = None,
    ) -> None:
        self.sandbox = sandbox or SandboxConfig()
        self.agent = agent or AgentConfig()
        self.rewards = rewards
        self.remote = remote


def parse_settings(data: dict[str, Any]) -> Settings:
    """Read the adapter sections; others (an agentbox.yaml's ``model``, ...) are ignored."""
    return Settings(
        sandbox=SandboxConfig.model_validate(data.get("sandbox") or {}),
        agent=AgentConfig.model_validate(data.get("agent") or {}),
        rewards=[RewardSpec.model_validate(r) for r in data["rewards"]]
        if data.get("rewards")
        else None,
        remote=RemoteConfig.model_validate(data["remote"]) if data.get("remote") else None,
    )


def load_settings(source: str | None = None) -> Settings:
    """Settings from ``source`` (default ``$AGENTBOX_VERL_CONFIG``): a path or the text."""
    source = source or os.environ.get("AGENTBOX_VERL_CONFIG")
    if not source:
        return Settings()
    text = source.strip()
    if text.startswith("{") or "\n" in text:  # the document itself
        data = yaml.safe_load(text)  # YAML is a superset of JSON
    else:
        body = Path(source).read_text(encoding="utf-8")
        data = yaml.safe_load(body) if source.endswith((".yaml", ".yml")) else json.loads(body)
    if data is not None and not isinstance(data, dict):
        raise ValueError("AgentBox settings must be a mapping of sections")
    return parse_settings(data or {})


@lru_cache(maxsize=4)
def task_index(source: str) -> dict[str, Task]:
    return {t.task_id: t for t in iter_tasks(source)}


def task_from_kwargs(kwargs: dict[str, Any]) -> Task:
    """A dataset row's task: ``extra_info.agentbox_task``, or its id + ``$AGENTBOX_TASKS``."""
    extra = kwargs.get("extra_info") or {}
    blob = extra.get("agentbox_task")
    if blob:
        return (
            Task.model_validate_json(blob) if isinstance(blob, str) else Task.model_validate(blob)
        )
    task_id = extra.get("task_id") or kwargs.get("task_id")
    source = os.environ.get("AGENTBOX_TASKS")
    if task_id and source:
        return task_index(source)[str(task_id)]
    raise KeyError("dataset row has no extra_info.agentbox_task (or task_id + $AGENTBOX_TASKS)")


def tool_text(outcome: Any) -> str:
    """Tool result text from a local :class:`ToolOutcome` or a remote result dict."""
    if isinstance(outcome, dict):
        return str(outcome.get("content") or "")
    return outcome.message.content or ""
