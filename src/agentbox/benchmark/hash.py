"""Stable content hashing for benchmark suite integrity.

Hash v2 covers only *fairness-relevant* fields, selected explicitly, so adding
new config options (or editing labels / pull policy) no longer changes the hash
of every frozen suite (GH #82).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agentbox.benchmark.schema import BenchmarkSuiteManifest
from agentbox.tasks.schema import Task

HASH_VERSION = 2

_AGENT_FIELDS = (
    "tools",
    "builtins",
    "system_prompt",
    "max_steps",
    "step_timeout_s",
    "episode_timeout_s",
    "tool_timeout_s",
    "max_tool_output_chars",
    "tool_protocol",
    "drop_tools",
    "drop_tools_prob",
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _task_payload(task: Task) -> dict[str, Any]:
    data = task.model_dump(mode="json", exclude_none=True)
    data.pop("metadata", None)  # free-form; does not change what the agent faces
    return data


def _fairness_config_payload(manifest: BenchmarkSuiteManifest) -> dict[str, Any]:
    sb = manifest.sandbox
    lim = sb.limits
    agent = manifest.agent.model_dump(mode="json")
    return {
        "suite_id": manifest.suite_id,
        "version": manifest.version,
        "sandbox": {
            "image": sb.image,
            "workspace_dir": sb.workspace_dir,
            "working_dir": sb.working_dir,
            "env": dict(sb.env),
            "ensure_pytest": sb.ensure_pytest,
            "limits": {
                "cpu_count": lim.cpu_count,
                "memory_mb": lim.memory_mb,
                "pids_limit": lim.pids_limit,
                "network_disabled": lim.network_disabled,
            },
        },
        "agent": {k: agent.get(k) for k in _AGENT_FIELDS},
        "n_per_task": manifest.n_per_task,
        "seed": manifest.seed,
        "scoring": manifest.scoring.model_dump(mode="json"),
    }


def compute_suite_content_hash(tasks: list[Task], manifest: BenchmarkSuiteManifest) -> str:
    """SHA-256 hex over tasks + fairness-critical suite config."""
    ordered = sorted(tasks, key=lambda t: t.task_id)
    payload = {
        "hash_version": HASH_VERSION,
        "config": _fairness_config_payload(manifest),
        "tasks": [_task_payload(t) for t in ordered],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
