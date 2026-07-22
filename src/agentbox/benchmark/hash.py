"""Stable content hashing for benchmark suite integrity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agentbox.benchmark.schema import BenchmarkSuiteManifest
from agentbox.tasks.schema import Task


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "description": task.description,
        "starter_files": task.starter_files,
        "setup_commands": task.setup_commands,
        "verifier": task.verifier.model_dump(mode="json"),
        "metadata": task.metadata,
        "system_prompt_extra": task.system_prompt_extra,
        "allowed_tools": task.allowed_tools,
        "max_steps": task.max_steps,
    }


def _fairness_config_payload(manifest: BenchmarkSuiteManifest) -> dict[str, Any]:
    """Config fields that affect evaluation fairness (exclude free-form notes)."""
    agent = manifest.agent.model_dump(
        mode="json",
        exclude={"custom_tools"},  # non-serializable / not used in suite freeze
    )
    return {
        "suite_id": manifest.suite_id,
        "version": manifest.version,
        "sandbox": manifest.sandbox.model_dump(mode="json"),
        "agent": agent,
        "n_per_task": manifest.n_per_task,
        "seed": manifest.seed,
        "scoring": manifest.scoring.model_dump(mode="json"),
    }


def compute_suite_content_hash(
    tasks: list[Task],
    manifest: BenchmarkSuiteManifest,
) -> str:
    """SHA-256 hex over tasks + fairness-critical suite config."""
    ordered = sorted(tasks, key=lambda t: t.task_id)
    payload = {
        "config": _fairness_config_payload(manifest),
        "tasks": [_task_payload(t) for t in ordered],
    }
    raw = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
