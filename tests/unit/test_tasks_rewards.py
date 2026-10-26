"""Task schema v2, loaders, verifier parsing and reward composition (no Docker)."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentbox.errors import TaskValidationError
from agentbox.rewards import RewardContext, RewardSpec, compute_reward, default_reward_specs
from agentbox.config import AgentConfig
from agentbox.tasks.loader import iter_tasks, load_task
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.tasks.verifier import parse_junit, parse_pytest_summary, parse_stdout_reward
from agentbox.types import VerifierType


def _task(tid: str = "t", **kw: object) -> Task:
    return Task(task_id=tid, description="desc", verifier=VerifierSpec(type=VerifierType.PYTEST), **kw)  # type: ignore[arg-type]


def test_verifier_spec_normalisation_and_validation() -> None:
    assert VerifierSpec(type=VerifierType.SHELL, command="true").type == VerifierType.COMMAND
    assert VerifierSpec(type=VerifierType.TEST).type == VerifierType.PYTEST
    assert VerifierSpec(type=VerifierType.COMMAND, command="x", parse_stdout=True).reward_mode == "stdout"
    with pytest.raises(ValueError):
        VerifierSpec(type=VerifierType.CUSTOM)
    with pytest.raises(ValueError):
        VerifierSpec(type=VerifierType.COMMAND, command="x", reward_mode="custom")
    assert not VerifierSpec(type=VerifierType.PYTEST).tamper_protection
    assert VerifierSpec(type=VerifierType.PYTEST, protected_paths=["t*"]).tamper_protection


def test_parsers() -> None:
    xml = b'<testsuites><testsuite tests="4" failures="1" errors="1" skipped="1"/></testsuites>'
    assert parse_junit(xml) == {"tests": 4, "failures": 1, "errors": 1, "skipped": 1}
    assert parse_junit(b"<not-xml") is None
    assert parse_pytest_summary("....\n2 failed, 3 passed, 1 skipped in 0.1s") == {
        "tests": 6, "failures": 2, "errors": 0, "skipped": 1
    }
    assert parse_pytest_summary("no summary") is None
    assert parse_stdout_reward("log\n0.75\n") == (0.75, {})
    assert parse_stdout_reward('{"reward": 1, "acc": 0.5}') == (1.0, {"acc": 0.5})
    assert parse_stdout_reward("not a number") is None


def test_task_prompt_seed_files_and_dir_roundtrip(tmp_path: Path) -> None:
    blob = bytes(range(10))
    t = _task(
        starter_files={"a.py": "x"},
        binary_files={"b.bin": base64.b64encode(blob).decode()},
        prompt="Do: {description}\nFiles:\n{files}",
    )
    assert t.prompt_text() == "Do: desc\nFiles:\na.py"
    assert t.seed_files() == {"a.py": b"x", "b.bin": blob}
    t.save_dir(tmp_path / "task")
    loaded = load_task(tmp_path / "task" / "task.json")  # sibling files/ must be loaded (GH #52)
    assert loaded.starter_files == {"a.py": "x"}
    assert base64.b64decode(loaded.binary_files["b.bin"]) == blob


def test_iter_tasks_mixed_sources(tmp_path: Path) -> None:
    _task("dir-task", starter_files={"f": "1"}).save_dir(tmp_path / "ds" / "t1")
    _task("loose").save_json(tmp_path / "ds" / "loose.json")
    with gzip.open(tmp_path / "ds" / "more.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(_task("j1").model_dump()) + "\n")
        fh.write(json.dumps(_task("loose").model_dump()) + "\n")  # duplicate id
        fh.write("{broken\n")
    (tmp_path / "ds" / "report.json").write_text("{}")
    ids = [t.task_id for t in iter_tasks(tmp_path / "ds")]
    assert sorted(ids) == ["dir-task", "j1", "loose"]
    with pytest.raises(TaskValidationError):
        list(iter_tasks(tmp_path / "ds", strict=True))
    with pytest.raises(TaskValidationError):
        list(iter_tasks(tmp_path / "missing"))


async def test_compute_reward_components() -> None:
    verify = SimpleNamespace(reward=1.0, success=True, tests_total=4, tests_passed=3, tampered=[])
    records = [SimpleNamespace(is_error=True, result="ERROR: unknown tool 'x'"), SimpleNamespace(is_error=False, result="ok")]
    loop = SimpleNamespace(steps=5, stop_reason="submitted", tool_call_records=records,
                           usage_totals={"completion_tokens": 1000}, messages=[])
    ctx = RewardContext(task=_task(), verify=verify, loop=loop)  # type: ignore[arg-type]
    total, parts = await compute_reward(
        [
            RewardSpec(name="verifier"),
            RewardSpec(name="step_penalty", params={"per_step": 0.01}),
            RewardSpec(name="tool_error_penalty", params={"per_error": 0.1}),
            RewardSpec(name="valid_tool_call_ratio", weight=0.5),
            RewardSpec(name="tests_fraction", weight=0.0),
            RewardSpec(name="submitted", weight=0.1),
            RewardSpec(name="verifier"),
        ],
        ctx,
    )
    assert parts["verifier"] == 1.0 and parts["verifier#2"] == 1.0
    assert abs(parts["step_penalty"] + 0.05) < 1e-12
    assert abs(parts["tool_error_penalty"] + 0.1) < 1e-12
    assert parts["valid_tool_call_ratio"] == 0.25
    assert abs(total - (1 - 0.05 - 0.1 + 0.25 + 0 + 0.1 + 1)) < 1e-9
    with pytest.raises(KeyError):
        await compute_reward([RewardSpec(name="nope")], ctx)


async def test_dotted_path_reward_and_defaults() -> None:
    ctx = RewardContext(task=_task(), verify=None, loop=None)
    total, parts = await compute_reward([RewardSpec(name="tests.unit.test_tasks_rewards:_const_reward")], ctx)
    assert total == 0.5 and "tests.unit.test_tasks_rewards:_const_reward" in parts
    specs = default_reward_specs(AgentConfig(step_penalty=0.1, timeout_penalty=0.2))
    assert [s.name for s in specs] == ["verifier", "step_penalty", "timeout_penalty"]


def _const_reward(ctx: RewardContext) -> float:
    return 0.5
