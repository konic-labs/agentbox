"""Streaming runner, sinks, schema upgrade and RL stats (no Docker)."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from agentbox.metrics.aggregate import aggregate_trajectories
from agentbox.model.mock import MockModelClient
from agentbox.rl.stats import group_advantages, group_stats, pass_at_k, pass_at_k_table
from agentbox.runner import parallel as parallel_mod
from agentbox.runner.parallel import ParallelRunner
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.trajectory.io import (
    DirectorySink,
    JsonlSink,
    completed_samples,
    iter_trajectories,
)
from agentbox.trajectory.schema import Trajectory, TrajectoryMetrics
from agentbox.types import FailureKind, FinalStatus, VerifierType


def _task(tid: str) -> Task:
    return Task(task_id=tid, description="d", verifier=VerifierSpec(type=VerifierType.PYTEST))


def _traj(task_id: str, sample: int = 0, *, ok: bool = True, kind: FailureKind = FailureKind.NONE,
          reward: float | None = None, duration: float = 1.0) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        run_id="run",
        rollout_id=f"{task_id}-{sample}",
        sample_index=sample,
        group_id=f"{task_id}@run",
        messages=[],
        reward=(1.0 if ok else 0.0) if reward is None else reward,
        final_status=FinalStatus.SUCCESS if ok else FinalStatus.FAILED,
        failure_kind=kind if not ok else FailureKind.NONE,
        metrics=TrajectoryMetrics(steps=2, tool_calls=3, duration_s=duration),
    )


@pytest.fixture
def fake_rollout(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace Rollout.run; failures scripted per (task_id, sample, attempt)."""
    state: dict[str, Any] = {"calls": [], "infra_once": set(), "fail": set(), "inflight": 0, "peak": 0}

    async def fake_run(task: Task, **kw: Any) -> Trajectory:
        key = (task.task_id, kw["sample_index"])
        state["calls"].append(key)
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        if key in state["infra_once"]:
            state["infra_once"].discard(key)
            return _traj(task.task_id, kw["sample_index"], ok=False, kind=FailureKind.SANDBOX)
        ok = key not in state["fail"]
        t = _traj(task.task_id, kw["sample_index"], ok=ok, kind=FailureKind.POLICY)
        t.group_id = kw["group_id"]
        return t

    monkeypatch.setattr(parallel_mod.Rollout, "run", staticmethod(fake_run))
    return state


def _runner(**kw: Any) -> ParallelRunner:
    return ParallelRunner(model=MockModelClient([]), retry_backoff_s=0.0, **kw)


async def test_stream_bounded_concurrency_and_retries(fake_rollout: dict[str, Any]) -> None:
    fake_rollout["infra_once"].add(("a", 1))
    tasks = [_task("a"), _task("b")]
    out = [t async for t in _runner(concurrency=3).stream(tasks, n_per_task=3)]
    assert len(out) == 6
    assert fake_rollout["peak"] <= 3
    retried = next(t for t in out if (t.task_id, t.sample_index) == ("a", 1))
    assert retried.success and retried.metrics.attempts == 2


async def test_persistent_infra_failure_is_not_trainable(fake_rollout: dict[str, Any]) -> None:
    async def always_infra(task: Task, **kw: Any) -> Trajectory:
        return _traj(task.task_id, kw["sample_index"], ok=False, kind=FailureKind.MODEL_API)

    parallel_mod.Rollout.run = staticmethod(always_infra)  # type: ignore[method-assign]
    out = await _runner(max_retries=2).run_tasks([_task("x")])
    assert out[0].metrics.attempts == 3 and not out[0].is_trainable


async def test_stream_groups_and_run_groups(fake_rollout: dict[str, Any]) -> None:
    groups = [g async for g in _runner(concurrency=4).stream_groups([_task("a"), _task("b")], group_size=3)]
    assert sorted(len(g) for g in groups) == [3, 3]
    for g in groups:
        assert [t.sample_index for t in g] == [0, 1, 2]
        assert len({t.group_id for t in g}) == 1
    ordered = await _runner().run_groups([_task("b"), _task("a")], group_size=2)
    assert [g[0].task_id for g in ordered] == ["b", "a"]


async def test_run_to_sink_resume(tmp_path: Path, fake_rollout: dict[str, Any]) -> None:
    path = tmp_path / "out" / "trajs.jsonl.gz"
    tasks = [_task("a"), _task("b")]
    fake_rollout["fail"].add(("b", 0))
    summary = await _runner().run_to_sink(tasks, JsonlSink(path), n_per_task=2)
    assert summary.n == 4 and summary.metrics.success_rate == 0.75
    assert completed_samples(path) == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}
    fake_rollout["calls"].clear()
    again = await _runner().run_to_sink(tasks, JsonlSink(path), n_per_task=3, resume=True)
    assert again.skipped == 4 and again.n == 2
    assert sorted(fake_rollout["calls"]) == [("a", 2), ("b", 2)]
    assert len(list(iter_trajectories(path))) == 6


async def test_streaming_memory_is_flat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tracemalloc

    from agentbox.trajectory.schema import Message

    blob = "x" * 50_000

    async def big_run(task: Task, **kw: Any) -> Trajectory:
        t = _traj(task.task_id, kw["sample_index"])
        t.messages = [Message(role="tool", content=blob, tool_call_id="c")]
        return t

    monkeypatch.setattr(parallel_mod.Rollout, "run", staticmethod(big_run))
    tasks = [_task(f"t{i}") for i in range(200)]
    tracemalloc.start()
    try:
        summary = await _runner(concurrency=8).run_to_sink(
            tasks, JsonlSink(tmp_path / "big.jsonl"), n_per_task=10
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert summary.n == 2000
    # 2000 x 50 KB = 100 MB if retained; streaming keeps a bounded window.
    assert peak < 25 * 1024 * 1024, f"peak traced memory {peak / 1e6:.1f} MB"


async def test_empty_run_needs_no_model() -> None:
    result = await ParallelRunner(concurrency=2).run([])
    assert result.trajectories == []


async def test_directory_sink_and_reader(tmp_path: Path) -> None:
    sink = DirectorySink(tmp_path / "models" / "m1" / "trajectories")
    await sink.write(_traj("t/1", 0))
    await sink.write(_traj("t/1", 1, ok=False, kind=FailureKind.POLICY))
    (tmp_path / "report.json").write_text("{}")
    loaded = list(iter_trajectories(tmp_path))
    assert len(loaded) == 2
    assert {t.sample_index for t in loaded} == {0, 1}
    assert list(iter_trajectories(tmp_path, model_id="m1"))


def test_v1_trajectory_upgrade() -> None:
    v1 = {
        "task_id": "t",
        "run_id": "abc",
        "messages": [{"role": "user", "content": "hi"}],
        "reward": 1.0,
        "final_status": "success",
        "metadata": {
            "verify_exit_code": 0,
            "verify_command": "pytest",
            "verify_success": True,
            "verify_stdout": "1 passed",
            "stop_reason": "final_answer",
            "failure_kind": "seed",
            "task_metadata": {"difficulty": "easy"},
        },
    }
    t = Trajectory.model_validate(v1)
    assert t.schema_version == 2 and t.rollout_id == "abc"
    assert t.verify is not None and t.verify.success and t.verify.stdout == "1 passed"
    assert t.stop_reason == "final_answer" and t.failure_kind == FailureKind.SEED
    assert "verify_stdout" not in t.metadata and t.metadata["task_metadata"]["difficulty"] == "easy"
    assert not t.is_trainable
    assert json.loads(t.to_json())["schema_version"] == 2


def test_pass_at_k_and_advantages() -> None:
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert math.isclose(pass_at_k(10, 3, 1), 0.3)
    assert math.isclose(pass_at_k(5, 1, 5), 1.0)
    assert math.isclose(pass_at_k(4, 2, 2), 1 - 1 / 6)
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 5)
    assert pass_at_k_table([(10, 3), (10, 0)], ks=(1, 5, 20)).keys() == {1, 5}
    adv = group_advantages([1, 0, 0, 1])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-9) and adv[0] > 0 > adv[1]
    assert group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]
    assert group_advantages([1, 0], method="rloo") == [1.0, -1.0]
    assert group_advantages([2, 0], method="drgrpo") == [1.0, -1.0]
    stats = group_stats([[1, 1], [0, 0], [1, 0]])
    assert math.isclose(stats["frac_zero_variance"], 2 / 3)


def test_metrics_accumulator() -> None:
    trajs = [
        _traj("a", 0, duration=1.0),
        _traj("a", 1, ok=False, kind=FailureKind.POLICY, duration=3.0),
        _traj("b", 0, ok=False, kind=FailureKind.SANDBOX, duration=2.0),
    ]
    m = aggregate_trajectories(trajs)
    assert m.n == 3 and math.isclose(m.success_rate, 1 / 3)
    assert m.infra_failures == 1
    assert m.by_failure_kind == {"none": 1, "policy": 1, "sandbox": 1}
    assert m.p50_duration_s == 2.0 and m.p90_duration_s == 3.0
    assert m.by_task["a"]["n"] == 2 and m.by_task["a"]["pass_at_1"] == 0.5
