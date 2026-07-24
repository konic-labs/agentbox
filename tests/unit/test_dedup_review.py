"""Dedup + review queue unit tests."""

from __future__ import annotations

from pathlib import Path

from agentbox.tasks.generate.dedup import (
    difficulty_heuristic,
    is_near_duplicate,
    task_signature_hash,
)
from agentbox.tasks.generate.review import (
    ReviewDecision,
    export_review_queue,
    filter_tasks_by_decisions,
    load_review_decisions,
    queue_item_from_task,
)
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def _task(tid: str, impl: str, tests: str) -> Task:
    return Task(
        task_id=tid,
        description=f"Implement in solution.py for {tid}",
        starter_files={"solution.py": impl, "test_solution.py": tests},
        verifier=VerifierSpec(type=VerifierType.PYTEST),
    )


def test_signature_hash_stable() -> None:
    t1 = _task("a", "def foo():\n    raise NotImplementedError\n", "def test_a():\n    assert True\n")
    t2 = _task("b", "def foo():\n    raise NotImplementedError\n", "def test_a():\n    assert True\n")
    assert task_signature_hash(t1) == task_signature_hash(t2)


def test_near_duplicate() -> None:
    tests = "def test_a():\n    assert True\n\ndef test_b():\n    assert 1==1\n"
    a = _task("a", "def foo():\n    raise NotImplementedError\n", tests)
    b = _task("b", "def foo():\n    raise NotImplementedError\n", tests)
    c = _task(
        "c",
        "def bar(x, y, z):\n    raise NotImplementedError\n",
        "def test_other():\n    assert bar(1,2,3)==9\n\ndef test_x():\n    assert False\n",
    )
    dup, reason = is_near_duplicate(b, [a])
    assert dup, reason
    dup2, _ = is_near_duplicate(c, [a], jaccard_threshold=0.95)
    assert not dup2


def test_difficulty_heuristic() -> None:
    t = _task(
        "d",
        "def foo():\n    raise NotImplementedError\n",
        "\n".join(f"def test_{i}():\n    assert {i}\n" for i in range(6)),
    )
    h = difficulty_heuristic(t)
    assert h["n_asserts"] >= 6
    assert h["heuristic_label"] in {"easy", "medium", "hard"}


def test_review_queue_roundtrip(tmp_path: Path) -> None:
    t = _task("r1", "def foo():\n    raise NotImplementedError\n", "def test_a():\n    assert True\n")
    item = queue_item_from_task(t, path="r1.json", score=0.7)
    qpath = tmp_path / "queue.jsonl"
    export_review_queue([item], qpath)
    assert qpath.exists()
    dpath = tmp_path / "decisions.jsonl"
    dpath.write_text(
        ReviewDecision(task_id="r1", decision="accept").model_dump_json() + "\n"
        + ReviewDecision(task_id="r2", decision="reject").model_dump_json() + "\n",
        encoding="utf-8",
    )
    dec = load_review_decisions(dpath)
    t2 = _task("r2", "def x():\n    pass\n", "def test_x():\n    assert True\n")
    kept = filter_tasks_by_decisions([t, t2], dec)
    assert [x.task_id for x in kept] == ["r1"]
