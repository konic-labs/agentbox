"""Task filter / curriculum tests."""

from agentbox.tasks.filter import filter_tasks, sample_curriculum
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def _task(tid: str, difficulty: str, tags: list[str] | None = None) -> Task:
    return Task(
        task_id=tid,
        description="d",
        verifier=VerifierSpec(type=VerifierType.PYTEST, command="pytest"),
        metadata={"difficulty": difficulty, "tags": tags or []},
    )


def test_filter_by_difficulty_and_tags() -> None:
    tasks = [
        _task("a", "easy", ["python"]),
        _task("b", "hard", ["algo"]),
        _task("c", "medium", ["python", "algo"]),
    ]
    easy = filter_tasks(tasks, difficulty="easy")
    assert [t.task_id for t in easy] == ["a"]
    py = filter_tasks(tasks, tags=["python"])
    assert {t.task_id for t in py} == {"a", "c"}


def test_curriculum_order() -> None:
    tasks = [_task("h", "hard"), _task("e", "easy"), _task("m", "medium")]
    sampled = sample_curriculum(tasks, n=2)
    assert sampled[0].metadata["difficulty"] == "easy"
