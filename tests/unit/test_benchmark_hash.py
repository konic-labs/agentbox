"""Suite content hash stability."""

from agentbox.benchmark.hash import compute_suite_content_hash
from agentbox.benchmark.schema import BenchmarkSuiteManifest
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def _task(tid: str, body: str = "x=1") -> Task:
    return Task(
        task_id=tid,
        description="d",
        starter_files={"a.py": body},
        verifier=VerifierSpec(type=VerifierType.PYTEST, command="pytest"),
    )


def test_hash_stable_under_task_reorder() -> None:
    m = BenchmarkSuiteManifest(suite_id="s", name="n", version="1.0.0")
    h1 = compute_suite_content_hash([_task("b"), _task("a")], m)
    h2 = compute_suite_content_hash([_task("a"), _task("b")], m)
    assert h1 == h2


def test_hash_changes_with_files() -> None:
    m = BenchmarkSuiteManifest(suite_id="s", name="n", version="1.0.0")
    h1 = compute_suite_content_hash([_task("a", "v1")], m)
    h2 = compute_suite_content_hash([_task("a", "v2")], m)
    assert h1 != h2
