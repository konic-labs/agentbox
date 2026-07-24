from pathlib import Path

from agentbox.jobs.artifacts import LocalArtifactStore
from agentbox.jobs.cache import ValidationCache, task_content_hash
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def test_artifact_store(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "arts")
    src = tmp_path / "a.json"
    src.write_text('{"x":1}', encoding="utf-8")
    store.put(src, "runs/r1/a.json")
    assert store.exists("runs/r1/a.json")
    dest = tmp_path / "out.json"
    store.get("runs/r1/a.json", dest)
    assert dest.read_text() == '{"x":1}'


def test_validation_cache(tmp_path: Path) -> None:
    task = Task(
        task_id="t",
        description="d",
        starter_files={"a.py": "x"},
        verifier=VerifierSpec(type=VerifierType.PYTEST),
    )
    h1 = task_content_hash(task)
    cache = ValidationCache(tmp_path / "cache")
    assert cache.get(task) is None
    cache.put(task, {"ok": True})
    assert cache.get(task) == {"ok": True}
    assert h1 == task_content_hash(task)
