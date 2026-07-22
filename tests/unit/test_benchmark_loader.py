"""Suite load/save/create tests."""

from agentbox.benchmark.loader import create_suite_from_tasks, load_suite, save_suite
from agentbox.benchmark.schema import BenchmarkSuite, BenchmarkSuiteManifest
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def test_create_load_roundtrip(tmp_path) -> None:
    tasks_dir = tmp_path / "src_tasks"
    tdir = tasks_dir / "t1"
    tdir.mkdir(parents=True)
    Task(
        task_id="t1",
        description="fix me",
        starter_files={"a.py": "print(1)\n"},
        verifier=VerifierSpec(type=VerifierType.COMMAND, command="true"),
        metadata={"difficulty": "easy"},
    ).save_json(tdir / "task.json")
    (tdir / "files").mkdir()
    # also via from_dir pattern
    dest = tmp_path / "suite"
    suite = create_suite_from_tasks(
        tasks_dir,
        dest,
        suite_id="test-suite",
        name="Test",
        version="0.1.0",
        freeze=True,
    )
    assert suite.manifest.content_hash
    assert len(suite.tasks) >= 1

    loaded = load_suite(dest)
    assert loaded.manifest.suite_id == "test-suite"
    assert loaded.verify_integrity()
    assert any(t.task_id == "t1" for t in loaded.tasks)


def test_integrity_fails_when_tampered(tmp_path) -> None:
    tasks_dir = tmp_path / "src"
    tasks_dir.mkdir()
    Task(
        task_id="x",
        description="d",
        starter_files={"a.py": "1"},
        verifier=VerifierSpec(type=VerifierType.COMMAND, command="true"),
    ).save_json(tasks_dir / "x.json")
    dest = tmp_path / "suite"
    create_suite_from_tasks(tasks_dir, dest, suite_id="s", name="n", freeze=True)
    loaded = load_suite(dest)
    loaded.tasks[0].starter_files["a.py"] = "tampered"
    assert not loaded.verify_integrity()
