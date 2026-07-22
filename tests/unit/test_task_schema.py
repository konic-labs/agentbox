"""Task schema validation tests."""

from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def test_task_roundtrip(tmp_path) -> None:
    task = Task(
        task_id="x",
        description="fix it",
        starter_files={"a.py": "print(1)"},
        setup_commands=["pip install -q pytest"],
        verifier=VerifierSpec(type=VerifierType.PYTEST, command="python -m pytest -q"),
        metadata={"difficulty": "easy"},
    )
    path = tmp_path / "task.json"
    task.save_json(path)
    loaded = Task.from_json(path)
    assert loaded.task_id == "x"
    assert loaded.setup_commands == ["pip install -q pytest"]
    assert loaded.starter_files["a.py"] == "print(1)"
