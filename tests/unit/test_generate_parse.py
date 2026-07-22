"""Task generation parse tests (no network)."""

import json

from agentbox.tasks.generate.validate import parse_task_from_prediction


def test_parse_task_from_prediction() -> None:
    task = parse_task_from_prediction(
        task_id="t1",
        description="Fix the bug",
        starter_files_json=json.dumps({"a.py": "x=1\n", "test_a.py": "def test():\n  assert False\n"}),
        setup_commands_json=json.dumps(["pip install -q pytest"]),
        verifier_json=json.dumps(
            {"type": "pytest", "command": "python -m pytest -q", "success_exit_code": 0}
        ),
        metadata_json=json.dumps({"difficulty": "easy", "tags": ["python"]}),
        generator_model="test-model",
    )
    assert task.task_id == "t1"
    assert "a.py" in task.starter_files
    assert task.setup_commands == ["pip install -q pytest"]
    assert task.metadata["source"] == "dspy"
    assert task.metadata["generator_model"] == "test-model"
