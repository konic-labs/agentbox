"""Unit tests for LLM task judge helpers (no network)."""

from __future__ import annotations

import json

from agentbox.tasks.generate.llm_validate import (
    _as_bool,
    _as_float,
    _prediction_to_report,
    merge_validation_reports,
)
from agentbox.tasks.generate.validate import ValidationReport, task_payload_for_judge
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def _sample_task(*, near_solution: bool = False) -> Task:
    if near_solution:
        starter = {
            "solution.py": (
                "def is_even(n):\n"
                "    # BUG: inverted\n"
                "    return n % 2 != 0\n"
            ),
            "test_solution.py": "from solution import is_even\ndef test():\n    assert is_even(2)\n",
        }
        desc = "Fix the inverted boolean; should use == 0 instead of != 0"
    else:
        starter = {
            "solution.py": (
                "def reverse_string(s: str) -> str:\n"
                '    """Return reversed string."""\n'
                "    raise NotImplementedError\n"
            ),
            "test_solution.py": (
                "from solution import reverse_string\n"
                "def test_basic():\n"
                '    assert reverse_string("ab") == "ba"\n'
            ),
        }
        desc = "Implement reverse_string so tests pass. Do not modify tests."
    return Task(
        task_id="t_sample",
        description=desc,
        starter_files=starter,
        setup_commands=["pip install -q pytest"],
        verifier=VerifierSpec(type=VerifierType.PYTEST, command="python -m pytest -q"),
        metadata={"difficulty": "easy", "language": "python"},
    )


def test_as_bool_and_float() -> None:
    assert _as_bool(True) is True
    assert _as_bool("false") is False
    assert _as_bool("yes") is True
    assert _as_float("0.85") == 0.85
    assert _as_float("score=0.7") == 0.7


def test_task_payload_for_judge() -> None:
    task = _sample_task()
    payload = task_payload_for_judge(task)
    assert payload["task_id"] == "t_sample"
    files = json.loads(payload["starter_files_json"])
    assert "solution.py" in files
    assert "NotImplementedError" in files["solution.py"]


def test_prediction_to_report_accept() -> None:
    class Pred:
        accept = True
        score = 0.9
        starter_is_near_solution = False
        description_leaks_fix = False
        agent_must_implement = True
        reasons = "stub starter with clear tests"
        suggested_fixes = ""

    report = _prediction_to_report(_sample_task(), Pred(), min_score=0.65)
    assert report.ok is True
    assert report.llm_score == 0.9
    assert report.llm_accept is True


def test_prediction_to_report_reject_near_solution() -> None:
    class Pred:
        accept = False
        score = 0.2
        starter_is_near_solution = True
        description_leaks_fix = True
        agent_must_implement = False
        reasons = "full solution with one flipped operator"
        suggested_fixes = "replace body with NotImplementedError"

    report = _prediction_to_report(
        _sample_task(near_solution=True), Pred(), min_score=0.65
    )
    assert report.ok is False
    assert report.llm_ok is False
    assert any("near_solution" in e for e in report.errors)


def test_merge_validation_reports() -> None:
    task = _sample_task()
    docker = ValidationReport(
        ok=True,
        task=task,
        seed_ok=True,
        verifier_runs=True,
        verifier_fails_on_starter=True,
    )
    llm = ValidationReport(
        ok=False,
        task=task,
        errors=["llm_judge reject"],
        llm_ok=False,
        llm_score=0.1,
        llm_accept=False,
        llm_reasons="near solution",
    )
    merged = merge_validation_reports(docker, llm, task=task)
    assert merged.ok is False
    assert merged.seed_ok is True
    assert merged.llm_score == 0.1
    assert "llm_judge reject" in merged.errors
