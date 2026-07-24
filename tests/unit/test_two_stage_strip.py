"""Two-stage strip + metadata helpers."""

from __future__ import annotations

from agentbox.tasks.generate.strip_impl import strip_impl_files
from agentbox.tasks.generate.static_qc import validate_task_static
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def test_two_stage_strip_passes_static() -> None:
    solution = {
        "parser.py": (
            "def parse(s):\n"
            "    '''Parse.'''\n"
            "    return s.strip().split(',')\n"
        ),
        "test_parser.py": (
            "from parser import parse\n"
            "def test_a():\n    assert parse('a,b')==['a','b']\n"
            "def test_b():\n    assert parse(' x ')==['x']\n"
            "def test_c():\n    assert parse('')==['']\n"
        ),
    }
    sol_src = {k: v for k, v in solution.items() if "test" not in k}
    tests = {k: v for k, v in solution.items() if "test" in k}
    starter = {**strip_impl_files(sol_src), **tests}
    assert "NotImplementedError" in starter["parser.py"]
    assert "return s.strip()" not in starter["parser.py"]
    assert "assert parse" in starter["test_parser.py"]

    task = Task(
        task_id="two_stage_demo",
        description="Implement parse() in parser.py",
        starter_files=starter,
        verifier=VerifierSpec(type=VerifierType.PYTEST),
        metadata={"has_hidden_solution": True},
    )
    rep = validate_task_static(task, min_asserts=3, require_stubs=True)
    assert rep.ok, rep.errors
