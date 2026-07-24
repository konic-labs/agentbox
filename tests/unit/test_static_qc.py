"""Static QC + strip_impl unit tests."""

from __future__ import annotations

from agentbox.tasks.generate.static_qc import validate_task_static
from agentbox.tasks.generate.strip_impl import strip_impl_files, strip_python_source
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType


def _task(files: dict[str, str], desc: str = "Implement foo") -> Task:
    return Task(
        task_id="t",
        description=desc,
        starter_files=files,
        setup_commands=["pip install -q pytest"],
        verifier=VerifierSpec(type=VerifierType.PYTEST, command="python -m pytest -q"),
    )


def test_static_ok_stub() -> None:
    t = _task(
        {
            "solution.py": "def foo(x):\n    raise NotImplementedError\n",
            "test_solution.py": "from solution import foo\n\ndef test_a():\n    assert foo(1)==1\n\ndef test_b():\n    assert foo(2)==2\n\ndef test_c():\n    assert foo(3)==3\n",
        }
    )
    r = validate_task_static(t, min_asserts=3)
    assert r.ok, r.errors


def test_static_reject_impl() -> None:
    t = _task(
        {
            "solution.py": "def foo(x):\n    return x + 1\n",
            "test_solution.py": "def test_a():\n    assert True\n\ndef test_b():\n    assert True\n\ndef test_c():\n    assert True\n",
        }
    )
    r = validate_task_static(t, min_asserts=3, require_stubs=True)
    assert not r.ok
    assert any("stub" in e for e in r.errors)


def test_static_reject_few_asserts() -> None:
    t = _task(
        {
            "solution.py": "def foo(x):\n    raise NotImplementedError\n",
            "test_solution.py": "def test_a():\n    assert True\n",
        }
    )
    r = validate_task_static(t, min_asserts=3)
    assert not r.ok


def test_strip_python_source() -> None:
    src = '''
def foo(x):
    """Doc."""
    return x + 1

class C:
    def bar(self):
        return 2
'''
    out = strip_python_source(src)
    assert "NotImplementedError" in out
    assert "return x + 1" not in out
    assert "return 2" not in out


def test_strip_impl_files_keeps_tests() -> None:
    files = {
        "solution.py": "def foo():\n    return 1\n",
        "test_solution.py": "def test_foo():\n    assert True\n",
    }
    out = strip_impl_files(files)
    assert "NotImplementedError" in out["solution.py"]
    assert "assert True" in out["test_solution.py"]
