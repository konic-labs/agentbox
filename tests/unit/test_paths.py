"""Unit tests for workspace path jail."""

import pytest

from agentbox.errors import PathEscapeError
from agentbox.sandbox.paths import resolve_workspace_path


def test_relative_path_ok() -> None:
    assert resolve_workspace_path("/workspace", "foo.py") == "/workspace/foo.py"
    assert resolve_workspace_path("/workspace", "src/a/b.py") == "/workspace/src/a/b.py"


def test_dot_path_ok() -> None:
    assert resolve_workspace_path("/workspace", ".") == "/workspace"
    assert resolve_workspace_path("/workspace", "./x") == "/workspace/x"


def test_parent_escape_rejected() -> None:
    with pytest.raises(PathEscapeError):
        resolve_workspace_path("/workspace", "../etc/passwd")
    with pytest.raises(PathEscapeError):
        resolve_workspace_path("/workspace", "a/../../outside")


def test_absolute_outside_rejected() -> None:
    with pytest.raises(PathEscapeError):
        resolve_workspace_path("/workspace", "/etc/passwd")


def test_absolute_inside_ok() -> None:
    assert resolve_workspace_path("/workspace", "/workspace/x.py") == "/workspace/x.py"
