"""Builtin tools against a real Docker sandbox."""

from __future__ import annotations

import pytest

from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.sandbox.manager import SandboxManager
from agentbox.tools.file_tools import ListFilesTool, ReadFileTool
from agentbox.tools.patch import ApplyPatchTool
from agentbox.tools.search import SearchFilesTool
from agentbox.tools.shell import RunCommandTool

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture
async def sandbox():  # type: ignore[no-untyped-def]
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    mgr = SandboxManager(
        SandboxConfig(ensure_pytest=False, limits=ResourceLimits(network_disabled=True))
    )
    sb = await mgr.create(task_id="tools", run_id="tools")
    await mgr.write_files(
        sb,
        {
            "src/app.py": "def add(a, b):\n    return a - b\n",
            "src/util.py": "VALUE = 1\n",
            "notes.txt": "\n".join(f"line {i}" for i in range(1, 101)) + "\n",
        },
    )
    try:
        yield sb
    finally:
        await mgr.destroy(sb)


async def test_search_files(sandbox) -> None:  # type: ignore[no-untyped-def]
    out = await SearchFilesTool().execute(sandbox, pattern=r"return a [-+] b", glob="*.py")
    assert out == "src/app.py:2:    return a - b"
    assert await SearchFilesTool().execute(sandbox, pattern="nomatch_xyz") == "(no matches)"


async def test_read_file_ranges(sandbox) -> None:  # type: ignore[no-untyped-def]
    out = await ReadFileTool().execute(
        sandbox, path="notes.txt", offset=10, limit=3, line_numbers=True
    )
    assert "10\tline 10" in str(out) and "12\tline 12" in str(out)
    assert "[lines 10-12 of 100]" in str(out)
    err = await ReadFileTool().execute(sandbox, path="src")
    assert "is a directory" in str(err)


async def test_apply_patch_real_sandbox(sandbox) -> None:  # type: ignore[no-untyped-def]
    patch = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
    out = await ApplyPatchTool().execute(sandbox, patch=patch)
    assert "M src/app.py" in str(out)
    res = await RunCommandTool().execute(
        sandbox,
        command="python -c 'import sys; sys.path.insert(0, \"src\"); import app; print(app.add(2, 3))'",
    )
    assert "5" in str(res)


async def test_list_files_marks_dirs(sandbox) -> None:  # type: ignore[no-untyped-def]
    out = str(await ListFilesTool().execute(sandbox, path="."))
    assert "src/" in out.splitlines()
    missing = await ListFilesTool().execute(sandbox, path="nope")
    assert getattr(missing, "is_error", False)
    # a file path lists itself, workspace-relative like every other tool path
    assert str(await ListFilesTool().execute(sandbox, path="src/app.py")) == "src/app.py"
