"""Unit tests for new tools, decorator schemas, registry modes and text protocol."""

from __future__ import annotations

import random
from typing import Any, Literal

import pytest

from agentbox.agent.prompts import render_system_prompt
from agentbox.agent.protocols import parse_hermes_tool_calls, render_hermes_tools
from agentbox.tools.base import ToolContext
from agentbox.tools.decorator import tool
from agentbox.tools.patch import PatchError, apply_hunks, parse_unified_diff
from agentbox.tools.registry import build_tool_registry
from agentbox.types import ToolMode
from agentbox.utils.text import truncate_middle


def test_unified_diff_apply_modify_create_delete() -> None:
    patch = """\
--- a/m.py
+++ b/m.py
@@ -1,3 +1,3 @@
 def f():
-    return 1
+    return 2
 # end
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+hello
+world
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-bye
"""
    fps = parse_unified_diff(patch)
    assert [(p.old_path, p.new_path) for p in fps] == [
        ("m.py", "m.py"),
        (None, "new.txt"),
        ("old.txt", None),
    ]
    assert apply_hunks("def f():\n    return 1\n# end\n", fps[0].hunks) == (
        "def f():\n    return 2\n# end\n"
    )
    assert apply_hunks("", fps[1].hunks) == "hello\nworld\n"


def test_unified_diff_offset_and_failure() -> None:
    fps = parse_unified_diff("--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n")
    # hunk header says line 1 but the context moved down by two lines
    assert apply_hunks("z\nz\na\nb\n", fps[0].hunks) == "z\nz\na\nB\n"
    with pytest.raises(PatchError):
        apply_hunks("nothing here\n", fps[0].hunks)
    with pytest.raises(PatchError):
        parse_unified_diff("not a diff")


async def test_apply_patch_tool_is_all_or_nothing() -> None:
    from agentbox.tools.patch import ApplyPatchTool

    class SB:
        def __init__(self) -> None:
            self.files = {"a.txt": "1\n2\n"}

        async def read_text(self, path: str) -> str:
            if path not in self.files:
                raise FileNotFoundError(path)
            return self.files[path]

        async def write_files(self, files: dict[str, str]) -> None:
            self.files.update(files)

        async def exec(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("no deletes expected")

    sb = SB()
    bad = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-1\n+one\n--- a/missing\n+++ b/missing\n@@ -1 +1 @@\n-x\n+y\n"
    res = await ApplyPatchTool().execute(sb, patch=bad)
    assert getattr(res, "is_error", False)
    assert sb.files["a.txt"] == "1\n2\n"  # untouched
    good = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-1\n+one\n"
    res = await ApplyPatchTool().execute(sb, patch=good)
    assert "M a.txt" in str(res)
    assert sb.files["a.txt"] == "one\n2\n"


def test_decorator_rich_schema_and_docstring() -> None:
    @tool()
    async def search(sandbox: Any, query: str, mode: Literal["fast", "exact"] = "fast", tags: list[str] | None = None) -> str:
        """Search things.

        Args:
            query: What to look for.
            mode: Search strategy.
        """
        return query

    params = search.parameters()
    assert search.description == "Search things."
    assert params["required"] == ["query"]
    assert params["properties"]["query"]["description"] == "What to look for."
    assert params["properties"]["mode"]["enum"] == ["fast", "exact"]
    assert "array" in str(params["properties"]["tags"])


async def test_decorator_sync_function_and_ctx() -> None:
    @tool(description="add")
    def add(sandbox: Any, a: int, b: int, ctx: ToolContext) -> str:
        return f"{a + b}@{ctx.task_id}"

    assert "ctx" not in add.parameters()["properties"]
    from agentbox.tools.base import _CURRENT

    token = _CURRENT.set(ToolContext(sandbox=None, task_id="T"))
    try:
        assert await add.execute(None, a=2, b=3) == "5@T"
    finally:
        _CURRENT.reset(token)


def test_registry_modes_and_drop_keeps_one() -> None:
    assert build_tool_registry(ToolMode.STRUCTURED).names() == [
        "list_files", "read_file", "write_file", "edit_file", "run_command", "run_tests"
    ]
    ext = build_tool_registry("extended").names()
    assert {"search_files", "apply_patch", "submit"} <= set(ext)
    reg = build_tool_registry(ToolMode.STRUCTURED, drop_prob=1.0, rng=random.Random(0))
    assert len(reg) == 1
    with pytest.raises(ValueError):
        build_tool_registry("bogus")
    assert build_tool_registry(["read_file", "submit"]).names() == ["read_file", "submit"]


def test_schemas_cached_until_change() -> None:
    reg = build_tool_registry(ToolMode.SHELL)
    first = reg.openai_tools()
    assert reg.openai_tools() is first
    reg.unregister("run_command")
    assert reg.openai_tools() == []


def test_prompt_override_with_braces() -> None:
    out = render_system_prompt(override='Reply as {"ok": true} in {workspace_dir}', workspace_dir="/w")
    assert out == 'Reply as {"ok": true} in /w'
    assert "submit" in render_system_prompt(mode=ToolMode.EXTENDED)


def test_hermes_parse_and_render() -> None:
    text = (
        'thinking\n<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": "{\\"y\\": 2}"}\n</tool_call>\n'
        "<tool_call>{broken"
    )
    rest, calls = parse_hermes_tool_calls(text)
    assert rest == "thinking"
    assert [c.function.name for c in calls] == ["a", "b", "invalid_tool_call"]
    assert calls[0].function.arguments == '{"x": 1}'
    assert calls[1].function.arguments == '{"y": 2}'
    assert len({c.id for c in calls}) == 3
    assert parse_hermes_tool_calls("plain answer") == ("plain answer", [])
    rendered = render_hermes_tools(build_tool_registry(ToolMode.SHELL).openai_tools())
    assert "<tools>" in rendered and '"run_command"' in rendered


def test_truncate_middle() -> None:
    assert truncate_middle("abc", 10) == "abc"
    out = truncate_middle("a" * 100 + "END", 20)
    assert out.startswith("a" * 10) and out.endswith("END") and "truncated" in out


def test_tool_context_clamp() -> None:
    ctx = ToolContext(sandbox=None, tool_timeout_s=30)
    assert ctx.clamp_timeout(1e9) == 30
    assert ctx.clamp_timeout(-5) == 30  # invalid -> default (60) -> clamped to 30
    assert ctx.clamp_timeout("abc", default=10) == 10
    assert ctx.clamp_timeout(0.1) == 1.0
