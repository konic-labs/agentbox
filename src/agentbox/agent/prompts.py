"""Default system prompts."""

from __future__ import annotations

from agentbox.types import ToolMode

STRUCTURED_SYSTEM_PROMPT = """\
You are a coding agent working inside an isolated Linux container.
Your workspace is {workspace_dir}.

You solve the user's task using the provided tools. Prefer small, correct edits.
When you are done, respond with a brief final message and NO tool calls.
Do not invent file paths. Always verify your work with tests or commands when possible.
"""

EXTENDED_SYSTEM_PROMPT = """\
You are a coding agent working inside an isolated Linux container.
Your workspace is {workspace_dir}.

You solve the user's task using the provided tools. Prefer small, correct edits and
verify your work with tests or commands. Do not invent file paths.
When the task is complete, call the `submit` tool exactly once.
"""

SHELL_SYSTEM_PROMPT = """\
You are a coding agent working inside an isolated Linux container.
Your workspace is {workspace_dir}.

You only have the run_command tool. Use standard Unix tools to inspect and edit files.
When you are done, respond with a brief final message and NO tool calls.
"""


def _fill(template: str, workspace_dir: str) -> str:
    # str.replace, not str.format: custom prompts may contain JSON braces (GH #36).
    return template.replace("{workspace_dir}", workspace_dir)


def render_system_prompt(
    *,
    workspace_dir: str = "/workspace",
    mode: ToolMode | str | list[str] = ToolMode.STRUCTURED,
    extra: str | None = None,
    override: str | None = None,
) -> str:
    if override:
        base = _fill(override, workspace_dir)
    else:
        if mode in (ToolMode.SHELL, "shell"):
            template = SHELL_SYSTEM_PROMPT
        elif mode in (ToolMode.EXTENDED, "extended") or (
            isinstance(mode, list) and "submit" in mode
        ):
            template = EXTENDED_SYSTEM_PROMPT
        else:
            template = STRUCTURED_SYSTEM_PROMPT
        base = _fill(template, workspace_dir)
    if extra:
        base = base.rstrip() + "\n\n" + extra.strip()
    return base
