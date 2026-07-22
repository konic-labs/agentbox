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

SHELL_SYSTEM_PROMPT = """\
You are a coding agent working inside an isolated Linux container.
Your workspace is {workspace_dir}.

You only have the run_command tool. Use standard Unix tools to inspect and edit files.
When you are done, respond with a brief final message and NO tool calls.
"""


def render_system_prompt(
    *,
    workspace_dir: str = "/workspace",
    mode: ToolMode | str = ToolMode.STRUCTURED,
    extra: str | None = None,
    override: str | None = None,
) -> str:
    if override:
        base = override.format(workspace_dir=workspace_dir)
    else:
        template = (
            SHELL_SYSTEM_PROMPT
            if mode in (ToolMode.SHELL, "shell")
            else STRUCTURED_SYSTEM_PROMPT
        )
        base = template.format(workspace_dir=workspace_dir)
    if extra:
        base = base.rstrip() + "\n\n" + extra.strip()
    return base
