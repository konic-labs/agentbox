# Tools

Tools are what the model acts with. Each tool is exposed in the OpenAI
function-calling format and executes **only inside the sandbox**, with paths
resolved against the workspace.

## Builtin tools and modes

`AgentConfig.tools` (or `agentbox run --tools`) selects a mode or an explicit
list of names:

| Mode | Tools |
| --- | --- |
| `structured` (default) | `list_files`, `read_file`, `write_file`, `edit_file`, `run_command`, `run_tests` |
| `extended` | structured + `search_files`, `apply_patch`, `submit` |
| `shell` | `run_command` |
| `custom` | none; only `custom_tools` |
| `["read_file", "run_tests", ...]` | exactly these builtins (plus `custom_tools`) |

| Tool | Parameters | Notes |
| --- | --- | --- |
| `list_files` | `path="."`, `recursive=false` | Directories end with `/`; recursive listings skip `.git`, `node_modules`, `__pycache__`, virtualenvs; bounded |
| `read_file` | `path`, `offset=1`, `limit`, `line_numbers=false` | 1-based line paging; 1 MiB cap; binary files are detected and refused |
| `write_file` | `path`, `content` | Creates parent directories |
| `edit_file` | `path`, `old_string`, `new_string`, `replace_all=false` | `old_string` must match exactly once unless `replace_all` |
| `run_command` | `command`, `timeout_s=60` | The timeout is clamped to `tool_timeout_s` and the episode deadline |
| `run_tests` | `command="python -m pytest -q"` | |
| `search_files` | `pattern`, `path="."`, `glob`, `ignore_case=false`, `max_results=100` | Extended regex; `path:line:text` results |
| `apply_patch` | `patch` | Unified diff incl. file creation/deletion; all-or-nothing |
| `submit` | `answer` | Terminal: ends the episode; later calls in the same turn are skipped |

Every tool is marked read-only or not. Read-only tools (`list_files`,
`read_file`, `search_files`) may run concurrently with each other; see
[agent: tool execution](agent.md#tool-execution).

Remove tools with `drop_tools: [run_command]`. For tool-set augmentation in
RL, `drop_tools_prob: p` drops each tool with probability `p` per rollout
(seeded, and at least one tool is always kept).

## Custom tools

A decorated function is the simplest custom tool:

```python
from agentbox import AgentConfig, tool
from agentbox.tools.base import ToolContext


@tool(read_only=True)
async def count_lines(sandbox, path: str, pattern: str | None = None) -> str:
    """Count lines in a workspace file.

    Args:
        path: File path relative to the workspace.
        pattern: Only count lines containing this text.
    """
    text = await sandbox.read_text(path)
    lines = text.splitlines()
    return str(sum(1 for line in lines if pattern is None or pattern in line))


@tool(description="Show the current step and remaining time")
def budget(sandbox, ctx: ToolContext) -> str:          # sync functions run in a worker thread
    left = ctx.remaining_s()                            # None when there is no deadline
    return f"step {ctx.step}, {'no' if left is None else round(left)} seconds left"


agent = AgentConfig(tools="structured", custom_tools=[count_lines, budget])
```

- The JSON schema comes from the type hints (any type pydantic can describe)
  and the descriptions from the Google-style `Args:` section.
- The first argument is the sandbox handle (`exec`, `read_text`,
  `write_text`, `write_files`, `list_dir`, `read_bytes`, `write_bytes`).
- Declaring a `ctx: ToolContext` parameter gives access to the step, run and
  task ids, the deadline (`remaining_s()`, `clamp_timeout()`) and output
  budgets. `agentbox.tools.base.get_tool_context()` returns the same object
  from anywhere inside a tool call.
- `@tool(terminal=True)` ends the episode when called, like `submit`.

For full control, subclass `BaseTool`:

```python
from typing import Any

from agentbox import BaseTool
from agentbox.tools.base import ToolResult


class Grep(BaseTool):
    name = "grep"
    description = "Search the workspace with grep -rn."
    read_only = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        }

    async def execute(self, sandbox: Any, pattern: str, **_: Any) -> str | ToolResult:
        res = await sandbox.exec(["grep", "-rn", "--", pattern, "."], timeout_s=30)
        if res.exit_code > 1:
            return ToolResult(content=res.stderr, is_error=True)
        return res.stdout or "(no matches)"
```

Return a string, or a `ToolResult(content, is_error=False, terminal=False, meta={})`
when the outcome is an error or should end the episode.

## Results, errors and truncation

- Each call produces exactly one tool message and one `ToolCallRecord` (name,
  arguments, result preview, `is_error`, `duration_s`, `truncated`, step).
- Arguments that do not parse, unknown tools, missing required arguments,
  timeouts and exceptions become error results the model can react to. They
  never crash the episode.
- The model sees at most `max_tool_output_chars` (16 000) of each result,
  keeping head and tail with an elision marker. Records keep a 2 000-character
  preview.
