# Tools

Tools are the only way agents mutate or inspect the sandbox. Definitions live
on the host; execution always goes through the sandbox facade.

## Builtins

| Name | Parameters | Behavior |
| --- | --- | --- |
| `list_files` | `path="."`, `recursive=false` | List entries under workspace |
| `read_file` | `path` | Read text (truncates very large files) |
| `write_file` | `path`, `content` | Create / overwrite |
| `edit_file` | `path`, `old_string`, `new_string` | Exact **one** occurrence replace |
| `run_command` | `command`, optional `timeout_s` | `/bin/sh -lc` in container |
| `run_tests` | `command="python -m pytest -q"` | Thin wrapper over `run_command` |

### `edit_file` semantics

- `0` matches → error tool result  
- `>1` matches → error (must be unique)  
- `1` match → replace once and write  

## Tool modes

| Mode | Builtin set |
| --- | --- |
| `structured` | All six builtins |
| `shell` | Only `run_command` |
| `custom` | None unless listed / custom tools provided |
| `list[str]` | Named builtins only |

Custom tools merge on top; **same `name` overrides** a builtin.

## Custom tools

### Class style

```python
from agentbox.tools import BaseTool

class SearchRepoTool(BaseTool):
    name = "search_repo"
    description = "Ripgrep-like search under /workspace"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, sandbox, query: str) -> str:
        r = await sandbox.exec(f"grep -RIn -- {query!r} . || true")
        return r.stdout or "(no matches)"
```

### Decorator style

```python
from agentbox.tools import tool

@tool(description="Count lines in a file")
async def count_lines(sandbox, path: str) -> str:
    r = await sandbox.exec(f"wc -l -- {path}")
    return r.stdout.strip()
```

Signature must start with `sandbox`. Simple JSON schemas are inferred from type
hints; pass `parameters=` for complex schemas.

### Registration

```python
from agentbox import Agent

agent = Agent(
    model="...",
    base_url="...",
    tools="structured",
    custom_tools=[SearchRepoTool(), count_lines],
)

# Custom only
agent = Agent(model="...", tools="custom", custom_tools=[SearchRepoTool()], builtins=False)
```

## OpenAI schema

Each tool exposes:

```json
{
  "type": "function",
  "function": {
    "name": "edit_file",
    "description": "...",
    "parameters": { "type": "object", "properties": { ... }, "required": [...] }
  }
}
```

This list is passed as `tools=` to Chat Completions.

## ToolRegistry

```python
from agentbox.tools import build_tool_registry
from agentbox.types import ToolMode

reg = build_tool_registry(
    ToolMode.STRUCTURED,
    custom_tools=[...],
    include_builtins=True,
    drop=["run_tests"],
    drop_prob=0.1,
    rng=random.Random(42),
)
schemas = reg.openai_tools()
```

## ToolExecutor

Maps model `tool_calls` → tool messages:

1. Parse `function.arguments` JSON  
2. Look up tool by name  
3. `await tool.execute(sandbox, **args)`  
4. Return `Message(role=tool, tool_call_id=..., content=...)`  

Unknown tools / bad JSON / exceptions become content starting with `ERROR:` —
the agent loop continues.

Parallel: `execute_many(..., parallel=True)` uses `asyncio.gather` and preserves
call order in results.

## ToolContext

Internal context for executor bookkeeping:

- `sandbox`, `manager`, `workspace_dir`
- `step`, `run_id`, `task_id`

Custom tools receive the **sandbox facade**, not the full `ToolContext`.

## Output truncation

| Source | Limit (approx.) |
| --- | --- |
| `read_file` | 200_000 chars + notice |
| `run_command` stdout/stderr | 50_000 chars each |

Prevents context blow-ups from runaway command output.
