# Development

Contributing and extending AgentBox.

## Install (dev)

```bash
git clone https://github.com/konic-labs/agentbox.git
cd agentbox
uv venv .venv --seed --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional: `.[rich]`, `.[art]`, `.[generate]`.

## Tests

```bash
pytest tests/unit -q
pytest tests/integration -q -m docker
```

| Suite | Docker | Focus |
| --- | --- | --- |
| `tests/unit` | No | Paths, tools, loop, schema, ART dict, filters, metrics |
| `tests/integration` | Yes | Sandbox I/O, mock rollouts, parallel |

Mark: `@pytest.mark.docker`. Skip when daemon unavailable.

Hermetic preference: use `MockModelClient` for control-flow tests.

## Layout conventions

- Package under `src/agentbox/`  
- Public re-exports in `agentbox/__init__.py` stay small  
- New export formats → `trajectory/formats/`  
- New tools → `tools/` + registry, not hard-coded in the loop  
- Docs: one concern per file under `docs/` (this tree)  

## Extension checklist

### New builtin tool

1. Implement `BaseTool` in `tools/`  
2. Register in `tools/builtins.default_tools`  
3. Add unit tests (schema + behavior with fake sandbox)  
4. Document in [Tools](tools.md)  

### New custom tool (user code)

No core changes — use `@tool` / `BaseTool` and `Agent(custom_tools=[...])`.

### New model backend

Implement `ModelClient` with OpenAI-shaped `complete`. Prefer wrapping
`AsyncOpenAI` unless the protocol truly differs.

### New trajectory format

Add `trajectory/formats/foo.py` and re-export if public.

## Commits

This repository uses **Conventional Commits** (`feat:`, `fix:`, `docs:`,
`test:`, `chore:`, …) with optional scopes (`sandbox`, `tools`, `agent`, …).

## Packaging

- Build backend: hatchling  
- `requires-python >= 3.11`  
- License: Apache-2.0  

```bash
uv build
```

## Docs maintenance

- No “phase” or “roadmap” language in user-facing docs  
- Link rather than copy large examples  
- Keep [docs/index.md](index.md) map updated when adding files  

## Related

- [Architecture](architecture.md)  
- [Setup](setup.md)  
