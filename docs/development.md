# Development

## Setup

```bash
git clone https://github.com/konic-labs/agentbox.git && cd agentbox
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"                 # + ".[hf,trl,server,generate]" as needed
```

## Checks

```bash
ruff check src tests examples scripts
ruff format --check src tests examples scripts
mypy src                                   # strict
pytest tests/unit -q                       # no Docker, a few seconds
pytest tests/integration -m docker -q      # real sandboxes, about 40 s
bash scripts/ci_smoke.sh --docker          # every CLI workflow with the mock model
python scripts/gen_reference_docs.py       # regenerate docs/cli-reference.md and docs/config-reference.md
```

CI (`.github/workflows/ci.yml`) runs lint, types and the reference-docs check;
unit tests on Python 3.11, 3.12 and 3.13; Docker integration tests plus the
Docker smoke test; and a build with `twine check --strict` and a clean wheel
install.

## Tests

| Directory | Needs | Covers |
| --- | --- | --- |
| `tests/unit` | nothing | Loop semantics (ordering, deadlines, protocols, salvage, hooks), tools with a fake sandbox, model client retries and token capture, TITO, rewards, exporters, adapters with fake trainers, CLI via `CliRunner`, warm pool bookkeeping |
| `tests/integration` (`@pytest.mark.docker`) | Docker | Sandbox I/O, timeouts and output caps, hostile paths, security defaults, verifier integrity (tampering, hidden tests, partial credit), caches, warm pool, kept sandboxes, 32-sandbox stress run, CLI and server end to end, TRL/SkyRL/OpenRLHF adapter episodes |

Docker tests skip when no daemon is reachable. Prefer `MockModelClient` /
`FunctionModelClient` to drive the loop deterministically; a shared mock plays
its script per conversation, so it is safe under concurrency.

## Extending

| To add | Do |
| --- | --- |
| A builtin tool | Subclass `BaseTool` in `tools/`, set `read_only`/`terminal`, add it to `tools/builtins.py` and a mode, test it with the fake sandbox and in Docker |
| A reward component | `@register_reward("name")` factory in `rewards.py` (or anywhere, as a user) |
| An exporter | `@register_exporter("name")` in `export/core.py` |
| A model client | Implement `complete(messages, *, tools=None, tool_choice="auto", **kw) -> ModelResponse` |
| A trainer integration | A module in `integrations/` that imports its framework lazily, plus an entry in `integrations/__init__.py` (`agentbox integrations` lists it) |
| A CLI command | A Typer command in `cli/`, then regenerate the reference docs |

Every config field needs a `Field(description=...)`: the reference docs and
`agentbox config show` are generated from them.

## Commits and releases

- [Conventional Commits](https://www.conventionalcommits.org/) with scopes
  (`sandbox`, `agent`, `tools`, `model`, `runner`, `tasks`, `rewards`,
  `integrations`, `cli`, `bench`, `generate`, `server`, `docs`, `ci`).
  Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- The version lives in `src/agentbox/version.py`; the build backend is
  hatchling. Update `CHANGELOG.md`, then `uv build` and `twine check --strict dist/*`.
