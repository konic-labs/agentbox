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
python scripts/check_docs.py              # links, anchors and images in the README and docs
```

CI (`.github/workflows/ci.yml`) runs lint, types, the reference-docs check and the docs check;
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

## Docs images

Every image the README and the docs show lives in `docs/assets/readme/`, and
`scripts/art/art.py` draws them all:

```bash
pip install fonttools
python scripts/art/art.py      # fetches the fonts once, writes docs/assets/readme/*.svg
```

The art comes from the AgentBox release film. A fog field sits under a
halftone screen. A pale chamber band opens for one rollout and prints its
container in Space Mono, then the exit code gives the verdict. Finally the
band shuts into a line of red dots: the trajectory. Konic red is the only
accent.

Each SVG stands alone. Its text is outlined from Inter Tight and Space Mono,
which the script fetches at pinned commits and checks by SHA-256. A file has
no scripts and no outside references. Both themes are inside the file and
follow the reader's system theme (`prefers-color-scheme`). Animations stop for
readers who ask for reduced motion. Without animation, each element shows its
final frame.

`training.svg` is drawn from `scripts/art/training.json`. `scripts/art/curve.py`
builds that file from real runs: verl console logs and an
`agentbox --json bench compare` report. `scripts/art/render.py` renders PNG
previews in both themes for a visual check, in the Playwright container:

```bash
docker run --rm -v "$PWD":/work -w /work mcr.microsoft.com/playwright/python:v1.64.1-noble \
  bash -c "pip install -q playwright==1.64.1 && python scripts/art/render.py docs/assets/readme /tmp/png"
```

`python scripts/art/art.py --og og/og-image.svg` also writes the social
preview (1280 × 640). Render it with `render.py og /tmp/png`, and save the dark
one as `.github/og-image.png`. GitHub has no API for the social preview, so
upload it under Settings → General → Social preview.

Show an image with a plain `<img>` or a Markdown image, never inside a
`<picture>`, because GitHub's mobile apps drop a `<picture>` and its image.
Give it alt text that says what it shows (`alt=""` for a decorative one).
`python scripts/check_docs.py` checks this, along with every relative link and
anchor, image sizes (under 300 KB), and that every file in `docs/assets/` is
shown by some page. CI runs it.

## Commits and releases

- [Conventional Commits](https://www.conventionalcommits.org/) with scopes
  (`sandbox`, `agent`, `tools`, `model`, `runner`, `tasks`, `rewards`,
  `integrations`, `cli`, `bench`, `generate`, `server`, `docs`, `ci`).
  Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- The version lives in `src/agentbox/version.py`; the build backend is
  hatchling. Update `CHANGELOG.md`, then `uv build` and `twine check --strict dist/*`.
