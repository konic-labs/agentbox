# AgentBox

Fully local, open-source framework for running LLM agents in isolated Docker sandboxes, collecting multi-turn trajectories for offline datasets and online RL (ART / GRPO).

> Status: early development (v0.1). See the implementation roadmap in the repo plan.

## Features (MVP)

- One Docker container per rollout
- Structured tools + pure shell mode
- Custom tools (`BaseTool` / `@tool`)
- OpenAI-compatible model endpoints (local or remote)
- Task seeding (`starter_files` + `setup_commands`) + verifiers
- Trajectory export (JSON + ART-compatible dict)

## Install

```bash
# Requires Python 3.11+ and Docker
uv pip install -e ".[dev]"
```

## Quick start

Coming soon — see `examples/` after the first MVP slices land.

## License

Apache-2.0
