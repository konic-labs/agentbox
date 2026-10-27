# Changelog

## 0.3.0 — trainer-agnostic RL infrastructure

AgentBox is no longer ART-first: one sandbox, tool, reward and trajectory
stack serves TRL, verl, OpenRLHF, SkyRL, ART and custom trainers. See
[docs/migration.md](docs/migration.md) for breaking changes.

### Trainer integrations
- Step-wise environment API: `AgentBoxEnv` / `SyncAgentBoxEnv` (reset, step,
  call_tool, evaluate, trajectory) and `RemoteEnv` over HTTP
- TRL `environment_factory` adapter + datasets; `agentbox train trl`;
  infrastructure failures return `None` (TRL's unscorable samples)
- verl `AgentBoxAgentLoop` (ToolAgentLoop subclass) + parquet datasets;
  zeroed response masks for infrastructure failures
- OpenRLHF agent executor running token-in/token-out on OpenRLHF's vLLM engine
- SkyRL `skyrl_gym` text environment (Hermes protocol)
- ART adapter: `to_art_dict` / `to_art_trajectory` / `to_art_groups`; trainable
  `Choice`s with vLLM token ids and logprobs (validated against ART's tokenizer)
- Exporters: chat, sharegpt, dpo, grpo (`--advantage grpo|drgrpo|rloo`), tokens,
  hf (Parquet / Hub), art
- RL utilities: unbiased pass@k, GRPO / Dr. GRPO / RLOO advantages,
  zero-variance groups, `agentbox task select` by observed pass rate

### Exact token data
- `capture_tokens` / `capture_logprobs`: per-turn prompt/completion token ids
  and sampler logprobs, stored with shared prefixes once (3–5x smaller)
- TITO (token-in/token-out) client: one prefix-consistent sequence per
  trajectory (`agentbox run --tito TOKENIZER`)

### Sandbox runtime
- New Docker backend: one shared client, bounded daemon thread pools,
  streaming exec with head+tail output caps, kill-on-timeout of the whole
  process tree, single-call teardown (5.11 s → 0.08 s)
- Hardened defaults: `cap_drop: ALL` + minimal set, `no-new-privileges`,
  init process, PID/memory/CPU limits, optional gVisor runtime
- Cached derived images (pytest once per image, setup-command snapshots),
  per-task environments, warm container pool, orphan-safe pruning,
  `keep_on_failure` that keeps only failures, interactive `agentbox sandbox`

### Agent, tools, models
- Ordered tool execution (read-only calls may run concurrently), no dangling
  tool calls, deadline-aware steps, context exhaustion as a policy failure
- Hermes text protocol; salvage of `<tool_call>` text the server failed to
  parse (#96); `AgentHooks` (before/after model, after tools, message injection)
- Tools: ranged `read_file`, `search_files`, `apply_patch`, `submit`,
  `edit_file(replace_all)`; `@tool` with pydantic schemas, docstring args,
  sync functions and `ToolContext`
- Model client: typed retries with jitter and Retry-After, adaptive
  max_tokens, reasoning fields, sampling passthrough, client reuse; mock,
  function and replay clients

### Tasks and rewards
- Task schema v2 (prompt templates, binary files, environment, per-task
  budgets, split, group key); lazy loaders for dirs, JSONL(.gz) and Parquet
- Tamper-resistant verification: hidden tests, protected files, forbidden
  files (planted `conftest.py`), hardened pytest; partial credit
  (`pytest_fraction`, stdout, custom)
- Composable reward components (penalties, format reward, LLM judge, any
  `module:function`) with per-component breakdowns
- `failure_kind` on every rollout: infrastructure failures are retried and
  never scored as policy failures

### Runner and trajectories
- Streaming `ParallelRunner` (`stream`, `stream_groups`, `run_to_sink`) with
  bounded memory, separate model concurrency, infrastructure retries, resume
- Trajectory schema v2 (rollout/group ids, verify info, reward breakdown,
  failure kind, task hash) with v1 upgrade; JSONL(.gz) sinks; `traj
  stats/filter/validate/replay/diff/render/export`

### CLI, server, benchmarks, generation
- Typer CLI rebuilt: global `--config/--profile/--json/-v/--log-format`, exit
  codes 0/1/2/3, project profiles with `${VAR:-default}` interpolation,
  `doctor`, `init`, `config`, `integrations`; legacy commands kept as hidden aliases
- `agentbox serve`: HTTP environment server (sessions, bearer auth, limits,
  idle reaper, metrics)
- Benchmarks: hash v2 over fairness-critical fields, streaming reports,
  resume, unbiased pass@k, paired comparison with bootstrap CI and sign test,
  `--fail-on-regression` CI gating
- Generation: client reuse and cleanup, O(n) near-duplicate index, atomic
  resume state, concurrent validators

### Quality
- ruff, ruff format and mypy --strict clean; CI with lint/types, unit tests on
  Python 3.11–3.13, Docker integration tests, CLI smoke tests and build checks
- Every config field described; CLI and configuration references generated
  from code (`scripts/gen_reference_docs.py`)
- Documentation rewritten for the new architecture, with integration guides

## 0.2.0 — production hardening (lab)

### Added
- **LLM task judge** (`validate_task_llm`, DSPy `ValidateCodingTask`) as generation QC
- **Static QC** (`validate_task_static`) for stubs, asserts, leak patterns, path consistency
- **Two-stage generation** (`--two-stage`): golden solution → AST stub strip → starter-fail + solution-pass Docker QC
- **Batch generate API** (`batch_generate`, `BatchGenerateConfig`) with resume, dedup, review queue
- **Dedup** (`task_signature_hash`, Jaccard near-dup) and difficulty heuristics
- **Review queue** export/import (`review_queue.jsonl` / decisions)
- **CLI** `agentbox generate` (`one`, `batch`, `validate-llm`, `validate-docker`)
- **CLI** `agentbox traj` (`show`, `render`) HTML dashboard (official vs agent self-check)
- **Doctor model probe** (`--model` / `--base-url` tool-calling check)
- **Config load** (`agentbox.yaml` / env interpolation; used by generate + bench)
- **Bench multi-student** (`--student id=model@url`, `--limit`, `--probe`, `--disable-thinking`)
- **Richer** `bench show --out` (task×model matrix, failure clusters)
- **Jobs** types + local artifact store + validation cache
- **HTTP retries** on OpenAI-compatible client (429/5xx)
- Official **verify_stdout/stderr** on trajectory metadata
- Docs: ART GRPO recipe, tool-calling matrix; example `examples/agentbox.yaml`
- CI: unit + hermetic smoke; optional provider doctor job

### Changed
- Stub-first generation prompts; concurrent DSPy uses `dspy.context`
- Generation defaults favor structured JSON + optional LLM/static QC
- Package version **0.2.0**

## 0.1.0
- Initial ART-native rollout engine, benches, DSPy generate + Docker QC
