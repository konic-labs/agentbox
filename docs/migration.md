# Migrating from 0.2 to 0.3

0.3 makes AgentBox trainer-agnostic and rebuilds the runtime for RL-scale
throughput. Most code that used `Rollout.run`, `ParallelRunner`, `Task` and
`Trajectory` keeps working. These are the changes to check.

## ART is an adapter, not the core

| 0.2 | 0.3 |
| --- | --- |
| `traj.to_art_dict()` | `agentbox.integrations.art.to_art_dict(traj)` |
| `traj.to_art()` | `agentbox.integrations.art.to_art_trajectory(traj)` |
| grouping by hand | `agentbox.integrations.art.to_art_groups(trajs)` (drops infrastructure failures) |
| `agentbox export --format art` | `agentbox traj export -f art` |

The old methods still work and emit a `DeprecationWarning`. `metrics.correct`
now comes from the verifier, metadata is scalarised, and assistant turns with
captured tokens become trainable `Choice`s.

## CLI

| 0.2 | 0.3 |
| --- | --- |
| `agentbox run-dir DIR` | `agentbox run DIR` (also files, JSONL(.gz), Parquet) |
| `agentbox export ...` | `agentbox traj export ...` |
| `agentbox prune` | `agentbox sandbox prune` (orphans only by default; `--all` for everything) |
| `agentbox build-image --path docker/python-sandbox` | `agentbox image build` (the Dockerfile ships with the package; `--dockerfile` for your own) |
| `--api-key VALUE` | `--api-key-env NAME`, or `api_key_env` in a profile |
| silent mock fallback | `--mock` must be explicit |

The old commands are hidden aliases for now. Exit codes are 0 / 1 / 2 / 3
([CLI](cli.md)), and `--json` works on every command.

## Trajectories (schema v2)

- New top-level fields: `rollout_id`, `group_id`, `sample_index`,
  `failure_kind`, `verify`, `reward_breakdown`, `task_hash`, `tool_mode`,
  `stop_reason`.
- `metadata` no longer carries `verify_*`, `stop_reason` or `failure_kind`. Read
  `traj.verify`, `traj.stop_reason` and `traj.failure_kind`.
- v1 files are upgraded on load (`Trajectory.load`, `iter_trajectories`).
- Runs stream to `trajectories.jsonl(.gz)` instead of one pretty-printed JSON
  file per rollout. `DirectorySink` still writes per-file output.

## Failure attribution

Infrastructure failures (sandbox, seed, model API, verifier) are no longer
scored as reward 0. They carry a `failure_kind`, are retried by the runner,
and are excluded from exports and from trainer rewards. If you computed
success rates from `final_status` only, filter on `traj.is_trainable` to
reproduce policy-only numbers.

## Agent loop and tools

- Tool calls of one turn run **in order**. Only consecutive read-only calls
  run concurrently (`parallel_tool_calls`).
- Every tool call gets a tool message, even on timeouts: no dangling calls.
- Context overflow is `stop_reason=context_length` and a **policy** failure,
  not an error.
- `Message.to_openai_dict()` omits reasoning unless `include_reasoning=True`.
- `ToolContext` is a dataclass and is available through `get_tool_context()`.
- `before_model` hooks get a copy of the messages; return messages to inject
  them.
- New tools in `extended` mode: `search_files`, `apply_patch`, `submit`.
  `read_file` gained `offset` / `limit` / `line_numbers`, and `edit_file`
  gained `replace_all`.

## Sandboxes

- String commands run with `bash -c` (or `sh -c`), not a login shell, so the
  image's `PATH` is no longer reset by `/etc/profile`.
- Containers drop all capabilities except a small set, use
  `no-new-privileges` and run an init process ([security](security.md)).
- `prune_agentbox_containers()` removes only orphans unless `scope="all"`.
- The `agentbox.sandbox.docker_backend` helpers were replaced by
  `agentbox.sandbox.backend`; a compatibility shim remains.
- `keep_on_failure` keeps only failed rollouts' sandboxes; `auto_remove=False`
  keeps all of them.

## Tasks and rewards

- `load_task(dir)` now reads the sibling `files/` directory, so directory
  tasks no longer lose their starter files.
- `Verifier.verify(sandbox, spec, task)` takes the task (for tamper checks).
  `VerifierSpec` normalises `shell` → `command` and `test` → `pytest`.
- `RolloutConfig.rewards` (or the `rewards:` section) replaces ad-hoc
  shaping. Without it, the reward is `verifier` plus the legacy
  `AgentConfig` penalties, so old configurations score the same.

## Configuration

- `RolloutConfig.save_dir` and `RolloutConfig.trajectory_format` were never
  read and are removed. Use sinks or `agentbox run --out`.
- `agentbox.yaml` has named profiles (`default_model` + `models`).
  `${VAR}` without a default fails when the variable is unset, instead of
  becoming an empty string. Unknown top-level keys are reported
  (`config validate --strict` fails on them).
- `pyyaml` is a core dependency. The extras are `rich`, `hf`, `trl`,
  `server`, `art`, `skyrl`, `generate` and `all`.

## Benchmarks

Suite content hashes moved to version 2 (fairness-critical fields only). Run
`agentbox bench freeze SUITE` once per suite. pass@k is now the unbiased
estimator; the old "any success among n" number is reported as pass@n.
