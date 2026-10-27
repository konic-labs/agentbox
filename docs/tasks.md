# Tasks

A task is everything one episode needs: starter files, optional setup, the
instruction, and a verifier that turns the final workspace into a reward.

## Formats

| Layout | Notes |
| --- | --- |
| `my_task/task.json` + `my_task/files/**` | Directory form: files under `files/` become starter files (binary files are base64-encoded automatically) |
| `my_task.json` | Inline `starter_files` (and `binary_files`) |
| `tasks.jsonl` / `tasks.jsonl.gz` | One task per line, streamed |
| `tasks.parquet` | One task per row (`pip install "agentbox-rl[hf]"`) |
| a directory of any of the above | Scanned recursively |

```python
from agentbox import iter_tasks, load_task

for task in iter_tasks(["tasks/", "more.jsonl.gz"], split="train"):   # lazy
    ...
task = load_task("tasks/example_add")        # one task (dir or file)
```

`iter_tasks` reads lazily. Duplicate `task_id`s are reported and skipped
(`dedupe=True`), and invalid entries are logged and skipped unless
`strict=True`.

## A complete task

```json
{
  "task_id": "fix_add_001",
  "description": "add(a, b) in calc.py returns the wrong value. Fix it so the tests pass.",
  "prompt": "{description}\n\nFiles in the workspace:\n{files}",
  "starter_files": {
    "calc.py": "def add(a, b):\n    return a - b\n",
    "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
  },
  "setup_commands": ["pip install -q numpy"],
  "verifier": {
    "type": "pytest",
    "command": "python -m pytest -q",
    "timeout_s": 120,
    "reward_mode": "pytest_fraction",
    "protected_paths": ["test_*.py"],
    "forbid_new_files": ["conftest.py", "*/conftest.py", "pytest.ini", "sitecustomize.py"],
    "hidden_files": {"test_hidden.py": "from calc import add\n\ndef test_neg():\n    assert add(-1, 1) == 0\n"}
  },
  "environment": {"memory_mb": 1024},
  "max_steps": 20,
  "split": "train",
  "metadata": {"difficulty": "easy", "source": "handwritten"}
}
```

Every field is described in the [configuration reference](config-reference.md#task).
The most important ones:

| Field | Purpose |
| --- | --- |
| `description` / `prompt` | The instruction. `prompt` is an optional template with `{description}` and `{files}` |
| `starter_files`, `binary_files` | Workspace contents (text; base64 for binary) |
| `setup_commands` | Run after seeding and before the agent; snapshotted into a cached image ([sandbox](sandbox.md#cached-derived-images)) |
| `verifier` | How the outcome is scored (below) |
| `environment` | Per-task image, environment variables, network, CPU, memory, working directory |
| `max_steps`, `episode_timeout_s`, `allowed_tools`, `system_prompt_extra` | Per-task agent overrides |
| `split`, `group_key`, `version`, `metadata` | Dataset bookkeeping, copied into trajectories (`version` is part of `task_hash`) |

## Verifiers

The verifier runs **after** the agent finishes, in the same sandbox.

| `type` | Runs |
| --- | --- |
| `pytest` (alias `test`) | `command`, default `python -m pytest -q [path]` |
| `command` (alias `shell`) | `command` (required) |
| `custom` | `command` and/or `reward_fn` |

Success means the exit code equals `success_exit_code` (0) and nothing timed
out, unless `success_threshold` is set, in which case success means
`reward >= success_threshold`. `pre_commands`, `env` and `workdir` customise how
the verifier is run.

### Reward modes

| `reward_mode` | Reward |
| --- | --- |
| `binary` (default) | `reward_success` (1.0) or `reward_failure` (0.0) |
| `pytest_fraction` | Linear in passed / (total − skipped), read from a JUnit XML report AgentBox requests (falls back to pytest's summary line) |
| `stdout` | Last non-empty stdout line, either a number or JSON `{"reward": x, ...}`; extra keys go to `verify.details`. Otherwise the binary reward |
| `custom` | `reward_fn: "package.module:function"`, called as `fn(result=ExecResult, task=Task, spec=VerifierSpec)`; may be async |

## Reward-hacking protection

Agents optimise whatever the verifier measures. Without protection, an agent
can pass any test suite by rewriting the tests or planting a `conftest.py` that
skips them. The integration test suite demonstrates both. AgentBox defends at
verification time:

| Defence | Field | What happens before the verifier runs |
| --- | --- | --- |
| Hidden tests | `hidden_files` | Written into the workspace; the agent never sees them |
| Protected files | `protected_paths` (globs over starter files) | Restored to their original content; edits are recorded |
| Forbidden files | `forbid_new_files` (globs, e.g. `PYTEST_TAMPER_FILES`) | Deleted if the agent created them; recorded |
| Hardened pytest | `hardened` (default true) | Fresh `PYTEST_ADDOPTS` (overrides anything the agent exported), no user site-packages, cache provider disabled |

Recorded tampering appears in `verify.tampered`. With `on_tamper: fail`
(default) the rollout is not a success and scores `tamper_reward` (default
`reward_failure`). `on_tamper: flag` only records it, and the `tamper_penalty`
reward component ([rewards](rewards.md)) can then shape it.

`PYTEST_TAMPER_FILES` (in `agentbox.tasks.schema`) lists the usual suspects:
`conftest.py`, `*/conftest.py`, `sitecustomize.py`, `usercustomize.py`,
`pytest.ini`, `*/pytest.ini`, `tox.ini` and `setup.cfg`.

## Authoring and QC

```bash
agentbox task new tasks/my_task --id my_task_001   # task.json + files/ with protection on
agentbox task validate tasks/                      # schema (any invalid task fails)
agentbox task lint tasks/ --min-asserts 3          # static QC: stub-only starters, assert count, spoilers, paths
agentbox task check tasks/ --solution golden/      # Docker QC: starter must FAIL; golden files must PASS
agentbox --json task ls tasks/                     # inventory (machine-readable)
agentbox task show tasks/my_task
agentbox task schema > task.schema.json            # JSON Schema for editors and external validators
```

`task check` is the most important gate for RL data. A task whose starter
already passes gives no signal, and a task whose golden solution fails
punishes correct behaviour.

## Identity and hashing

`task_hash` (stored on every trajectory) is a hash of the complete task, so
trajectories can be matched to the exact task revision they came from.
Benchmark suites hash tasks together with fairness-critical sandbox and agent
settings ([benchmarks](benchmarks.md)).
