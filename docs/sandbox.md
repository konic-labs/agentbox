# Sandbox

The sandbox layer owns Docker container lifecycle and all in-container I/O.

## Components

| Type | Role |
| --- | --- |
| `SandboxManager` | Create, exec, read/write files, destroy |
| `Sandbox` | Handle + thin async facade for tools (`exec`, `read_text`, `write_text`, `list_dir`) |
| `ExecResult` | `exit_code`, `stdout`, `stderr`, `duration_s`, `timed_out` |
| `docker_backend` | Sync docker-py helpers (always via `asyncio.to_thread`) |
| `paths` | Workspace path jail |
| `prune` | Remove labeled orphans |
| `snapshot` | `docker commit` image from a live sandbox |
| `images` | Named image presets |

## Lifecycle

```txt
ensure_image (optional pull)
  -> containers.create + start (command=sleep infinity)
  -> optional ensure_pytest
  -> mkdir workspace
  -> [seed / agent / verify]
  -> stop + remove (force)
```

Containers are labeled:

- `agentbox=1`
- `agentbox.task_id`
- `agentbox.run_id`

Name pattern: `agentbox-{run8}-{task_slug}-{rand}` (DNS-safe, ≤63 chars).

## Path jail

All tool and seed paths resolve under `workspace_dir` (default `/workspace`).

Rejected:

- `..` escapes past workspace
- Absolute paths outside workspace (e.g. `/etc/passwd`)

Allowed:

- Relative paths (`src/main.py`)
- Absolute paths still under workspace (`/workspace/x.py`)

Implementation: pure logical posix resolve (`agentbox.sandbox.paths`), not host
`Path.resolve()` against the host filesystem.

## File I/O

| API | Mechanism |
| --- | --- |
| `write_file` / `write_files` | In-memory tar + `put_archive` |
| `read_file` | `get_archive` + untar |
| `list_dir` | `ls` / `find` via exec |

Parent directories are created with `mkdir -p` before writes.

## Exec

```python
result = await manager.exec(sandbox, "python -c 'print(1)'", timeout_s=30)
# or list argv
result = await manager.exec(sandbox, ["python", "-m", "pytest", "-q"])
```

String commands run as `["/bin/sh", "-lc", command]`. Timeouts return
`ExecResult(timed_out=True, exit_code=-1)` rather than raising by default.

## Resource limits

Applied at create time via docker-py kwargs:

- Memory / memswap
- `nano_cpus`
- `pids_limit`
- `network_mode=none` when `network_disabled=True`

See [Configuration](configuration.md).

## Prune

```python
from agentbox.sandbox import prune_agentbox_containers
removed_ids = prune_agentbox_containers()
```

CLI: `agentbox prune` / `agentbox doctor --prune`.

## Snapshots

```python
from agentbox.sandbox import commit_sandbox
image_tag = await commit_sandbox(sandbox, repository="agentbox/snap", tag="task1")
# e.g. agentbox/snap:task1
```

Useful for debugging a mid-rollout filesystem or building a derived base image.
Does not replace fresh containers for training rollouts.

## Image presets

```python
from agentbox.sandbox.images import sandbox_config_for_preset

cfg = sandbox_config_for_preset("python")  # python:3.12-slim-bookworm
cfg = sandbox_config_for_preset("node")    # node:22-bookworm-slim
cfg = sandbox_config_for_preset("go")      # golang:1.22-bookworm
cfg = sandbox_config_for_preset("baked")   # agentbox/sandbox:latest
```

Non-Python presets set `ensure_pytest=False`.

## Security notes

- Never mount the host Docker socket into a sandbox.
- Prefer `network_disabled=True` unless installs require network.
- Root-in-container still trusts the Docker/kernel TCB; acceptable for local research sandboxes.

More: [Security](security.md).
