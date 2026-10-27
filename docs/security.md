# Security

AgentBox runs untrusted, model-generated actions. The design goal is that a
misbehaving or adversarial policy can neither damage the host nor game its
reward. Docker (or gVisor, see below) is the isolation boundary.

## Threat model

| In scope | Out of scope |
| --- | --- |
| Agent tools touching host paths | A malicious Docker daemon or kernel exploit |
| Runaway CPU, memory, processes, output, time | Multi-tenant isolation between mutually hostile users |
| Privilege escalation inside the container | Supply-chain attacks through packages installed by `setup_commands` with network on |
| Reward hacking: editing tests, planting `conftest.py`, exploiting the verifier environment | Leaks through the model provider (prompts and outputs go to the endpoint you configure) |
| Leftover containers, leaked secrets in logs | |

## Isolation defaults

| Control | Default |
| --- | --- |
| Where tools run | Inside the container only; the host filesystem is never mounted |
| Paths | Resolved against the workspace; `..` and absolute paths outside it are rejected; symlinks are followed only inside the workspace |
| Network | `network_mode=none` (`limits.network_disabled`) |
| Capabilities | `cap_drop: [ALL]`, then CHOWN, DAC_OVERRIDE, FOWNER, FSETID, KILL, SETGID, SETUID, NET_BIND_SERVICE added back |
| Privileges | `no-new-privileges`; never `--privileged`; the Docker socket is never mounted |
| PID 1 | Init process (reaps zombies, forwards signals) |
| Resources | CPU, memory (no swap), PIDs; optional nofile, `/dev/shm`, read-only root with tmpfs |
| Commands | Timeouts kill the whole process tree; output beyond 64 MiB kills the command |
| User | The image's default; set `security.user: "1000:1000"` to run unprivileged |

Stronger isolation: `security.runtime: runsc` runs sandboxes under gVisor
(install `runsc` and register it with Docker). Combine it with
`read_only_rootfs: true` plus `tmpfs` mounts for tasks that need no writable
system paths.

## Reward integrity

The verifier is the reward. AgentBox assumes the policy will attack it:

- **Hidden tests** (`hidden_files`) are written only after the agent finishes.
- **Protected files** (`protected_paths`) are restored before verification,
  and any modification is recorded in `verify.tampered`.
- **Forbidden files** (`forbid_new_files`, e.g. `PYTEST_TAMPER_FILES`:
  `conftest.py`, `sitecustomize.py`, `pytest.ini`, ...) are deleted if created
  and recorded.
- **Hardened pytest** (`hardened: true`): AgentBox sets `PYTEST_ADDOPTS`
  itself, overriding anything the agent exported, disables user site-packages
  and pytest's cache provider, and reads results from a JUnit file at a random
  path.
- With `on_tamper: fail`, a tampered rollout is not a success and scores
  `tamper_reward`.

See [tasks](tasks.md#reward-hacking-protection). `agentbox task check` should
pass on every task before it is used for training.

## Secrets

- Give keys by reference: `api_key_env: NAME` in profiles, and `--api-key-env`
  on the CLI. Keys are redacted from `config show`, benchmark reports and
  trajectories, and are never sent into sandboxes.
- Do not put secrets in `starter_files`, `setup_commands`, `SandboxConfig.env`
  or task metadata. All of these are visible to the agent or recorded.
- The HTTP server's bearer token is compared in constant time. Put the server
  behind TLS outside trusted networks.
- Treat trajectories as sensitive. They contain code, tool output and model
  text.

## Operations

- `agentbox sandbox prune` removes orphaned sandboxes (owner process gone).
  Other processes' live sandboxes are never touched.
- `agentbox cache prune` bounds disk use by cached images. The cache is also
  LRU-bounded (`cache_max_images`).
- Keep `network_disabled: true` for rollouts. Cached image builds may use the
  network (`prepare_network`), which keeps package installs out of the
  episode's attack surface.
- QC generated tasks (`agentbox task lint` / `check`) before use, because
  their `setup_commands` run in your sandboxes.
