# Environment server

`agentbox serve` exposes AgentBox environments over HTTP. Trainers on GPU
nodes can then drive sandboxes that live on CPU or Docker hosts, and several
trainers or evaluators can share one pool of sandbox machines.

```bash
pip install "agentbox-rl[server]"
export AGENTBOX_SERVER_TOKEN=$(openssl rand -hex 16)
agentbox serve --host 0.0.0.0 --port 8765 --tasks tasks/ --max-sessions 128 --idle-ttl 1800
```

The server reads the sandbox, agent and reward sections of `agentbox.yaml`.
`--tasks` loads a task source so clients can refer to tasks by id. With a
token set (`--token-env NAME`, default `AGENTBOX_SERVER_TOKEN`), every
`/v1/*` request needs `Authorization: Bearer <token>`.

## Client

`RemoteEnv` has the same surface as `AgentBoxEnv`:

```python
from agentbox.env.remote import RemoteEnv

env = RemoteEnv("http://sandbox-host:8765")      # token from $AGENTBOX_SERVER_TOKEN or token=
obs = await env.reset("fix_add_001")             # a task id known to the server, or a Task object
step = await env.step(assistant_message)         # text, message dict or tool calls
out = await env.call_tool("read_file", {"path": "calc.py"})
result = await env.evaluate()
traj = await env.trajectory()
await env.close()                                # deletes the session, closes the HTTP client
```

## API

| Method | Path | Body / result |
| --- | --- | --- |
| `POST` | `/v1/sessions` | `{task | task_id, group_id?, sample_index?, seed?, agent?}` → `{session_id, observation}` (`agent` overrides fields of the server's agent config) |
| `POST` | `/v1/sessions/{id}/step` | `{action: text | message | tool_calls}` → `StepResult` |
| `POST` | `/v1/sessions/{id}/tool` | `{name, arguments}` → tool outcome |
| `POST` | `/v1/sessions/{id}/evaluate` | → `EvalResult` (reward, breakdown, success, failure kind, verifier result) |
| `GET` | `/v1/sessions/{id}/trajectory` | → `Trajectory` JSON |
| `DELETE` | `/v1/sessions/{id}` | removes the sandbox |
| `GET` | `/v1/tasks` | task ids from `--tasks` |
| `GET` | `/healthz` | `{ok, sessions, tasks}` (no auth) |
| `GET` | `/metrics` | Prometheus text: session and request counters (no auth) |

## Errors and limits

- Creating a session beyond `--max-sessions` returns **429**.
- A provisioning failure returns **503** with
  `{"failure_kind": "sandbox" | "seed" | "setup_check", "error": ...}`, so
  clients can retry or skip without scoring it as a policy failure.
- An unknown session returns 404, and requests to one session are serialised.
- Sessions idle for longer than `--idle-ttl` seconds are reaped and their
  sandboxes removed, as are all sessions on shutdown.

Run the server behind TLS (a reverse proxy) when it leaves a trusted network.
Sandbox isolation on the server host is exactly as described in
[security](security.md).
