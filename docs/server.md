# Environment server

`agentbox serve` exposes AgentBox environments over HTTP. Trainers on GPU
nodes can then drive sandboxes that live on CPU or Docker hosts, and several
trainers or evaluators can share one pool of sandbox machines.

```bash
pip install "agentbox-rl[server]"
export AGENTBOX_SERVER_TOKEN=$(openssl rand -hex 16)
agentbox serve --host 0.0.0.0 --port 8765 --tasks tasks/ --max-sessions 128 --idle-ttl 1800
```

The server reads the sandbox, agent and reward sections of `agentbox.yaml`
(`agentbox -C FILE serve` reads another file). `--tasks` loads a task source so
clients can refer to tasks by id. With a token set (`--token-env NAME`, default
`AGENTBOX_SERVER_TOKEN`), every `/v1/*` request needs
`Authorization: Bearer <token>`.

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

`RemoteEnv(url, agent=AgentConfig(...), rewards=[...])` sends the agent config
and reward components with every session, so the client's settings apply on
the server. `RemoteEnv.from_config(RemoteConfig(url=...))` shares one pooled
connection per process and event loop, as the trainer adapters do.

Requests are retried where that is safe:

| Request | Retried on |
| --- | --- |
| Create a session (`reset`) | 429 (server full) until `session_wait_s`, honouring `Retry-After`; 503 (provisioning failed), other 5xx and transport errors up to `max_retries`. It then raises `ProvisionError`, like a local sandbox that cannot be provisioned |
| `step`, `call_tool` | Only when the request never reached the server, so a tool call never runs twice |
| `evaluate`, `trajectory`, `close` | Transport errors and 5xx (idempotent on the server) |

A 4xx answer (bad token, invalid task, a refused reward) raises
`RemoteEnvError` at once: retrying cannot fix it.

## Trainer adapters

The verl, OpenRLHF and SkyRL adapters run their episodes on a server when their
settings have a `remote` section. The GPU nodes then need no Docker:

```yaml
# $AGENTBOX_VERL_CONFIG (or $AGENTBOX_OPENRLHF_CONFIG, $AGENTBOX_SKYRL_CONFIG)
remote:
  url: http://sandbox-host:8765
  token_env: AGENTBOX_SERVER_TOKEN     # the variable holding the token, on every GPU node
  max_retries: 5                       # 503 / transport errors
  session_wait_s: 1800                 # 429: wait this long for a free session
agent:                                 # sent with every session
  tools: [list_files, read_file, write_file, run_tests]
rewards:                               # registered names only
  - {name: verifier}
```

- **verl**: the agent loop's tool calls and the verifier run in a session.
- **OpenRLHF**: AgentBox's agent loop still runs token-in/token-out on the vLLM
  engine; only the tools and the verifier run on the server
  (`agentbox.env.remote.remote_rollout`).
- **SkyRL**: the text environment drives a session (Hermes parsing happens on
  the server).

If the server is lost in the middle of an episode, the episode becomes an
infra failure. verl zeroes its response mask, so its tokens are never trained,
and SkyRL ends it with `metadata.infra_failure`. On one node with two GPUs, a
verl step took 13.3 s with remote sandboxes and 13.2 s with local ones, and a
tool call 6 ms longer ([verl](integrations/verl.md#remote-sandboxes)).

## API

| Method | Path | Body / result |
| --- | --- | --- |
| `POST` | `/v1/sessions` | `{task | task_id, agent?, rewards?, group_id?, sample_index?, seed?}` → `{session_id, observation}`. `agent` overrides fields of the server's agent config, `rewards` replaces its reward components |
| `POST` | `/v1/sessions/{id}/step` | `{action: text | message | tool_calls}` → `StepResult` |
| `POST` | `/v1/sessions/{id}/tool` | `{name, arguments, call_id?}` → `{content, is_error, terminal, duration_s}` |
| `POST` | `/v1/sessions/{id}/evaluate` | optional `{steps, stop_reason}` from a client that ran the loop itself → `EvalResult` (reward, breakdown, success, failure kind, verifier result) |
| `GET` | `/v1/sessions/{id}/trajectory` | → `Trajectory` JSON |
| `DELETE` | `/v1/sessions/{id}` | removes the sandbox |
| `GET` | `/v1/tasks` | task ids from `--tasks` |
| `GET` | `/healthz` | `{ok, sessions, pending, max_sessions, tasks}` (no auth) |
| `GET` | `/metrics` | Prometheus text: active, pending and maximum sessions; created and rejected sessions, provisioning failures, steps, tool calls, evaluations, reaped sessions (no auth) |

## Errors and limits

- `--max-sessions` counts sessions still being provisioned, so a burst of
  requests cannot overshoot it. Beyond it, creating a session returns **429**
  with `Retry-After: 1`.
- A provisioning failure returns **503** with
  `{"failure_kind": "sandbox" | "seed" | "setup_check", "error": ...}`, so
  clients can retry or skip without scoring it as a policy failure.
- An unknown session returns 404, and requests to one session are serialised.
- Sessions idle for longer than `--idle-ttl` seconds are reaped and their
  sandboxes removed, as are all sessions on shutdown.
- The sandbox section's `host_max_sandboxes` caps the sandboxes on the server
  host as it does anywhere else ([sandbox](sandbox.md#host-wide-cap)). Keep
  `--max-sessions` at or below it: clients then wait on 429 instead of holding
  a request open while a session waits for a slot. A client that gives up on a
  slow request retries, and the session it abandoned holds a sandbox until
  `--idle-ttl`.

## Code from clients

A client may not name code to import on the server host. The server refuses
(400) a reward given as `module:function`, and a task whose verifier has a
custom `reward_fn`. Both would run outside any sandbox. Registered reward
names are accepted, and so are tasks loaded with `--tasks`, which are the
operator's own. `--allow-client-callables` lifts the restriction for clients
you trust as much as the host.

Run the server behind TLS (a reverse proxy) when it leaves a trusted network.
Sandbox isolation on the server host is exactly as described in
[security](security.md).
