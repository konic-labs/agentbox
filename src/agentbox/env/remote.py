"""Client for ``agentbox serve`` with the same surface as :class:`AgentBoxEnv` (GH #88, #98).

Trainers on GPU nodes without Docker drive sandboxes hosted on CPU machines.
Requests are retried only where that is safe:

=================================  ================================================
request                            retried on
=================================  ================================================
create a session (``reset``)       429 (server at ``--max-sessions``) until
                                   ``session_wait_s``; 503 (provisioning failed)
                                   and transport errors up to ``max_retries``
``step`` / ``call_tool``           only when the request never reached the server:
                                   a tool call must not run twice
``evaluate`` / ``trajectory`` /    transport errors and 5xx (idempotent on the
``close_session``                  server)
=================================  ================================================

A session that cannot be created raises :class:`ProvisionError`, like a local
sandbox that cannot be provisioned, so adapters treat both as infra failures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from agentbox.agent.loop import AgentLoop
from agentbox.config import AgentConfig, RemoteConfig
from agentbox.env.env import EvalResult, Observation, StepResult
from agentbox.model.base import ModelClient
from agentbox.rewards import RewardSpec
from agentbox.runner.provision import ProvisionError
from agentbox.runner.rollout import loop_metrics, task_hash, tool_mode_label
from agentbox.tasks.schema import Task
from agentbox.tito import SESSION_KEY
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import (
    SKIPPED_AFTER_SUBMIT,
    ToolOutcome,
    make_outcome,
    parse_arguments,
)
from agentbox.trajectory.schema import ToolCall, Trajectory, TrajectoryMetrics
from agentbox.types import FailureKind, FinalStatus

logger = logging.getLogger("agentbox.remote")

# the request never reached the server: retrying cannot run anything twice
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class RemoteEnvError(RuntimeError):
    """The environment server failed or could not be reached."""


def agent_overrides(agent: AgentConfig | dict[str, Any] | None) -> dict[str, Any] | None:
    """The agent config a client sends with each session (custom tool objects stay local)."""
    if agent is None:
        return None
    if isinstance(agent, AgentConfig):
        return agent.model_dump(mode="json", exclude={"custom_tools"})
    return {k: v for k, v in agent.items() if k != "custom_tools"}


_CLIENTS: dict[tuple[int, str, str | None], tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]]
_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def shared_client(
    base_url: str, token: str | None, *, timeout_s: float, max_connections: int
) -> httpx.AsyncClient:
    """One pooled client per server, token and event loop (httpx clients are loop-bound)."""
    loop = asyncio.get_running_loop()
    key = (id(loop), base_url, token)
    with _CLIENTS_LOCK:
        hit = _CLIENTS.get(key)
        if hit is not None and hit[0] is loop and not hit[1].is_closed:
            return hit[1]
        client = httpx.AsyncClient(
            base_url=base_url,
            headers={"authorization": f"Bearer {token}"} if token else {},
            timeout=timeout_s,
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )
        _CLIENTS[key] = (loop, client)
        return client


def _detail(resp: httpx.Response) -> Any:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


class RemoteEnv:
    """Drive a sandbox hosted by ``agentbox serve`` (GH #88).

    ``agent`` and ``rewards`` travel with every session, so the client's
    settings apply on the server. Servers accept registered reward names only.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 600.0,
        client: httpx.AsyncClient | None = None,
        agent: AgentConfig | dict[str, Any] | None = None,
        rewards: list[RewardSpec] | None = None,
        max_retries: int = 5,
        session_wait_s: float = 1800.0,
        retry_backoff_s: float = 0.5,
        retry_max_backoff_s: float = 10.0,
        shared: bool = False,
        max_connections: int = 512,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("AGENTBOX_SERVER_TOKEN")
        self.timeout_s = timeout_s
        self.agent = agent_overrides(agent)
        self.rewards = rewards
        self.max_retries = max_retries
        self.session_wait_s = session_wait_s
        self.retry_backoff_s = retry_backoff_s
        self.retry_max_backoff_s = retry_max_backoff_s
        self.max_connections = max_connections
        if client is None and not shared:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"authorization": f"Bearer {self.token}"} if self.token else {},
                timeout=timeout_s,
            )
            self._owns = True
        else:
            self._owns = False
        self._client = client  # None: the per-event-loop shared client
        self.session_id: str | None = None
        self._done = False

    @classmethod
    def from_config(
        cls,
        cfg: RemoteConfig,
        *,
        agent: AgentConfig | dict[str, Any] | None = None,
        rewards: list[RewardSpec] | None = None,
    ) -> RemoteEnv:
        """An env on the process-wide connection pool for ``cfg.url``."""
        return cls(
            cfg.url,
            token=os.environ.get(cfg.token_env),
            timeout_s=cfg.timeout_s,
            agent=agent,
            rewards=rewards,
            max_retries=cfg.max_retries,
            session_wait_s=cfg.session_wait_s,
            retry_backoff_s=cfg.retry_backoff_s,
            retry_max_backoff_s=cfg.retry_max_backoff_s,
            shared=True,
            max_connections=cfg.max_connections,
        )

    # ---------------------------------------------------------------- http
    def _http(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return shared_client(
            self.base_url,
            self.token,
            timeout_s=self.timeout_s,
            max_connections=self.max_connections,
        )

    def _backoff(self, attempt: int, floor: float | None = None) -> float:
        cap = min(self.retry_max_backoff_s, self.retry_backoff_s * 2 ** max(attempt - 1, 0))
        delay = random.uniform(0, cap)  # full jitter
        return max(delay, floor or 0.0)

    async def _call(self, method: str, path: str, body: Any = None, *, idempotent: bool) -> Any:
        attempt = 0
        while True:
            try:
                resp = await self._http().request(method, path, json=body)
            except _NOT_SENT as exc:
                if attempt >= self.max_retries:
                    raise RemoteEnvError(f"{method} {path}: server unreachable: {exc}") from exc
            except httpx.TransportError as exc:
                if not idempotent or attempt >= self.max_retries:
                    raise RemoteEnvError(f"{method} {path}: {type(exc).__name__}: {exc}") from exc
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code < 500 or not idempotent or attempt >= self.max_retries:
                    raise RemoteEnvError(
                        f"{method} {path}: HTTP {resp.status_code}: {_detail(resp)}"
                    )
            attempt += 1
            await asyncio.sleep(self._backoff(attempt))

    async def _create_session(self, body: dict[str, Any]) -> dict[str, Any]:
        attempt = waited = 0
        full_until: float | None = None
        while True:
            try:
                resp = await self._http().post("/v1/sessions", json=body)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise ProvisionError(
                        FailureKind.SANDBOX, f"agentbox server {self.base_url} unreachable: {exc}"
                    ) from exc
                attempt += 1
                await asyncio.sleep(self._backoff(attempt))
                continue
            if resp.status_code == 429:  # at --max-sessions: wait for a session to close
                now = time.monotonic()
                full_until = full_until or now + self.session_wait_s
                if now >= full_until:
                    raise ProvisionError(
                        FailureKind.SANDBOX,
                        f"agentbox server {self.base_url} stayed at its session limit "
                        f"for {self.session_wait_s:.0f}s",
                    )
                waited += 1
                await asyncio.sleep(self._backoff(min(waited, 6), _retry_after(resp)))
                continue
            if resp.status_code >= 500:  # 503: provisioning failed; others: server trouble
                detail = _detail(resp)
                info = detail if isinstance(detail, dict) else {"error": str(detail)}
                try:
                    kind = FailureKind(info.get("failure_kind", "sandbox"))
                except ValueError:
                    kind = FailureKind.SANDBOX
                if attempt >= self.max_retries:
                    raise ProvisionError(kind, str(info.get("error", "provisioning failed")))
                attempt += 1
                await asyncio.sleep(self._backoff(attempt))
                continue
            if resp.status_code >= 400:  # auth, bad task, refused rewards: fail loudly
                raise RemoteEnvError(f"POST /v1/sessions: HTTP {resp.status_code}: {_detail(resp)}")
            data: dict[str, Any] = resp.json()
            return data

    # ------------------------------------------------------------- episode
    @property
    def done(self) -> bool:
        return self._done

    async def reset(
        self,
        task: Task | str,
        *,
        group_id: str | None = None,
        sample_index: int = 0,
        seed: int | None = None,
    ) -> Observation:
        await self.close_session()
        body: dict[str, Any] = {"group_id": group_id, "sample_index": sample_index, "seed": seed}
        if isinstance(task, Task):
            body["task"] = task.model_dump(mode="json")
        else:
            body["task_id"] = task
        if self.agent is not None:
            body["agent"] = self.agent
        if self.rewards is not None:
            body["rewards"] = [r.model_dump(mode="json") for r in self.rewards]
        data = await self._create_session(body)
        self.session_id = data["session_id"]
        self._done = False
        return Observation.model_validate(data["observation"])

    def _sid(self) -> str:
        if self.session_id is None:
            raise RuntimeError("call reset() first")
        return self.session_id

    async def step(self, action: Any) -> StepResult:
        if isinstance(action, list):
            action = {
                "content": None,
                "tool_calls": [
                    a.model_dump(mode="json") if hasattr(a, "model_dump") else a for a in action
                ],
            }
        elif hasattr(action, "model_dump"):
            action = action.model_dump(mode="json", exclude_none=True)
        res = StepResult.model_validate(
            await self._call(
                "POST", f"/v1/sessions/{self._sid()}/step", {"action": action}, idempotent=False
            )
        )
        self._done = res.done
        return res

    async def call_tool(
        self, name: str, arguments: Any, *, call_id: str | None = None
    ) -> dict[str, Any]:
        """Run one tool call; returns ``{content, is_error, terminal, duration_s}``."""
        body = {"name": name, "arguments": arguments, "call_id": call_id}
        result: dict[str, Any] = await self._call(
            "POST", f"/v1/sessions/{self._sid()}/tool", body, idempotent=False
        )
        if result.get("terminal"):
            self._done = True
        return result

    async def evaluate(
        self, *, steps: int | None = None, stop_reason: str | None = None
    ) -> EvalResult:
        """Run the verifier on the server.

        ``steps`` and ``stop_reason`` describe a loop that ran on the client, for
        reward components such as ``step_penalty``.
        """
        body = {k: v for k, v in {"steps": steps, "stop_reason": stop_reason}.items() if v}
        return EvalResult.model_validate(
            await self._call("POST", f"/v1/sessions/{self._sid()}/evaluate", body, idempotent=True)
        )

    async def trajectory(self) -> Trajectory:
        data = await self._call("GET", f"/v1/sessions/{self._sid()}/trajectory", idempotent=True)
        return Trajectory.model_validate(data)

    async def close_session(self) -> None:
        """Close the session; best effort (the server's idle reaper is the backstop)."""
        if self.session_id is None:
            return
        sid, self.session_id = self.session_id, None
        try:
            await self._call("DELETE", f"/v1/sessions/{sid}", idempotent=True)
        except RemoteEnvError as exc:
            logger.warning("could not close remote session %s: %s", sid, exc)

    async def close(self) -> None:
        await self.close_session()
        if self._owns and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> RemoteEnv:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class RemoteToolExecutor:
    """Run the tool calls of a local :class:`AgentLoop` in a server session.

    Duck-types :class:`ToolExecutor`. Calls run one at a time in emission order
    (the server serialises a session's requests anyway). After the first
    transport failure, :attr:`error` is set, the remaining calls are answered
    with an error, and the episode counts as an infra failure.
    """

    def __init__(
        self, env: RemoteEnv, schemas: list[dict[str, Any]], *, max_output_chars: int
    ) -> None:
        self.env = env
        self._schemas = schemas
        self.max_output_chars = max_output_chars
        self.error: str | None = None

    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def names(self) -> list[str]:
        return [s["function"]["name"] for s in self._schemas]

    async def run_call(
        self, tc: ToolCall, ctx: ToolContext, *, deadline: float | None = None
    ) -> ToolOutcome:
        args, _ = parse_arguments(tc.function.arguments)
        t0 = time.monotonic()
        content, is_error, terminal = "", True, False
        if self.error is None:
            try:
                out = await self.env.call_tool(
                    tc.function.name, tc.function.arguments or "", call_id=tc.id
                )
            except RemoteEnvError as exc:
                self.error = str(exc)
            else:
                content = out.get("content") or ""
                is_error = bool(out.get("is_error"))
                terminal = bool(out.get("terminal"))
        if self.error is not None:
            content = f"ERROR: the sandbox server is unavailable ({self.error})"
        return make_outcome(
            tc,
            ctx.step,
            content,
            is_error=is_error,
            args=args,
            max_output_chars=self.max_output_chars,
            duration_s=time.monotonic() - t0,
            terminal=terminal,
        )

    async def run_calls(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolContext,
        *,
        deadline: float | None = None,
        parallel: bool | None = None,
    ) -> list[ToolOutcome]:
        outcomes: list[ToolOutcome] = []
        for tc in tool_calls:
            if outcomes and outcomes[-1].terminal:
                outcomes.append(
                    make_outcome(
                        tc,
                        ctx.step,
                        SKIPPED_AFTER_SUBMIT,
                        is_error=True,
                        args=None,
                        max_output_chars=self.max_output_chars,
                    )
                )
                continue
            outcomes.append(await self.run_call(tc, ctx, deadline=deadline))
        return outcomes


async def remote_rollout(
    task: Task,
    *,
    model: ModelClient,
    env: RemoteEnv,
    agent: AgentConfig,
    run_id: str | None = None,
    rollout_id: str | None = None,
    group_id: str | None = None,
    sample_index: int = 0,
) -> Trajectory:
    """One episode with AgentBox's agent loop here and its tools on the server.

    The trajectory keeps the local loop's messages, with any token data a TITO
    client recorded, and takes the reward and verifier result from the server.
    The session is closed before returning.
    """
    t0 = time.monotonic()
    created_at = datetime.now(UTC)
    run_id = run_id or uuid.uuid4().hex
    rollout_id = rollout_id or uuid.uuid4().hex
    metrics = TrajectoryMetrics()
    base: dict[str, Any] = {
        "task_id": task.task_id,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "group_id": group_id,
        "sample_index": sample_index,
        "model": getattr(model, "model", None),
        "tool_mode": tool_mode_label(agent.tools),
        "task_hash": task_hash(task),
        "created_at": created_at,
    }
    meta: dict[str, Any] = {"task_metadata": task.metadata, "remote": env.base_url}

    def failed(kind: FailureKind, error: str, **extra: Any) -> Trajectory:
        metrics.duration_s = time.monotonic() - t0
        extra.setdefault("messages", [])
        return Trajectory(
            **base,
            **extra,
            final_status=FinalStatus.ERROR,
            failure_kind=kind,
            error=error,
            metrics=metrics,
            metadata=meta,
            finished_at=datetime.now(UTC),
        )

    # the server renders the plain system prompt; a hermes loop adds its tool block here
    env.agent = {**(agent_overrides(agent) or {}), "tool_protocol": "native"}
    try:
        obs = await env.reset(task, group_id=group_id, sample_index=sample_index)
    except ProvisionError as exc:
        return failed(exc.kind, str(exc))
    try:
        executor = RemoteToolExecutor(env, obs.tools, max_output_chars=agent.max_tool_output_chars)
        ctx = ToolContext(
            sandbox=None,
            run_id=run_id,
            task_id=task.task_id,
            tool_timeout_s=agent.tool_timeout_s,
            max_output_chars=agent.max_tool_output_chars,
        )
        loop = AgentLoop(model, executor, agent)  # type: ignore[arg-type]
        session = SESSION_KEY.set(rollout_id)  # per-rollout state for TITO clients
        try:
            result = await loop.run(
                system_prompt=obs.system_prompt,
                user_prompt=obs.prompt,
                ctx=ctx,
                episode_deadline=time.monotonic() + agent.episode_timeout_s,
            )
        finally:
            SESSION_KEY.reset(session)
            end = getattr(getattr(model, "inner", model), "end_session", None)
            if end is not None:
                end(rollout_id)
        loop_metrics(metrics, result)
        common: dict[str, Any] = {
            "messages": result.messages,
            "tool_call_records": result.tool_call_records,
            "stop_reason": result.stop_reason,
            "tools": obs.tools,
        }
        if executor.error is not None:
            return failed(FailureKind.SANDBOX, f"sandbox server failed: {executor.error}", **common)
        t_verify = time.monotonic()
        try:
            ev = await env.evaluate(steps=result.steps, stop_reason=result.stop_reason)
        except RemoteEnvError as exc:
            return failed(FailureKind.VERIFIER, f"remote evaluation failed: {exc}", **common)
        metrics.verify_s = time.monotonic() - t_verify
        if result.failure_kind.is_infra and not ev.failure_kind.is_infra:
            status, failure = FinalStatus.ERROR, result.failure_kind  # e.g. the model API
        else:
            status, failure = ev.final_status, ev.failure_kind
        metrics.duration_s = time.monotonic() - t0
        return Trajectory(
            **base,
            **common,
            reward=ev.reward,
            reward_breakdown=ev.breakdown,
            final_status=status,
            failure_kind=failure,
            verify=ev.verify,
            error=ev.error or result.error,
            metrics=metrics,
            metadata=meta,
            finished_at=datetime.now(UTC),
        )
    finally:
        await env.close_session()
