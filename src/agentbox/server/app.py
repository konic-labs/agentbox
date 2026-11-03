"""HTTP environment server (GH #88): AgentBox sandboxes as a network service.

Trainers on GPU nodes drive environments hosted on CPU/Docker nodes::

    POST   /v1/sessions                    {task | task_id, agent?, rewards?, group_id?, ...}
    POST   /v1/sessions/{id}/step          {action: text | message | tool_calls}
    POST   /v1/sessions/{id}/tool          {name, arguments, call_id?}
    POST   /v1/sessions/{id}/evaluate      {steps?, stop_reason?}
    GET    /v1/sessions/{id}/trajectory
    DELETE /v1/sessions/{id}
    GET    /v1/tasks · /healthz · /metrics

Optional bearer auth (``AGENTBOX_SERVER_TOKEN``), a session cap (sessions still
being provisioned count), and an idle reaper. Clients choose the agent config
and registered reward components per session. Nothing a client sends may name
code to import on this host (a reward ``module:function`` or a task's custom
``reward_fn``) unless the server allows it.
"""

import asyncio
import contextlib
import hmac
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentbox.config import AgentConfig, SandboxConfig
from agentbox.env.env import AgentBoxEnv
from agentbox.rewards import RewardSpec, available_rewards
from agentbox.runner.provision import ProvisionError
from agentbox.tasks.loader import iter_tasks
from agentbox.tasks.schema import Task

logger = logging.getLogger("agentbox.server")


@dataclass
class _Session:
    env: AgentBoxEnv
    last_used: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ServerSettings:
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rewards: list[RewardSpec] | None = None
    tasks: str | None = None
    token: str | None = None
    max_sessions: int = 64
    idle_ttl_s: float = 1800.0
    allow_client_callables: bool = False


def create_app(settings: ServerSettings | None = None) -> Any:
    try:
        from fastapi import Body, Depends, FastAPI, HTTPException, Request
        from fastapi.responses import PlainTextResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("the server needs: pip install 'agentbox-rl[server]'") from exc

    cfg = settings or ServerSettings(token=os.environ.get("AGENTBOX_SERVER_TOKEN"))
    sessions: dict[str, _Session] = {}
    pending = {"n": 0}  # sessions being provisioned: they count against max_sessions
    counters = {
        "sessions_created": 0,
        "sessions_rejected": 0,
        "provision_failures": 0,
        "steps": 0,
        "tool_calls": 0,
        "evaluations": 0,
        "reaped": 0,
    }
    task_index: dict[str, Task] = {t.task_id: t for t in iter_tasks(cfg.tasks)} if cfg.tasks else {}

    async def reaper() -> None:
        while True:
            await asyncio.sleep(min(60.0, cfg.idle_ttl_s / 2))
            now = time.monotonic()
            for sid, s in list(sessions.items()):
                if now - s.last_used > cfg.idle_ttl_s and not s.lock.locked():
                    sessions.pop(sid, None)
                    counters["reaped"] += 1
                    with contextlib.suppress(Exception):
                        await s.env.close()

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        task = asyncio.create_task(reaper())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(
                *(s.env.close() for s in sessions.values()), return_exceptions=True
            )
            sessions.clear()

    app = FastAPI(title="AgentBox environment server", lifespan=lifespan)

    expected = f"Bearer {cfg.token}".encode() if cfg.token else None

    async def auth(request: Request) -> None:
        if expected is None:
            return
        given = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(given, expected):  # constant time
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def get(sid: str) -> _Session:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail=f"unknown session {sid}")
        s.last_used = time.monotonic()
        return s

    def client_task(raw: Any) -> Task:
        try:
            task = Task.model_validate(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid task: {exc}") from exc
        if task.verifier.reward_fn and not cfg.allow_client_callables:
            raise HTTPException(
                status_code=400,
                detail="tasks with a custom reward_fn run code on the server host; "
                "load them with --tasks or start the server with --allow-client-callables",
            )
        return task

    def client_agent(raw: Any) -> AgentConfig:
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="agent must be an object")
        merged = {**cfg.agent.model_dump(exclude={"custom_tools"}), **raw}
        merged.pop("custom_tools", None)  # tool objects cannot travel over HTTP
        try:
            return AgentConfig.model_validate(merged)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid agent: {exc}") from exc

    def client_rewards(raw: Any) -> list[RewardSpec]:
        try:
            specs = [RewardSpec.model_validate(r) for r in raw]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid rewards: {exc}") from exc
        known = set(available_rewards())
        refused = [s.name for s in specs if s.name not in known]
        if refused and not cfg.allow_client_callables:
            raise HTTPException(
                status_code=400,
                detail=f"rewards must be registered names {sorted(known)}; refused {refused}",
            )
        return specs

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "sessions": len(sessions),
            "pending": pending["n"],
            "max_sessions": cfg.max_sessions,
            "tasks": len(task_index),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        lines = [f"agentbox_{k}_total {v}" for k, v in counters.items()]
        lines.append(f"agentbox_sessions_active {len(sessions)}")
        lines.append(f"agentbox_sessions_pending {pending['n']}")
        lines.append(f"agentbox_sessions_max {cfg.max_sessions}")
        return "\n".join(lines) + "\n"

    @app.get("/v1/tasks", dependencies=[Depends(auth)])
    async def list_tasks() -> list[str]:
        return sorted(task_index)

    @app.post("/v1/sessions", dependencies=[Depends(auth)])
    async def create_session(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("task") is not None:
            task = client_task(body["task"])
        elif body.get("task_id") in task_index:
            task = task_index[body["task_id"]]
        else:
            raise HTTPException(status_code=400, detail="provide task or a known task_id")
        agent = client_agent(body["agent"]) if body.get("agent") else cfg.agent
        rewards = client_rewards(body["rewards"]) if body.get("rewards") else cfg.rewards
        if len(sessions) + pending["n"] >= cfg.max_sessions:
            counters["sessions_rejected"] += 1
            raise HTTPException(
                status_code=429, detail="session limit reached", headers={"Retry-After": "1"}
            )
        pending["n"] += 1
        try:
            env = AgentBoxEnv(sandbox=cfg.sandbox, agent=agent, rewards=rewards)
            try:
                obs = await env.reset(
                    task,
                    group_id=body.get("group_id"),
                    sample_index=int(body.get("sample_index") or 0),
                    seed=body.get("seed"),
                )
            except ProvisionError as exc:
                counters["provision_failures"] += 1
                raise HTTPException(
                    status_code=503, detail={"failure_kind": exc.kind.value, "error": str(exc)}
                ) from exc
            except BaseException:
                await env.close()
                raise
            sid = uuid.uuid4().hex
            sessions[sid] = _Session(env=env)
        finally:
            pending["n"] -= 1
        counters["sessions_created"] += 1
        return {"session_id": sid, "observation": obs.model_dump()}

    @app.post("/v1/sessions/{sid}/step", dependencies=[Depends(auth)])
    async def step(sid: str, body: dict[str, Any]) -> dict[str, Any]:
        s = get(sid)
        async with s.lock:
            res = await s.env.step(body.get("action", ""))
        counters["steps"] += 1
        return res.model_dump()

    @app.post("/v1/sessions/{sid}/tool", dependencies=[Depends(auth)])
    async def tool(sid: str, body: dict[str, Any]) -> dict[str, Any]:
        s = get(sid)
        async with s.lock:
            out = await s.env.call_tool(
                body["name"], body.get("arguments", {}), call_id=body.get("call_id")
            )
        counters["tool_calls"] += 1
        return {
            "content": out.message.content,
            "is_error": out.record.is_error,
            "terminal": out.terminal,
            "duration_s": out.record.duration_s,
        }

    @app.post("/v1/sessions/{sid}/evaluate", dependencies=[Depends(auth)])
    async def evaluate(
        sid: str,
        body: dict[str, Any] | None = Body(default=None),  # noqa: B008 - FastAPI marker
    ) -> dict[str, Any]:
        s = get(sid)
        loop = body or {}
        steps, stop_reason = loop.get("steps"), loop.get("stop_reason")
        async with s.lock:
            if steps is not None or stop_reason is not None:  # the client ran the loop
                s.env.record_loop(
                    steps=int(steps) if steps is not None else None,
                    stop_reason=str(stop_reason) if stop_reason else None,
                )
            res = await s.env.evaluate()
        counters["evaluations"] += 1
        return res.model_dump(mode="json")

    @app.get("/v1/sessions/{sid}/trajectory", dependencies=[Depends(auth)])
    async def trajectory(sid: str) -> dict[str, Any]:
        s = get(sid)
        return s.env.trajectory().model_dump(mode="json")

    @app.delete("/v1/sessions/{sid}", dependencies=[Depends(auth)])
    async def close(sid: str) -> dict[str, Any]:
        s = sessions.pop(sid, None)
        if s is not None:
            await s.env.close()
        return {"closed": s is not None}

    app.state.sessions = sessions
    return app


def settings_from_project(
    project: Any,
    *,
    tasks: Path | None,
    token: str | None,
    max_sessions: int,
    idle_ttl_s: float,
    allow_client_callables: bool = False,
) -> ServerSettings:
    return ServerSettings(
        sandbox=project.sandbox or SandboxConfig(),
        agent=project.agent or AgentConfig(),
        rewards=project.rewards,
        tasks=str(tasks) if tasks else None,
        token=token,
        max_sessions=max_sessions,
        idle_ttl_s=idle_ttl_s,
        allow_client_callables=allow_client_callables,
    )
