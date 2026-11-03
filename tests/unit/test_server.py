"""The environment server's guards and accounting, with a fake environment (GH #98)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from agentbox.config import AgentConfig
from agentbox.env.env import EvalResult, Observation
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import FailureKind, FinalStatus

OBS = {
    "task_id": "t",
    "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    "tools": [],
    "system_prompt": "s",
    "prompt": "u",
}


def _task(**verifier: Any) -> Task:
    return Task(
        task_id="t",
        description="d",
        verifier=VerifierSpec(type="command", command="true", **verifier),
    )


class _FakeEnv:
    gate: asyncio.Event | None = None
    created: list[_FakeEnv] = []

    def __init__(self, *, sandbox: Any, agent: AgentConfig, rewards: Any) -> None:
        self.agent, self.rewards = agent, rewards
        self.loop_state: dict[str, Any] = {}
        _FakeEnv.created.append(self)

    async def reset(self, task: Task, **kw: Any) -> Observation:
        if _FakeEnv.gate is not None:
            await _FakeEnv.gate.wait()
        return Observation.model_validate(OBS)

    def record_loop(self, **kw: Any) -> None:
        self.loop_state = kw

    async def evaluate(self) -> EvalResult:
        return EvalResult(
            reward=1.0,
            success=True,
            final_status=FinalStatus.SUCCESS,
            failure_kind=FailureKind.NONE,
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("fastapi")
    import agentbox.server.app as app_mod

    _FakeEnv.gate, _FakeEnv.created = None, []
    monkeypatch.setattr(app_mod, "AgentBoxEnv", _FakeEnv)

    def make(**kw: Any) -> httpx.AsyncClient:
        app = app_mod.create_app(app_mod.ServerSettings(**kw))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")

    return make


async def test_sessions_being_provisioned_count_against_the_cap(server: Any) -> None:
    client = server(max_sessions=2)
    _FakeEnv.gate = asyncio.Event()
    task = _task().model_dump(mode="json")
    first = [
        asyncio.create_task(client.post("/v1/sessions", json={"task": task})) for _ in range(2)
    ]
    await asyncio.sleep(0.05)  # both are provisioning, none is a session yet
    third = await client.post("/v1/sessions", json={"task": task})
    assert third.status_code == 429 and third.headers["retry-after"] == "1"
    _FakeEnv.gate.set()
    assert [r.status_code for r in await asyncio.gather(*first)] == [200, 200]
    health = (await client.get("/healthz")).json()
    assert health["sessions"] == 2 and health["pending"] == 0 and health["max_sessions"] == 2
    assert "agentbox_sessions_rejected_total 1" in (await client.get("/metrics")).text


async def test_clients_choose_registered_rewards_only(server: Any) -> None:
    client = server()
    task = _task().model_dump(mode="json")
    ok = await client.post(
        "/v1/sessions", json={"task": task, "rewards": [{"name": "verifier", "weight": 0.5}]}
    )
    assert ok.status_code == 200 and _FakeEnv.created[-1].rewards[0].weight == 0.5
    refused = await client.post(
        "/v1/sessions", json={"task": task, "rewards": [{"name": "os:system"}]}
    )
    assert refused.status_code == 400 and "os:system" in refused.text


async def test_client_tasks_cannot_name_host_code(server: Any) -> None:
    task = _task(reward_mode="custom", reward_fn="os:getcwd").model_dump(mode="json")
    refused = await server().post("/v1/sessions", json={"task": task})
    assert refused.status_code == 400 and "--allow-client-callables" in refused.text
    allowed = await server(allow_client_callables=True).post("/v1/sessions", json={"task": task})
    assert allowed.status_code == 200


async def test_agent_overrides_and_client_loop_state(server: Any) -> None:
    client = server()
    body = {"task": _task().model_dump(mode="json"), "agent": {"max_steps": 2, "custom_tools": [1]}}
    sid = (await client.post("/v1/sessions", json=body)).json()["session_id"]
    env = _FakeEnv.created[-1]
    assert env.agent.max_steps == 2 and env.agent.custom_tools == []
    r = await client.post(f"/v1/sessions/{sid}/evaluate", json={"steps": 3, "stop_reason": "x"})
    assert r.status_code == 200 and env.loop_state == {"steps": 3, "stop_reason": "x"}
    assert (await client.post(f"/v1/sessions/{sid}/evaluate")).status_code == 200  # no body
