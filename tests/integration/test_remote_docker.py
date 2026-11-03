"""Remote sandboxes end to end: an `agentbox serve` app, RemoteEnv and the adapters (GH #98)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agentbox.config import AgentConfig, RemoteConfig, ResourceLimits, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.types import FailureKind, FinalStatus, VerifierType

pytestmark = pytest.mark.docker

FIX = "def add(a, b):\n    return a + b\n"
TASK = Task(
    task_id="remote_add",
    description="Fix add() in calc.py so the tests pass.",
    starter_files={
        "calc.py": "def add(a, b):\n    return a - b\n",
        "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    },
    verifier=VerifierSpec(type=VerifierType.PYTEST, command="python -m pytest -q", timeout_s=60),
)


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An in-process server; RemoteEnv.from_config clients reach it through ASGI."""
    if not _docker_available():
        pytest.skip("Docker is not available")
    pytest.importorskip("fastapi")
    import httpx

    import agentbox.env.remote as remote_mod
    from agentbox.server.app import ServerSettings, create_app

    app = create_app(
        ServerSettings(
            sandbox=SandboxConfig(limits=ResourceLimits(network_disabled=True)),
            max_sessions=8,
        )
    )
    transport = httpx.ASGITransport(app=app)

    def client(*a: Any, **kw: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url="http://t", timeout=120)

    monkeypatch.setattr(remote_mod, "shared_client", client)
    return app


def _write_fix() -> ModelResponse:
    args = json.dumps({"path": "calc.py", "content": FIX})
    return ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="c1", function=FunctionCall(name="write_file", arguments=args))],
    )


async def test_remote_rollout_keeps_local_messages_and_server_reward(served: Any) -> None:
    from agentbox.env.remote import RemoteEnv, remote_rollout

    env = RemoteEnv.from_config(RemoteConfig(url="http://t"))
    model = MockModelClient([_write_fix(), ModelResponse(content="Fixed.", tool_calls=[])])
    traj = await remote_rollout(TASK, model=model, env=env, agent=AgentConfig(max_steps=4))
    assert traj.final_status == FinalStatus.SUCCESS and traj.reward == 1.0, traj.error
    assert [m.role for m in traj.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert traj.metrics.tool_calls == 1 and traj.verify is not None and traj.verify.success
    assert traj.metadata["remote"] == "http://t"
    assert served.state.sessions == {}  # the session (and its sandbox) is gone


async def test_remote_rollout_reports_provisioning_failures(served: Any) -> None:
    from agentbox.env.remote import RemoteEnv, remote_rollout

    broken = TASK.model_copy(update={"setup_commands": ["exit 3"]})
    env = RemoteEnv.from_config(RemoteConfig(url="http://t", max_retries=0))
    traj = await remote_rollout(
        broken, model=MockModelClient([]), env=env, agent=AgentConfig(max_steps=2)
    )
    assert traj.final_status == FinalStatus.ERROR and traj.failure_kind.is_infra
    assert traj.failure_kind == FailureKind.SEED and traj.metrics.model_calls == 0


async def test_verl_agent_loop_with_remote_sandboxes(served: Any) -> None:
    from agentbox.integrations.settings import Settings
    from agentbox.integrations.verl import build_agent_loop_class, tasks_to_verl_rows

    class FakeBase:  # stands in for verl's ToolAgentLoop: one write, then the turn ends
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            call = SimpleNamespace(
                name="write_file", arguments=json.dumps({"path": "calc.py", "content": FIX})
            )
            text, _, _ = await self._call_tool(call, {}, None)
            return SimpleNamespace(
                reward_score=None, response_mask=[1, 1], extra_fields={"t": text}
            )

    settings = Settings(
        agent=AgentConfig(tools=["read_file", "write_file", "run_tests"]),
        remote=RemoteConfig(url="http://t"),
    )
    loop = build_agent_loop_class(FakeBase, tool_response=lambda t: t, settings=settings)()
    out = await loop.run({}, extra_info=tasks_to_verl_rows([TASK])[0]["extra_info"])
    assert out.reward_score == 1.0 and out.response_mask == [1, 1]
    assert out.extra_fields["agentbox"]["success"] is True
    assert served.state.sessions == {}


def test_skyrl_core_with_remote_sandboxes(served: Any) -> None:
    from agentbox.integrations.settings import Settings
    from agentbox.integrations.skyrl import AgentBoxTextEnvCore

    core = AgentBoxTextEnvCore(
        TASK, settings=Settings(agent=AgentConfig(max_steps=4), remote=RemoteConfig(url="http://t"))
    )
    messages, info = core.init()
    assert "<tools>" in messages[0]["content"] and "write_file" in info["tools"]
    call = {"name": "write_file", "arguments": {"path": "calc.py", "content": FIX}}
    out = core.step(f"<tool_call>\n{json.dumps(call)}\n</tool_call>")
    assert not out["done"] and out["observations"][0]["role"] == "user"
    out = core.step("Fixed.")
    assert out["done"] and out["reward"] == 1.0 and out["metadata"]["infra_failure"] is False
    assert served.state.sessions == {}


async def test_server_shutdown_removes_its_warm_pool() -> None:
    """uvicorn re-raises SIGTERM after shutdown, skipping atexit: the lifespan must clean up."""
    if not _docker_available():
        pytest.skip("Docker is not available")
    pytest.importorskip("fastapi")
    import asyncio
    import time

    import httpx

    from agentbox.sandbox.backend import get_backend, owner_label
    from agentbox.server.app import ServerSettings, create_app

    def mine() -> int:
        owner = f"agentbox.owner={owner_label()}"
        return len(get_backend().list_containers_sync([owner]))

    app = create_app(ServerSettings(sandbox=SandboxConfig(warm_pool=2, ensure_pytest=False)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=120) as c:
            sid = (await c.post("/v1/sessions", json={"task": TASK.model_dump(mode="json")})).json()
            await c.delete(f"/v1/sessions/{sid['session_id']}")
        deadline = time.monotonic() + 30
        while mine() < 2 and time.monotonic() < deadline:  # the pool refills in the background
            await asyncio.sleep(0.2)
        assert mine() >= 2
    deadline = time.monotonic() + 15
    while mine() and time.monotonic() < deadline:  # creations in flight are dropped on arrival
        await asyncio.sleep(0.2)
    assert mine() == 0
