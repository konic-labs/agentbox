"""Remote environments without Docker: client retries, server guards, adapter settings (GH #98)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from agentbox.config import AgentConfig, RemoteConfig
from agentbox.env.env import EvalResult, Observation
from agentbox.env.remote import RemoteEnv, RemoteEnvError
from agentbox.integrations.settings import load_settings
from agentbox.rewards import RewardSpec
from agentbox.runner.provision import ProvisionError
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import FailureKind

OBS = {
    "task_id": "t",
    "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    "tools": [],
    "system_prompt": "s",
    "prompt": "u",
}
EVAL = {
    "reward": 1.0,
    "success": True,
    "final_status": "success",
    "failure_kind": "none",
    "breakdown": {"verifier": 1.0},
}


def _task(**verifier: Any) -> Task:
    return Task(
        task_id="t",
        description="d",
        verifier=VerifierSpec(type="command", command="true", **verifier),
    )


def _env(handler: Any, **kw: Any) -> tuple[RemoteEnv, list[str]]:
    seen: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return handler(request, len(seen))  # type: ignore[no-any-return]

    client = httpx.AsyncClient(transport=httpx.MockTransport(record), base_url="http://t")
    kw.setdefault("retry_backoff_s", 0.001)
    kw.setdefault("retry_max_backoff_s", 0.002)
    return RemoteEnv("http://t", client=client, **kw), seen


async def test_reset_waits_while_the_server_is_full() -> None:
    def handler(req: httpx.Request, n: int) -> httpx.Response:
        if n <= 2:
            return httpx.Response(429, json={"detail": "full"}, headers={"retry-after": "0"})
        body = json.loads(req.content)
        assert body["agent"]["max_steps"] == 3 and "custom_tools" not in body["agent"]
        assert body["rewards"] == [{"name": "verifier", "weight": 1.0, "params": {}}]
        return httpx.Response(200, json={"session_id": "s1", "observation": OBS})

    env, seen = _env(handler, agent=AgentConfig(max_steps=3), rewards=[RewardSpec(name="verifier")])
    obs = await env.reset(_task())
    assert isinstance(obs, Observation) and env.session_id == "s1"
    assert seen == ["POST /v1/sessions"] * 3


async def test_reset_gives_up_when_the_server_stays_full() -> None:
    env, _ = _env(lambda req, n: httpx.Response(429, json={}), session_wait_s=0.05)
    with pytest.raises(ProvisionError) as info:
        await env.reset(_task())
    assert info.value.kind is FailureKind.SANDBOX and "session limit" in str(info.value)


async def test_provisioning_failures_are_retried_then_attributed() -> None:
    detail = {"detail": {"failure_kind": "seed", "error": "seed failed"}}
    env, seen = _env(lambda req, n: httpx.Response(503, json=detail), max_retries=2)
    with pytest.raises(ProvisionError) as info:
        await env.reset(_task())
    assert info.value.kind is FailureKind.SEED and len(seen) == 3


async def test_client_errors_fail_loudly() -> None:
    env, seen = _env(lambda req, n: httpx.Response(401, json={"detail": "bad token"}))
    with pytest.raises(RemoteEnvError, match="401"):
        await env.reset(_task())
    assert len(seen) == 1


async def test_tool_calls_are_retried_only_when_never_sent() -> None:
    def handler(req: httpx.Request, n: int) -> httpx.Response:
        if req.url.path.endswith("/tool"):
            if n == 1:
                raise httpx.ConnectError("refused", request=req)
            if n == 3:
                raise httpx.ReadTimeout("slow", request=req)
            return httpx.Response(200, json={"content": "ok", "is_error": False, "terminal": False})
        return httpx.Response(200, json={})

    env, seen = _env(handler)
    env.session_id = "s1"
    assert (await env.call_tool("read_file", "{}"))["content"] == "ok"  # connect error retried
    with pytest.raises(RemoteEnvError, match="ReadTimeout"):  # may have run: not retried
        await env.call_tool("run_tests", "{}")
    assert seen.count("POST /v1/sessions/s1/tool") == 3


async def test_evaluate_is_retried_and_carries_the_loop_state() -> None:
    bodies: list[Any] = []

    def handler(req: httpx.Request, n: int) -> httpx.Response:
        bodies.append(json.loads(req.content or b"null"))
        if n == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=EVAL)

    env, _ = _env(handler)
    env.session_id = "s1"
    ev = await env.evaluate(steps=4, stop_reason="final_answer")
    assert isinstance(ev, EvalResult) and ev.success
    assert bodies[-1] == {"steps": 4, "stop_reason": "final_answer"}


async def test_close_is_best_effort() -> None:
    env, seen = _env(lambda req, n: httpx.Response(500, text="down"), max_retries=1)
    env.session_id = "s1"
    await env.close()  # logs, does not raise
    assert env.session_id is None and seen == ["DELETE /v1/sessions/s1"] * 2


async def test_shared_clients_are_per_event_loop() -> None:
    cfg = RemoteConfig(url="http://sandboxes:8765")
    a, b = RemoteEnv.from_config(cfg), RemoteEnv.from_config(cfg)
    assert a._http() is b._http()  # one pool per process and loop
    other = await asyncio.to_thread(lambda: asyncio.run(_client_of(cfg)))
    assert other is not a._http()


async def _client_of(cfg: RemoteConfig) -> httpx.AsyncClient:
    return RemoteEnv.from_config(cfg)._http()


# --------------------------------------------------------------------- settings
def test_settings_accept_a_path_or_the_document(tmp_path: Path) -> None:
    doc = "remote: {url: 'http://cpu1:8765', max_retries: 2}\nsandbox: {host_max_sandboxes: 8}\n"
    path = tmp_path / "s.yaml"
    path.write_text(doc)
    for source in (str(path), doc, json.dumps({"remote": {"url": "http://cpu1:8765"}})):
        s = load_settings(source)
        assert s.remote is not None and s.remote.url == "http://cpu1:8765"
    assert load_settings(doc).sandbox.host_max_sandboxes == 8
    full = "model: {model: qwen}\nagent: {max_steps: 5}\n"  # an agentbox.yaml works too
    assert load_settings(full).agent.max_steps == 5 and load_settings(full).remote is None


# ------------------------------------------------------------------ verl adapter
async def test_verl_remote_failures_are_never_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentbox.integrations.verl as verl_mod
    from agentbox.integrations.settings import Settings

    class FlakyRemote:
        @classmethod
        def from_config(cls, cfg: RemoteConfig, **kw: Any) -> FlakyRemote:
            return cls()

        async def reset(self, task: Task) -> None:
            pass

        async def call_tool(self, name: str, arguments: str) -> Any:
            raise RemoteEnvError("POST /tool: server unreachable")

        async def evaluate(self) -> Any:
            raise AssertionError("an episode that lost its server is not evaluated")

        async def close(self) -> None:
            pass

    class FakeBase:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            texts = [
                (await self._call_tool(SimpleNamespace(name=n, arguments="{}"), {}, None))[0]
                for n in ("read_file", "run_tests")
            ]
            return SimpleNamespace(
                reward_score=None, response_mask=[1, 1], extra_fields={"t": texts}
            )

    monkeypatch.setattr(verl_mod, "RemoteEnv", FlakyRemote)
    settings = Settings(remote=RemoteConfig(url="http://cpu1:8765"))
    loop = verl_mod.build_agent_loop_class(FakeBase, tool_response=lambda t: t, settings=settings)()
    out = await loop.run({}, extra_info=verl_mod.tasks_to_verl_rows([_task()])[0]["extra_info"])
    assert out.response_mask == [0, 0] and out.reward_score == 0.0
    assert out.extra_fields["agentbox"]["infra_failure"] is True
    assert out.extra_fields["agentbox"]["failure_kind"] == "sandbox"
    assert all("sandbox server is unavailable" in t for t in out.extra_fields["t"])


def test_skyrl_rows_carry_the_task_at_top_level_or_nested() -> None:
    from agentbox.integrations.skyrl import row_extras

    blob = _task().model_dump_json()
    assert row_extras({"agentbox_task": blob, "data_source": "x"})["agentbox_task"] == blob
    assert row_extras({"extras": {"agentbox_task": blob}})["agentbox_task"] == blob
    assert row_extras(None) == {}
