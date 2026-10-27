"""Client for `agentbox serve` with the same surface as :class:`AgentBoxEnv`."""

from __future__ import annotations

import os
from typing import Any

import httpx

from agentbox.env.env import EvalResult, Observation, StepResult
from agentbox.tasks.schema import Task
from agentbox.trajectory.schema import Trajectory


class RemoteEnv:
    """Drive a sandbox hosted by ``agentbox serve`` (GH #88)."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        token = token or os.environ.get("AGENTBOX_SERVER_TOKEN")
        headers = {"authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )
        self._owns = client is None
        self.session_id: str | None = None

    async def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        resp = await self._client.post(path, json=body or {})
        resp.raise_for_status()
        return resp.json()

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
        data = await self._post("/v1/sessions", body)
        self.session_id = data["session_id"]
        return Observation.model_validate(data["observation"])

    def _sid(self) -> str:
        if self.session_id is None:
            raise RuntimeError("call reset() first")
        return self.session_id

    async def step(self, action: Any) -> StepResult:
        if isinstance(action, list):
            action = {
                "content": None,
                "tool_calls": [a.model_dump() if hasattr(a, "model_dump") else a for a in action],
            }
        return StepResult.model_validate(
            await self._post(f"/v1/sessions/{self._sid()}/step", {"action": action})
        )

    async def call_tool(self, name: str, arguments: Any) -> dict[str, Any]:
        result: dict[str, Any] = await self._post(
            f"/v1/sessions/{self._sid()}/tool", {"name": name, "arguments": arguments}
        )
        return result

    async def evaluate(self) -> EvalResult:
        return EvalResult.model_validate(await self._post(f"/v1/sessions/{self._sid()}/evaluate"))

    async def trajectory(self) -> Trajectory:
        resp = await self._client.get(f"/v1/sessions/{self._sid()}/trajectory")
        resp.raise_for_status()
        return Trajectory.model_validate(resp.json())

    async def close_session(self) -> None:
        if self.session_id is not None:
            sid, self.session_id = self.session_id, None
            resp = await self._client.delete(f"/v1/sessions/{sid}")
            resp.raise_for_status()

    async def close(self) -> None:
        await self.close_session()
        if self._owns:
            await self._client.aclose()

    async def __aenter__(self) -> RemoteEnv:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
