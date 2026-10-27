"""SkyRL integration (GH #74): a ``skyrl_gym`` text environment.

SkyRL envs are text-in/text-out; AgentBox renders tools into the system prompt
(Hermes protocol), parses ``<tool_call>`` blocks from each action and returns
tool results as user observations::

    from agentbox.integrations.skyrl import register
    register()   # registers "agentbox" with skyrl_gym

Dataset rows carry the task in ``extras["agentbox_task"]`` (JSON) or
``extras["task_id"]`` with ``$AGENTBOX_TASKS`` set.
"""

from __future__ import annotations

import os
from typing import Any

from agentbox.env.env import SyncAgentBoxEnv
from agentbox.integrations.verl import Settings, load_settings, task_from_kwargs
from agentbox.runner.provision import ProvisionError
from agentbox.tasks.schema import Task

PROVISION_ATTEMPTS = 3


class AgentBoxTextEnvCore:
    """Framework-free core (``init``/``step``/``close``) used by the SkyRL class."""

    def __init__(self, task: Task, *, settings: Settings | None = None) -> None:
        cfg = settings or load_settings(os.environ.get("AGENTBOX_SKYRL_CONFIG"))
        agent = cfg.agent.model_copy(
            update={"tool_protocol": "hermes", "tool_response_role": "user"}
        )
        self.task = task
        self.env = SyncAgentBoxEnv(sandbox=cfg.sandbox, agent=agent, rewards=cfg.rewards)
        self.turns = 0
        self.max_turns = agent.max_steps

    def init(self, prompt: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for attempt in range(PROVISION_ATTEMPTS):
            try:
                obs = self.env.reset(self.task)
                break
            except ProvisionError:  # transient Docker trouble should not kill the run
                if attempt == PROVISION_ATTEMPTS - 1:
                    raise
        return obs.messages, {
            "task_id": self.task.task_id,
            "tools": [t["function"]["name"] for t in obs.tools],
        }

    def step(self, action: str) -> dict[str, Any]:
        self.turns += 1
        res = self.env.step(action)
        if res.done or self.turns >= self.max_turns:
            result = self.env.evaluate()
            self.env.close()
            return {
                "observations": [],
                "reward": result.reward,
                "done": True,
                "metadata": {
                    "success": result.success,
                    "status": result.final_status.value,
                    "reward_breakdown": result.breakdown,
                },
            }
        return {"observations": res.messages, "reward": 0.0, "done": False, "metadata": {}}

    def close(self) -> None:
        self.env.close()


def _build() -> type:
    from skyrl_gym.envs.base_text_env import BaseTextEnv

    class AgentBoxTextEnv(BaseTextEnv):  # type: ignore[misc]
        def __init__(self, env_config: Any = None, extras: dict[str, Any] | None = None) -> None:
            super().__init__()
            extras = extras or {}
            task = task_from_kwargs({"extra_info": extras})
            self._core = AgentBoxTextEnvCore(task)
            self.max_turns = self._core.max_turns

        def init(self, prompt: Any) -> Any:
            return self._core.init(prompt)

        def step(self, action: str) -> Any:
            out = self._core.step(action)
            self.turns = self._core.turns
            return out

        def close(self) -> None:
            self._core.close()

    return AgentBoxTextEnv


def register(env_id: str = "agentbox") -> None:
    from skyrl_gym.envs import register as skyrl_register

    skyrl_register(id=env_id, entry_point="agentbox.integrations.skyrl:AgentBoxTextEnv")


def __getattr__(name: str) -> Any:
    if name == "AgentBoxTextEnv":
        cls = _build()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
