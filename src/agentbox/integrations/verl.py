"""verl integration (GH #72).

``AgentBoxAgentLoop`` subclasses verl's ``ToolAgentLoop`` so verl keeps full
control of tokenisation (continuous-token, no retokenisation drift) and only
tool execution and the reward come from AgentBox:

* the episode's task is read from the dataset row (``extra_info.agentbox_task``
  as JSON, or ``extra_info.task_id`` resolved against ``$AGENTBOX_TASKS``);
* a sandbox is provisioned per trajectory and every tool call runs in it;
* ``reward_score`` is the AgentBox reward (verifier + reward components).

Register it in the agent-loop config consumed by verl::

    # agent_loops.yaml
    - name: agentbox
      _target_: agentbox.integrations.verl.AgentBoxAgentLoop

and build the dataset with :func:`tasks_to_verl_rows` / :func:`write_verl_parquet`.
Sandbox/agent/reward settings come from ``$AGENTBOX_VERL_CONFIG`` (YAML/JSON with
``sandbox``, ``agent``, ``rewards`` keys) or defaults.
"""

from __future__ import annotations

import contextvars
import json
import os
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentbox.agent.prompts import render_system_prompt
from agentbox.config import AgentConfig, SandboxConfig
from agentbox.env.env import AgentBoxEnv
from agentbox.rewards import RewardSpec
from agentbox.runner.provision import ProvisionError
from agentbox.runner.rollout import build_tools
from agentbox.tasks.loader import iter_tasks
from agentbox.tasks.schema import Task

_ENV: contextvars.ContextVar[AgentBoxEnv | None] = contextvars.ContextVar(
    "agentbox_verl_env", default=None
)
AGENT_NAME = "agentbox"


class Settings:
    def __init__(
        self,
        sandbox: SandboxConfig | None = None,
        agent: AgentConfig | None = None,
        rewards: list[RewardSpec] | None = None,
    ) -> None:
        self.sandbox = sandbox or SandboxConfig()
        self.agent = agent or AgentConfig()
        self.rewards = rewards


def load_settings(path: str | None = None) -> Settings:
    path = path or os.environ.get("AGENTBOX_VERL_CONFIG")
    if not path:
        return Settings()
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return Settings(
        sandbox=SandboxConfig.model_validate(data.get("sandbox", {})),
        agent=AgentConfig.model_validate(data.get("agent", {})),
        rewards=[RewardSpec.model_validate(r) for r in data["rewards"]]
        if data.get("rewards")
        else None,
    )


@lru_cache(maxsize=4)
def _task_index(source: str) -> dict[str, Task]:
    return {t.task_id: t for t in iter_tasks(source)}


def task_from_kwargs(kwargs: dict[str, Any]) -> Task:
    extra = kwargs.get("extra_info") or {}
    blob = extra.get("agentbox_task")
    if blob:
        return (
            Task.model_validate_json(blob) if isinstance(blob, str) else Task.model_validate(blob)
        )
    task_id = extra.get("task_id") or kwargs.get("task_id")
    source = os.environ.get("AGENTBOX_TASKS")
    if task_id and source:
        return _task_index(source)[str(task_id)]
    raise KeyError("dataset row has no extra_info.agentbox_task (or task_id + $AGENTBOX_TASKS)")


class _SchemaView:
    """Duck-typed stand-in for verl's ``OpenAIFunctionToolSchema`` (no validation)."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema
        fn = schema.get("function", {})
        self.type = schema.get("type", "function")
        self.function = SimpleNamespace(
            name=fn.get("name"), description=fn.get("description"), parameters=fn.get("parameters")
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._schema


def _verl_tool(schema: dict[str, Any]) -> Any:
    """What verl's loop reads from ``self.tools`` values: ``name`` and ``tool_schema``.

    verl passes ``tool.tool_schema`` objects to its tool parsers and per-sample tool
    selection; AgentBox executes the calls itself, so only the schema is needed.
    """
    tool_schema: Any
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema

        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    except Exception:  # verl not installed, or a schema verl's model cannot express
        tool_schema = _SchemaView(schema)
    return SimpleNamespace(name=schema["function"]["name"], tool_schema=tool_schema)


def build_agent_loop_class(
    base: type,
    *,
    tool_response: Callable[[str], Any] | None = None,
    settings: Settings | None = None,
) -> type:
    """Subclass verl's ``ToolAgentLoop`` (``base``) with AgentBox tools and rewards."""

    def make_response(text: str) -> Any:
        if tool_response is not None:
            return tool_response(text)
        from verl.tools.schemas import ToolResponse

        return ToolResponse(text=text)

    class AgentBoxAgentLoop(base):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._agentbox = settings or load_settings()
            registry, _ = build_tools(self._agentbox.agent, None)
            self.tool_schemas = registry.openai_tools()
            self.tools = {s["function"]["name"]: _verl_tool(s) for s in self.tool_schemas}

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            cfg = self._agentbox
            env = AgentBoxEnv(sandbox=cfg.sandbox, agent=cfg.agent, rewards=cfg.rewards)
            provision_error: ProvisionError | None = None
            try:
                await env.reset(task_from_kwargs(kwargs))
            except ProvisionError as exc:  # keep the batch alive; the sample is unscorable
                provision_error = exc
            token = _ENV.set(env if provision_error is None else None)
            try:
                output = await super().run(sampling_params, **kwargs)
                if provision_error is None:
                    result = await env.evaluate()
                    info: dict[str, Any] = {
                        "success": result.success,
                        "status": result.final_status.value,
                        "failure_kind": result.failure_kind.value,
                        "reward_breakdown": result.breakdown,
                    }
                    reward, infra = result.reward, result.failure_kind.is_infra
                else:
                    info = {
                        "failure_kind": provision_error.kind.value,
                        "error": str(provision_error),
                    }
                    reward, infra = 0.0, True
                output.reward_score = reward
                if infra:
                    # verl has no "unscorable" sample: at least never train on its tokens
                    output.response_mask = [0] * len(output.response_mask)
                info["infra_failure"] = infra
                extra = getattr(output, "extra_fields", None)
                if isinstance(extra, dict):
                    extra["agentbox"] = info
                return output
            finally:
                _ENV.reset(token)
                await env.close()

        async def _call_tool(self, tool_call: Any, tools_kwargs: Any, agent_data: Any) -> Any:
            env = _ENV.get()
            if env is None:
                return make_response("ERROR: no AgentBox environment for this episode"), 0.0, {}
            outcome = await env.call_tool(tool_call.name, tool_call.arguments)
            return make_response(outcome.message.content or ""), 0.0, {}

    # verl registers loops under f"{__module__}.{__qualname__}" and hydra re-imports
    # that path in every agent-loop worker: it must not point inside this function
    AgentBoxAgentLoop.__name__ = AgentBoxAgentLoop.__qualname__ = "AgentBoxAgentLoop"
    AgentBoxAgentLoop.__module__ = __name__
    return AgentBoxAgentLoop


def tasks_to_verl_rows(
    tasks: Iterable[Task],
    *,
    agent: AgentConfig | None = None,
    data_source: str = "agentbox",
    embed_task: bool = True,
    workspace_dir: str = "/workspace",
) -> list[dict[str, Any]]:
    """verl RL dataset rows routed to the ``agentbox`` agent loop."""
    agent = agent or AgentConfig()
    rows = []
    for t in tasks:
        system = render_system_prompt(
            workspace_dir=workspace_dir,
            mode=agent.tools,
            extra=t.system_prompt_extra,
            override=agent.system_prompt,
        )
        extra: dict[str, Any] = {"task_id": t.task_id}
        if embed_task:
            extra["agentbox_task"] = t.model_dump_json(exclude_defaults=True)
        rows.append(
            {
                "data_source": data_source,
                "agent_name": AGENT_NAME,
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": t.prompt_text()},
                ],
                "ability": "agentic_coding",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": extra,
            }
        )
    return rows


def write_verl_parquet(tasks: Iterable[Task], path: str | Path, **kwargs: Any) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = tasks_to_verl_rows(tasks, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def __getattr__(name: str) -> Any:
    """``AgentBoxAgentLoop`` is built (and registered) lazily: it needs verl."""
    if name == "AgentBoxAgentLoop":
        from verl.experimental.agent_loop.agent_loop import register
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

        cls = build_agent_loop_class(ToolAgentLoop)
        register(AGENT_NAME)(cls)
        globals()[name] = cls
        return cls
    raise AttributeError(name)
