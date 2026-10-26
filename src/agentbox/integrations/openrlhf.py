"""OpenRLHF integration (GH #73).

Pass this module as the agent function::

    python -m openrlhf.cli.train_ppo_ray ... \\
        --train.agent_func_path $(python -c "import agentbox.integrations.openrlhf as m; print(m.__file__)")

Each prompt's ``label`` selects the task (a task id resolved against
``$AGENTBOX_TASKS``, or an inline task JSON). Episodes run token-in/token-out on
OpenRLHF's own vLLM engine with its tokenizer, so the returned
``observation_tokens``/``action_ranges``/``rollout_log_probs`` are exactly what
was sampled — no chat-template re-rendering. Settings: ``$AGENTBOX_OPENRLHF_CONFIG``
(same YAML/JSON layout as the verl integration).
"""

from __future__ import annotations

import copy
import os
from typing import Any

from agentbox.config import ModelConfig, RolloutConfig
from agentbox.integrations.verl import Settings, _task_index, load_settings
from agentbox.runner.rollout import Rollout
from agentbox.tasks.schema import Task
from agentbox.tito import ChatTokenizer, GenOutput, TitoModelClient


def resolve_task(label: Any) -> Task:
    if isinstance(label, Task):
        return label
    if isinstance(label, dict):
        return Task.model_validate(label)
    text = str(label).strip()
    if text.startswith("{"):
        return Task.model_validate_json(text)
    source = os.environ.get("AGENTBOX_TASKS")
    if not source:
        raise KeyError("label is a task id but $AGENTBOX_TASKS is not set")
    return _task_index(source)[text]


def _logprob(entry: Any, token_id: int) -> float:
    if entry is None:
        return 0.0
    lp = entry.get(token_id) if isinstance(entry, dict) else None
    return float(getattr(lp, "logprob", lp or 0.0)) if lp is not None else 0.0


def engine_generator(llm_engine: Any, sampling_params: Any, max_length: int) -> Any:
    """Adapt an OpenRLHF/vLLM async engine to the TITO ``generate`` callable."""

    async def generate(prompt_ids: list[int], sampling: dict[str, Any]) -> GenOutput:
        sp = copy.copy(sampling_params)
        room = max_length - len(prompt_ids)
        if room <= 0:
            from agentbox.errors import ContextLengthError

            raise ContextLengthError(f"prompt ({len(prompt_ids)}) exceeds max_length ({max_length})")
        sp.max_tokens = min(int(getattr(sp, "max_tokens", room) or room), room)
        if getattr(sp, "logprobs", None) is None:
            sp.logprobs = 1
        result = await llm_engine.generate(prompt_ids, sp)
        out = result.outputs[0]
        ids = list(out.token_ids)
        lps = [_logprob(e, t) for e, t in zip(out.logprobs or [], ids, strict=False)] or None
        return GenOutput(token_ids=ids, text=out.text or "", logprobs=lps,
                         finish_reason=getattr(out, "finish_reason", None))

    return generate


async def run_episode(
    prompt: str,
    label: Any,
    sampling_params: Any,
    max_length: int,
    hf_tokenizer: Any,
    llm_engine: Any,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """One TITO episode in OpenRLHF's sample format."""
    cfg = settings or load_settings(os.environ.get("AGENTBOX_OPENRLHF_CONFIG"))
    task = resolve_task(label)
    chat = ChatTokenizer(hf_tokenizer)
    client = TitoModelClient(
        chat,
        engine_generator(llm_engine, sampling_params, max_length),
        model=getattr(hf_tokenizer, "name_or_path", "policy"),
        max_context_tokens=max_length,
    )
    agent = cfg.agent.model_copy(update={"tool_protocol": "native"})
    traj = await Rollout.run(
        task,
        model=client,
        agent=agent,
        sandbox=cfg.sandbox,
        config=RolloutConfig(model=ModelConfig(model="tito"), agent=agent, rewards=cfg.rewards),
    )
    trace = traj.token_trace()
    if trace is None or not trace.sequences:
        raise RuntimeError(f"episode produced no tokens ({traj.final_status.value}: {traj.error})")
    seq = trace.sequences[0]
    obs = seq.prompt_ids + seq.response_ids
    ranges: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(seq.response_mask):
        pos = len(seq.prompt_ids) + i
        if m and start is None:
            start = pos
        elif not m and start is not None:
            ranges.append((start, pos))
            start = None
    if start is not None:
        ranges.append((start, len(obs)))
    log_probs = None
    if seq.response_logprobs is not None:
        log_probs = [0.0] * len(seq.prompt_ids) + list(seq.response_logprobs)
    return {
        "prompt": prompt,
        "label": label,
        "observation_tokens": obs,
        "action_ranges": ranges,
        "rollout_log_probs": log_probs,
        "truncated": traj.stop_reason == "context_length",
        "reward": traj.reward,
        "scores": traj.reward,
        "extra_logs": {
            "agentbox_success": float(traj.success),
            "agentbox_steps": float(traj.metrics.steps),
            "agentbox_tool_errors": float(traj.metrics.tool_errors),
            "agentbox_sequences": float(len(trace.sequences)),
        },
    }


def __getattr__(name: str) -> Any:
    """``AgentExecutor`` is built lazily (needs ``openrlhf``)."""
    if name == "AgentExecutor":
        from openrlhf.utils.agent import AgentExecutorBase

        class AgentExecutor(AgentExecutorBase):  # type: ignore[misc, valid-type]
            async def execute(  # type: ignore[override]
                self, prompt: Any, label: Any, sampling_params: Any, max_length: int,
                hf_tokenizer: Any, llm_engine: Any,
            ) -> dict[str, Any]:
                return await run_episode(
                    prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine
                )

        globals()[name] = AgentExecutor
        return AgentExecutor
    raise AttributeError(name)
