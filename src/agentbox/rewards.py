"""Composable, trainer-agnostic reward functions (GH #51).

A run's reward is a weighted sum of named components::

    rewards:
      - {name: verifier}                         # task verifier reward (ground truth)
      - {name: step_penalty, params: {per_step: 0.01}}
      - {name: tool_error_penalty, params: {per_error: 0.05}}
      - {name: my_pkg.rewards:style_bonus, weight: 0.2}   # any importable callable

Every component sees a :class:`RewardContext`; the per-component values land in
``Trajectory.reward_breakdown``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentbox.agent.loop import AgentLoopResult
    from agentbox.config import AgentConfig
    from agentbox.tasks.schema import Task
    from agentbox.tasks.verifier import VerifyResult

logger = logging.getLogger("agentbox.rewards")

RewardValue = float | Awaitable[float]
RewardFn = Callable[["RewardContext"], RewardValue]


@dataclass
class RewardContext:
    task: Task
    verify: VerifyResult | None
    loop: AgentLoopResult | None

    @property
    def steps(self) -> int:
        return self.loop.steps if self.loop else 0

    @property
    def stop_reason(self) -> str | None:
        return self.loop.stop_reason if self.loop else None


class RewardSpec(BaseModel):
    name: str
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


_REGISTRY: dict[str, Callable[..., RewardFn]] = {}


def register_reward(name: str) -> Callable[[Callable[..., RewardFn]], Callable[..., RewardFn]]:
    """Register a reward *factory*: ``factory(**params) -> RewardFn``."""

    def deco(factory: Callable[..., RewardFn]) -> Callable[..., RewardFn]:
        _REGISTRY[name] = factory
        return factory

    return deco


def available_rewards() -> list[str]:
    return sorted(_REGISTRY)


def resolve_reward(spec: RewardSpec) -> RewardFn:
    factory = _REGISTRY.get(spec.name)
    if factory is not None:
        return factory(**spec.params)
    if ":" in spec.name or "." in spec.name:
        from agentbox.tasks.verifier import load_callable

        fn = load_callable(spec.name)
        if spec.params:
            return lambda ctx: fn(ctx, **spec.params)  # type: ignore[no-any-return]
        return fn  # type: ignore[return-value]
    raise KeyError(f"unknown reward {spec.name!r}; available: {available_rewards()}")


async def compute_reward(
    specs: list[RewardSpec], ctx: RewardContext
) -> tuple[float, dict[str, float]]:
    total = 0.0
    breakdown: dict[str, float] = {}
    for spec in specs:
        value = resolve_reward(spec)(ctx)
        if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
            value = await value
        weighted = spec.weight * float(value)  # type: ignore[arg-type]
        key = spec.name
        n = 2
        while key in breakdown:
            key = f"{spec.name}#{n}"
            n += 1
        breakdown[key] = weighted
        total += weighted
    return total, breakdown


def default_reward_specs(agent: AgentConfig | None = None) -> list[RewardSpec]:
    """Verifier reward plus the legacy AgentConfig penalties (backwards compatible)."""
    specs = [RewardSpec(name="verifier")]
    if agent is not None:
        if agent.step_penalty:
            specs.append(RewardSpec(name="step_penalty", params={"per_step": agent.step_penalty}))
        if agent.timeout_penalty:
            specs.append(
                RewardSpec(name="timeout_penalty", params={"penalty": agent.timeout_penalty})
            )
        if agent.max_steps_penalty:
            specs.append(
                RewardSpec(name="max_steps_penalty", params={"penalty": agent.max_steps_penalty})
            )
    return specs


# --------------------------------------------------------------------- built-ins
@register_reward("verifier")
def _verifier() -> RewardFn:
    return lambda ctx: ctx.verify.reward if ctx.verify is not None else 0.0


@register_reward("success")
def _success(value: float = 1.0) -> RewardFn:
    return lambda ctx: value if ctx.verify is not None and ctx.verify.success else 0.0


@register_reward("tests_fraction")
def _tests_fraction() -> RewardFn:
    def fn(ctx: RewardContext) -> float:
        v = ctx.verify
        if v is None or not v.tests_total:
            return 0.0
        return max(0.0, (v.tests_passed or 0) / v.tests_total)

    return fn


@register_reward("step_penalty")
def _step_penalty(per_step: float = 0.01) -> RewardFn:
    return lambda ctx: -per_step * max(0, ctx.steps)


@register_reward("timeout_penalty")
def _timeout_penalty(penalty: float = 0.5) -> RewardFn:
    return lambda ctx: -penalty if ctx.stop_reason == "timeout" else 0.0


@register_reward("max_steps_penalty")
def _max_steps_penalty(penalty: float = 0.5) -> RewardFn:
    return lambda ctx: -penalty if ctx.stop_reason == "max_steps" else 0.0


@register_reward("tool_error_penalty")
def _tool_error_penalty(per_error: float = 0.05, cap: float | None = None) -> RewardFn:
    def fn(ctx: RewardContext) -> float:
        n = sum(1 for r in ctx.loop.tool_call_records if r.is_error) if ctx.loop else 0
        value = per_error * n
        return -(min(value, cap) if cap is not None else value)

    return fn


@register_reward("valid_tool_call_ratio")
def _valid_tool_calls() -> RewardFn:
    """Format reward: fraction of tool calls that parsed and named a real tool."""

    def fn(ctx: RewardContext) -> float:
        records = ctx.loop.tool_call_records if ctx.loop else []
        if not records:
            return 1.0
        bad = sum(
            1
            for r in records
            if r.result.startswith(("ERROR: unknown tool", "ERROR: invalid arguments"))
        )
        return 1.0 - bad / len(records)

    return fn


@register_reward("length_penalty")
def _length_penalty(per_token: float = 0.0001, budget: int = 0) -> RewardFn:
    def fn(ctx: RewardContext) -> float:
        tokens = ctx.loop.usage_totals.get("completion_tokens", 0) if ctx.loop else 0
        return -per_token * max(0, tokens - budget)

    return fn


@register_reward("submitted")
def _submitted(value: float = 1.0) -> RewardFn:
    return lambda ctx: value if ctx.stop_reason == "submitted" else 0.0


@register_reward("tamper_penalty")
def _tamper_penalty(penalty: float = 1.0) -> RewardFn:
    return lambda ctx: -penalty if ctx.verify is not None and ctx.verify.tampered else 0.0


_JUDGE_PROMPT = """You are grading an AI coding agent's attempt at a task.
Task:
{task}

Verifier result: {verify}

Final transcript excerpt:
{transcript}

{rubric}
Reply with a JSON object: {{"score": <float between 0 and 1>, "reason": "<short>"}}."""


@register_reward("llm_judge")
def _llm_judge(
    model: dict[str, Any] | None = None,
    rubric: str = "Score how well the agent solved the task (1 = fully, 0 = not at all).",
    max_transcript_chars: int = 6000,
) -> RewardFn:
    """Rubric score from any OpenAI-compatible model (RULER-style judges, GH #75)."""
    if not model:
        raise ValueError("llm_judge requires params.model (a ModelConfig dict)")

    async def fn(ctx: RewardContext) -> float:
        from agentbox.config import ModelConfig
        from agentbox.model.openai_compat import OpenAICompatClient
        from agentbox.utils.text import truncate_middle

        cfg = ModelConfig(**{"temperature": 0.0, "max_tokens": 256, **model})
        lines = []
        for m in ctx.loop.messages[2:] if ctx.loop else []:
            role = m.role.value if hasattr(m.role, "value") else m.role
            body = m.content or ""
            if m.tool_calls:
                body += " " + " ".join(f"{t.function.name}({t.function.arguments})" for t in m.tool_calls)
            lines.append(f"[{role}] {body}")
        verify = "n/a"
        if ctx.verify is not None:
            verify = f"success={ctx.verify.success} exit={ctx.verify.exit_code}"
        prompt = _JUDGE_PROMPT.format(
            task=ctx.task.description,
            verify=verify,
            transcript=truncate_middle("\n".join(lines), max_transcript_chars),
            rubric=rubric,
        )
        async with OpenAICompatClient(cfg) as client:
            resp = await client.complete([{"role": "user", "content": prompt}])
        text = resp.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            score = float(json.loads(m.group(0))["score"]) if m else float(text.strip())
        except (ValueError, KeyError, TypeError):
            logger.warning("llm_judge returned unparsable output: %r", text[:200])
            return 0.0
        return max(0.0, min(1.0, score))

    return fn
