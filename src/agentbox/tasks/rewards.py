"""Reward shaping helpers (binary + optional process / step costs)."""

from __future__ import annotations

from agentbox.tasks.verifier import VerifyResult
from agentbox.types import FinalStatus


def binary_reward(verify: VerifyResult) -> float:
    return verify.reward


def shaped_reward(
    verify: VerifyResult,
    *,
    steps: int = 0,
    step_penalty: float = 0.0,
    timeout: bool = False,
    timeout_penalty: float = 0.0,
    max_steps_hit: bool = False,
    max_steps_penalty: float = 0.0,
) -> float:
    """Start from verifier reward and apply optional process penalties."""
    r = float(verify.reward)
    if step_penalty:
        r -= step_penalty * max(0, steps)
    if timeout and timeout_penalty:
        r -= timeout_penalty
    if max_steps_hit and max_steps_penalty:
        r -= max_steps_penalty
    return r


def status_from_verify(
    success: bool,
    *,
    stop_reason: str,
) -> FinalStatus:
    if success:
        return FinalStatus.SUCCESS
    if stop_reason == "timeout":
        return FinalStatus.TIMEOUT
    if stop_reason == "max_steps":
        return FinalStatus.MAX_STEPS
    if stop_reason == "error":
        return FinalStatus.ERROR
    return FinalStatus.FAILED
