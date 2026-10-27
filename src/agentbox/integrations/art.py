"""OpenPipe ART adapter (GH #75, #67).

AgentBox no longer depends on ART; this module converts trajectories into ART's
``Trajectory`` / ``TrajectoryGroup`` shapes when ``openpipe-art`` is installed.

ART trains on assistant turns given as OpenAI ``Choice`` objects with logprobs
(plain assistant dicts only with ``allow_training_without_logprobs``) and
assembles its training sequences from vLLM's ``prompt_token_ids`` /
``token_ids`` on each Choice. When a rollout captured token ids + logprobs
(``ModelConfig.capture_tokens`` and ``capture_logprobs``), every assistant turn
whose full prompt can be rebuilt from the trace becomes such a ``Choice``
(logprob tokens also use the ``token_id:<id>`` form older ART releases read), so
ART trains on exactly the sampled tokens and never re-tokenizes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Literal

from agentbox.trajectory.schema import Message, Trajectory
from agentbox.trajectory.tokens import iter_message_samples
from agentbox.types import FinalStatus, MessageRole

_SCALARS = (str, int, float, bool, type(None))


def _scalar(value: Any) -> Any:
    """ART releases before the trajectories refactor accept only scalar metadata."""
    if isinstance(value, _SCALARS):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "function_call",
}


def art_choice(message: Message, prompt_ids: list[int] | None) -> Any | None:
    """OpenAI ``Choice`` for an assistant turn with full token data, else None.

    ``prompt_ids`` is the turn's complete prompt (see
    :func:`agentbox.trajectory.tokens.iter_message_samples`); without it ART
    cannot place the turn, so no Choice is built.
    """
    tok = message.tokens
    if (
        message.role != MessageRole.ASSISTANT
        or prompt_ids is None
        or tok is None
        or tok.logprobs is None
        or len(tok.logprobs) != len(tok.completion_ids)
        or not tok.completion_ids
    ):
        return None
    from openai.types.chat.chat_completion import Choice

    wire = message.to_openai_dict(include_reasoning=False)
    finish = _FINISH_REASONS.get(
        message.finish_reason or "", "tool_calls" if message.tool_calls else "stop"
    )
    return Choice.model_validate(
        {
            "index": 0,
            "finish_reason": finish,
            "message": {
                "role": "assistant",
                "content": wire.get("content"),
                "tool_calls": wire.get("tool_calls") or None,
            },
            "logprobs": {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": lp, "bytes": None, "top_logprobs": []}
                    for tid, lp in zip(tok.completion_ids, tok.logprobs, strict=True)
                ]
            },
            # vLLM return_token_ids fields, read by ART to assemble training sequences
            "prompt_token_ids": list(prompt_ids),
            "token_ids": list(tok.completion_ids),
        }
    )


def _messages_and_choices(traj: Trajectory, *, choices: bool, as_dicts: bool) -> list[Any]:
    prompts = {i: p for i, p, _, _ in iter_message_samples(traj.messages)} if choices else {}
    out: list[Any] = []
    for i, m in enumerate(traj.messages):
        choice = art_choice(m, prompts.get(i)) if choices else None
        if choice is None:
            out.append(m.to_openai_dict(include_reasoning=True))
        else:
            out.append(choice.model_dump(mode="json", exclude_none=True) if as_dicts else choice)
    return out


def to_art_dict(
    traj: Trajectory,
    *,
    tools: list[dict[str, Any]] | None = None,
    scalar_metadata: bool = True,
    choices: bool = True,
) -> dict[str, Any]:
    """Portable dict in ART ``Trajectory`` shape.

    With ``choices`` (default), assistant turns that carry token logprobs are
    serialised ``Choice`` objects (trainable by ART); other turns stay messages.
    """
    messages_and_choices = _messages_and_choices(traj, choices=choices, as_dicts=True)
    if traj.verify is not None:
        correct = 1.0 if traj.verify.success else 0.0
    else:
        correct = 1.0 if traj.final_status == FinalStatus.SUCCESS else 0.0
    metrics: dict[str, float | int | bool] = {
        "duration": traj.metrics.duration_s,
        "steps": traj.metrics.steps,
        "tool_calls": traj.metrics.tool_calls,
        "tool_errors": traj.metrics.tool_errors,
        "correct": correct,
    }
    if traj.metrics.model_calls:
        metrics["model_calls"] = traj.metrics.model_calls
    if traj.metrics.total_tokens is not None:
        metrics["total_tokens"] = traj.metrics.total_tokens
    for name, value in traj.reward_breakdown.items():
        metrics[f"reward/{name}"] = value

    metadata: dict[str, Any] = {
        "task_id": traj.task_id,
        "run_id": traj.run_id,
        "rollout_id": traj.rollout_id,
        "final_status": traj.final_status.value,
        "failure_kind": str(getattr(traj.failure_kind, "value", traj.failure_kind)),
        **traj.metadata,
    }
    if traj.group_id is not None:
        metadata["group_id"] = traj.group_id
    if traj.stop_reason is not None:
        metadata["stop_reason"] = traj.stop_reason
    if traj.verify is not None:
        metadata["verify_exit_code"] = traj.verify.exit_code
        metadata["verify_success"] = traj.verify.success
    if traj.model:
        metadata["model"] = traj.model
    if traj.error:
        metadata["error"] = traj.error
    if scalar_metadata:
        metadata = {k: _scalar(v) for k, v in metadata.items()}

    out: dict[str, Any] = {
        "messages_and_choices": messages_and_choices,
        "reward": traj.reward,
        "metrics": metrics,
        "metadata": metadata,
    }
    tool_schemas = tools if tools is not None else traj.tools
    if tool_schemas:
        out["tools"] = tool_schemas
    return out


def _art() -> Any:
    try:
        import art
    except ImportError as exc:
        raise ImportError("openpipe-art is required: pip install 'agentbox-rl[art]'") from exc
    return art


def to_art_trajectory(traj: Trajectory, *, choices: bool = True) -> Any:
    """A live ``art.Trajectory`` (``Choice`` objects for turns with token logprobs)."""
    art = _art()
    data = to_art_dict(traj, choices=False)
    return art.Trajectory(
        messages_and_choices=_messages_and_choices(traj, choices=choices, as_dicts=False),
        tools=data.get("tools"),
        reward=data["reward"],
        metrics=data["metrics"],
        metadata=data["metadata"],
    )


def to_art_groups(trajectories: Iterable[Trajectory], *, trainable_only: bool = True) -> list[Any]:
    """``art.TrajectoryGroup`` per ``group_id`` (infra failures dropped by default)."""
    art = _art()
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        if trainable_only and not t.is_trainable:
            continue
        groups.setdefault(t.group_id or t.task_id, []).append(t)
    return [art.TrajectoryGroup([to_art_trajectory(t) for t in g]) for g in groups.values()]
