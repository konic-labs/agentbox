"""Token-level trajectory data for RL trainers (token-in / token-out).

Each assistant turn may carry :class:`TurnTokens` captured from the server
(vLLM ``return_token_ids`` + logprobs). Storage is O(total tokens): the full
prompt is kept only for the first turn or after a *prefix break* (the server
re-rendered earlier turns differently, e.g. a chat template that drops past
``<think>`` blocks or re-serialises tool-call JSON); otherwise only the new
prompt tokens since the previous completion (tool outputs, chat scaffolding)
are stored.

:func:`build_token_trace` turns the turns into training sequences in the
layout verl / OpenRLHF / SkyRL use::

    prompt_ids | response_ids = gen_1 env_2 gen_2 env_3 gen_3 ...
               | response_mask =  1..1  0..0  1..1  0..0  1..1
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field


class TurnTokens(BaseModel):
    """Exact token data for one assistant turn."""

    completion_ids: list[int]
    logprobs: list[float] | None = None
    prompt_ids: list[int] | None = Field(
        default=None, description="Full prompt (first turn or after a prefix break)"
    )
    prompt_delta_ids: list[int] | None = Field(
        default=None, description="Prompt tokens appended since the previous completion"
    )
    prefix_break: bool = False


class TokenTracker:
    """Decides how to store each turn's tokens (full prompt vs. delta)."""

    def __init__(self) -> None:
        self._sequence: array[int] | None = None  # prompt + completion of the last turn

    def observe(
        self,
        prompt_ids: list[int] | None,
        completion_ids: list[int] | None,
        logprobs: list[float] | None,
    ) -> TurnTokens | None:
        if completion_ids is None:
            return None
        if prompt_ids is None:
            self._sequence = None
            return TurnTokens(completion_ids=completion_ids, logprobs=logprobs)
        prev = self._sequence
        if prev is not None and len(prompt_ids) >= len(prev) and _is_prefix(prev, prompt_ids):
            turn = TurnTokens(
                completion_ids=completion_ids,
                logprobs=logprobs,
                prompt_delta_ids=prompt_ids[len(prev) :],
            )
        else:
            turn = TurnTokens(
                completion_ids=completion_ids,
                logprobs=logprobs,
                prompt_ids=prompt_ids,
                prefix_break=prev is not None,
            )
        seq = array("q", prompt_ids)
        seq.extend(completion_ids)
        self._sequence = seq
        return turn


def _is_prefix(prefix: array[int], ids: list[int]) -> bool:
    return array("q", ids[: len(prefix)]) == prefix


class TokenSequence(BaseModel):
    """One contiguous training sequence."""

    prompt_ids: list[int]
    response_ids: list[int] = Field(default_factory=list)
    response_mask: list[int] = Field(default_factory=list)
    response_logprobs: list[float] | None = None
    num_turns: int = 0

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


class TokenTrace(BaseModel):
    """Training sequences for a trajectory (normally exactly one)."""

    sequences: list[TokenSequence] = Field(default_factory=list)
    exact: bool = True
    has_logprobs: bool = False

    @property
    def prefix_breaks(self) -> int:
        return max(0, len(self.sequences) - 1)


def build_token_trace(turns: Iterable[TurnTokens | None]) -> TokenTrace | None:
    """Assemble per-turn tokens into sequences; ``None`` if nothing was captured."""
    sequences: list[TokenSequence] = []
    current: TokenSequence | None = None
    have_logprobs = True
    exact = True
    any_turn = False
    for turn in turns:
        if turn is None:
            exact = False
            continue
        any_turn = True
        lps = turn.logprobs
        if lps is None or len(lps) != len(turn.completion_ids):
            have_logprobs = False
        if turn.prompt_ids is not None:
            current = TokenSequence(prompt_ids=list(turn.prompt_ids), response_logprobs=[])
            sequences.append(current)
        elif current is None:
            exact = False  # prompt never captured: cannot anchor the sequence
            continue
        elif turn.prompt_delta_ids:
            current.response_ids.extend(turn.prompt_delta_ids)
            current.response_mask.extend([0] * len(turn.prompt_delta_ids))
            if current.response_logprobs is not None:
                current.response_logprobs.extend([0.0] * len(turn.prompt_delta_ids))
        current.response_ids.extend(turn.completion_ids)
        current.response_mask.extend([1] * len(turn.completion_ids))
        if current.response_logprobs is not None:
            if lps is not None and len(lps) == len(turn.completion_ids):
                current.response_logprobs.extend(lps)
            else:
                current.response_logprobs = None
        current.num_turns += 1
    if not any_turn:
        return None
    for seq in sequences:
        if not have_logprobs:
            seq.response_logprobs = None
    return TokenTrace(sequences=sequences, exact=exact, has_logprobs=have_logprobs)


def trace_from_messages(messages: Iterable[Any]) -> TokenTrace | None:
    """Build a :class:`TokenTrace` from assistant messages carrying ``tokens``."""
    turns = [
        getattr(m, "tokens", None)
        for m in messages
        if (m.role.value if hasattr(m.role, "value") else m.role) == "assistant"
    ]
    return build_token_trace(turns)
