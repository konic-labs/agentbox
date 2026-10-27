"""Token-level trajectory data for RL trainers (token-in / token-out).

Each assistant turn may carry :class:`TurnTokens` captured from the server
(vLLM ``return_token_ids`` + logprobs). The turn's exact prompt is stored as::

    prompt = previous_turn(prompt + completion)[:prompt_reuse] + prompt_ids

``prompt_reuse`` is the longest common prefix with the previous turn, so
storage stays ~O(total tokens) even when the server re-renders history (chat
templates that drop past ``<think>`` blocks or re-serialise tool-call JSON
cause a *prefix break*: the previous completion is not a prefix of the next
prompt). Breaks are exact, just not concatenable.

:func:`build_token_trace` assembles turns into training sequences in the
verl / OpenRLHF / SkyRL layout (one sequence per unbroken run of turns)::

    prompt_ids | response_ids  = gen_1 env_2 gen_2 env_3 gen_3 ...
               | response_mask =  1..1  0..0  1..1  0..0  1..1

:func:`iter_turn_samples` yields every turn as an exact
``(prompt_ids, completion_ids, logprobs)`` sample regardless of breaks.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, Field


class TurnTokens(BaseModel):
    """Exact token data for one assistant turn."""

    completion_ids: list[int]
    logprobs: list[float] | None = None
    prompt_reuse: int = Field(
        default=0, description="Leading prompt tokens shared with the previous turn's sequence"
    )
    prompt_ids: list[int] | None = Field(
        default=None, description="Prompt tokens after the reused prefix (None: not captured)"
    )
    prefix_break: bool = False


def _common_prefix(a: array[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    # fast path: whole prefix equal
    if n and array("q", b[:n]) == a[:n]:
        return n
    lo = 0
    while lo < n and a[lo] == b[lo]:
        lo += 1
    return lo


class TokenTracker:
    """Turns server token data into compact :class:`TurnTokens` (O(n) state)."""

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
        reuse = _common_prefix(prev, prompt_ids) if prev is not None else 0
        turn = TurnTokens(
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_reuse=reuse,
            prompt_ids=prompt_ids[reuse:],
            prefix_break=prev is not None and reuse < len(prev),
        )
        seq = array("q", prompt_ids)
        seq.extend(completion_ids)
        self._sequence = seq
        return turn


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
    """Training sequences for a trajectory (one per unbroken run of turns)."""

    sequences: list[TokenSequence] = Field(default_factory=list)
    exact: bool = True
    has_logprobs: bool = False

    @property
    def prefix_breaks(self) -> int:
        return max(0, len(self.sequences) - 1)


def _samples(
    turns: Iterable[tuple[int, TurnTokens | None]],
) -> Iterator[tuple[int, list[int], list[int], list[float] | None]]:
    prev: list[int] | None = None
    for key, turn in turns:
        if turn is None or turn.prompt_ids is None:
            prev = None
            continue
        if turn.prompt_reuse and prev is None:
            continue  # cannot reconstruct without the previous turn
        prompt = (prev[: turn.prompt_reuse] if prev is not None else []) + turn.prompt_ids
        yield key, prompt, turn.completion_ids, turn.logprobs
        prev = prompt + turn.completion_ids


def iter_turn_samples(
    turns: Iterable[TurnTokens | None],
) -> Iterator[tuple[list[int], list[int], list[float] | None]]:
    """Exact per-turn ``(prompt_ids, completion_ids, logprobs)``; skips uncaptured turns."""
    for _, prompt, completion, logprobs in _samples(enumerate(turns)):
        yield prompt, completion, logprobs


def iter_message_samples(
    messages: Iterable[Any],
) -> Iterator[tuple[int, list[int], list[int], list[float] | None]]:
    """``(message_index, prompt_ids, completion_ids, logprobs)`` per reconstructable turn."""
    return _samples(
        (i, getattr(m, "tokens", None))
        for i, m in enumerate(messages)
        if (m.role.value if hasattr(m.role, "value") else m.role) == "assistant"
    )


def build_token_trace(turns: Iterable[TurnTokens | None]) -> TokenTrace | None:
    """Assemble per-turn tokens into sequences; ``None`` if nothing was captured."""
    sequences: list[TokenSequence] = []
    current: TokenSequence | None = None
    have_logprobs = True
    exact = True
    any_turn = False
    prev_full: list[int] | None = None
    for turn in turns:
        if turn is None:
            exact = False
            prev_full = None
            current = None
            continue
        any_turn = True
        lps = turn.logprobs
        if lps is None or len(lps) != len(turn.completion_ids):
            have_logprobs = False
        if turn.prompt_ids is None or (turn.prompt_reuse and prev_full is None):
            exact = False  # prompt not captured: cannot anchor this turn
            prev_full = None
            current = None
            continue
        continues = (
            current is not None
            and prev_full is not None
            and not turn.prefix_break
            and turn.prompt_reuse == len(prev_full)
        )
        if continues and current is not None:
            env = turn.prompt_ids
            current.response_ids.extend(env)
            current.response_mask.extend([0] * len(env))
            if current.response_logprobs is not None:
                current.response_logprobs.extend([0.0] * len(env))
            prompt_full = (prev_full or []) + env
        else:
            prompt_full = (prev_full[: turn.prompt_reuse] if prev_full else []) + turn.prompt_ids
            current = TokenSequence(prompt_ids=prompt_full, response_logprobs=[])
            sequences.append(current)
        current.response_ids.extend(turn.completion_ids)
        current.response_mask.extend([1] * len(turn.completion_ids))
        if current.response_logprobs is not None:
            if lps is not None and len(lps) == len(turn.completion_ids):
                current.response_logprobs.extend(lps)
            else:
                current.response_logprobs = None
        current.num_turns += 1
        prev_full = prompt_full + turn.completion_ids
    if not any_turn:
        return None
    if not have_logprobs:
        for seq in sequences:
            seq.response_logprobs = None
    return TokenTrace(sequences=sequences, exact=exact, has_logprobs=have_logprobs)


def _assistant_turns(messages: Iterable[Any]) -> list[TurnTokens | None]:
    return [
        getattr(m, "tokens", None)
        for m in messages
        if (m.role.value if hasattr(m.role, "value") else m.role) == "assistant"
    ]


def trace_from_messages(messages: Iterable[Any]) -> TokenTrace | None:
    """Build a :class:`TokenTrace` from assistant messages carrying ``tokens``."""
    return build_token_trace(_assistant_turns(messages))


def turn_samples_from_messages(
    messages: Iterable[Any],
) -> list[tuple[list[int], list[int], list[float] | None]]:
    return list(iter_turn_samples(_assistant_turns(messages)))
