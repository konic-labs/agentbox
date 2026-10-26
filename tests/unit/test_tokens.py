"""Token tracking and trace assembly."""

from __future__ import annotations

from agentbox.trajectory.tokens import (
    TokenTracker,
    TurnTokens,
    build_token_trace,
    iter_turn_samples,
)


def test_prefix_consistent_turns_make_one_sequence() -> None:
    tr = TokenTracker()
    t1 = tr.observe([1, 2, 3], [10, 11], [-0.1, -0.2])
    t2 = tr.observe([1, 2, 3, 10, 11, 20, 21], [12], [-0.3])
    assert t1 is not None and t1.prompt_reuse == 0 and t1.prompt_ids == [1, 2, 3]
    assert t2 is not None and t2.prompt_reuse == 5 and t2.prompt_ids == [20, 21]
    assert not t2.prefix_break
    trace = build_token_trace([t1, t2])
    assert trace is not None and len(trace.sequences) == 1 and trace.exact
    seq = trace.sequences[0]
    assert seq.prompt_ids == [1, 2, 3]
    assert seq.response_ids == [10, 11, 20, 21, 12]
    assert seq.response_mask == [1, 1, 0, 0, 1]
    assert seq.response_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3]
    assert seq.num_turns == 2


def test_prefix_break_reuses_common_prefix_exactly() -> None:
    tr = TokenTracker()
    t1 = tr.observe([1, 2], [10], None)
    t2 = tr.observe([1, 9, 10, 30], [11], None)  # history re-rendered differently
    assert t2 is not None and t2.prefix_break
    assert t2.prompt_reuse == 1 and t2.prompt_ids == [9, 10, 30]  # only the suffix is stored
    trace = build_token_trace([t1, t2])
    assert trace is not None and trace.prefix_breaks == 1 and not trace.has_logprobs
    assert trace.sequences[1].prompt_ids == [1, 9, 10, 30]
    samples = list(iter_turn_samples([t1, t2]))
    assert samples == [([1, 2], [10], None), ([1, 9, 10, 30], [11], None)]


def test_missing_tokens() -> None:
    assert build_token_trace([None, None]) is None
    trace = build_token_trace([TurnTokens(completion_ids=[1]), None])
    assert trace is not None and not trace.exact
    tr = TokenTracker()
    assert tr.observe(None, None, None) is None
    only_completion = tr.observe(None, [4, 5], [-1.0, -1.0])
    assert only_completion is not None and only_completion.prompt_ids is None
