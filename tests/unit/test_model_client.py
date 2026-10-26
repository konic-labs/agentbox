"""OpenAICompatClient: params, typed retries, context handling, parsing (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from agentbox.config import ModelConfig
from agentbox.errors import ContextLengthError, ModelError
from agentbox.model.openai_compat import OpenAICompatClient, is_context_length_error
from agentbox.model.replay import ReplayModelClient
from agentbox.trajectory.schema import FunctionCall, Message, ToolCall


def _http(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "http://test/v1/chat/completions"))


def _resp(content: str = "ok", **choice_extra: Any) -> Any:
    message = SimpleNamespace(
        content=content, tool_calls=None, **choice_extra.pop("message_extra", {})
    )
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None, **choice_extra)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **params: Any) -> Any:
        self.calls.append(params)
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def client_with(outcomes: list[Any], **cfg: Any) -> tuple[OpenAICompatClient, FakeCompletions]:
    fake = FakeCompletions(outcomes)
    oa = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    config = ModelConfig(model="m", retry_backoff_s=0.0001, **cfg)
    return OpenAICompatClient(config, client=oa), fake  # type: ignore[arg-type]


async def test_build_params_passthrough() -> None:
    c, fake = client_with(
        [_resp()],
        temperature=None,
        max_tokens=64,
        max_tokens_param="max_completion_tokens",
        seed=7,
        stop=["END"],
        top_k=20,
        min_p=0.05,
        capture_tokens=True,
        capture_logprobs=True,
        tool_choice="required",
        parallel_tool_calls=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    await c.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    p = fake.calls[0]
    assert "temperature" not in p
    assert p["max_completion_tokens"] == 64 and "max_tokens" not in p
    assert p["seed"] == 7 and p["stop"] == ["END"]
    assert p["tool_choice"] == "required" and p["parallel_tool_calls"] is False
    assert p["logprobs"] is True
    assert p["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
        "min_p": 0.05,
        "return_token_ids": True,
    }


async def test_retries_retryable_then_succeeds() -> None:
    c, fake = client_with(
        [
            openai.RateLimitError("slow down", response=_http(429), body=None),
            openai.InternalServerError("boom", response=_http(502), body=None),
            _resp("finally"),
        ]
    )
    out = await c.complete([{"role": "user", "content": "x"}])
    assert out.content == "finally"
    assert len(fake.calls) == 3


async def test_no_retry_on_auth_and_bounded_retries() -> None:
    c, fake = client_with([openai.AuthenticationError("bad key", response=_http(401), body=None)])
    with pytest.raises(ModelError):
        await c.complete([])
    assert len(fake.calls) == 1
    errors = [openai.APITimeoutError(request=httpx.Request("POST", "http://t")) for _ in range(3)]
    c, fake = client_with(errors, max_retries=2)
    with pytest.raises(ModelError):
        await c.complete([])
    assert len(fake.calls) == 3


async def test_context_length_shrinks_max_tokens_once() -> None:
    msg = (
        "'max_tokens' is too large: 4096. This model's maximum context length is 8192 tokens "
        "and your request has 5000 input tokens (4096 > 8192 - 5000)."
    )
    c, fake = client_with(
        [openai.BadRequestError(msg, response=_http(400), body=None), _resp("fits")],
        max_tokens=4096,
    )
    out = await c.complete([])
    assert out.content == "fits"
    assert fake.calls[1]["max_tokens"] == 3192


async def test_context_length_without_room_raises() -> None:
    msg = (
        "This model's maximum context length is 8192 tokens and your request has 8190 input tokens"
    )
    c, _ = client_with([openai.BadRequestError(msg, response=_http(400), body=None)])
    with pytest.raises(ContextLengthError):
        await c.complete([])
    assert is_context_length_error("Error: prompt is too long for this model")
    assert not is_context_length_error("invalid temperature")


async def test_parse_reasoning_think_tokens_and_usage() -> None:
    raw = _resp(
        "<think>plan</think>answer",
        message_extra={"reasoning": "server-side"},
        token_ids=[5, 6, 7],
    )
    raw.prompt_token_ids = [1, 2, 3]
    raw.choices[0].logprobs = SimpleNamespace(
        content=[SimpleNamespace(token="a", logprob=-0.1) for _ in range(3)]
    )
    raw.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=8)
    c, _ = client_with([raw], strip_think=True, capture_tokens=True, capture_logprobs=True)
    out = await c.complete([])
    assert out.content == "answer"
    assert out.reasoning_content == "server-side\nplan"
    assert out.prompt_token_ids == [1, 2, 3] and out.completion_token_ids == [5, 6, 7]
    assert out.completion_logprobs == [-0.1, -0.1, -0.1]
    assert out.usage and out.usage["cached_tokens"] == 8
    assert out.raw_choice is None  # record_raw is opt-in


async def test_token_ids_from_logprob_tokens() -> None:
    raw = _resp("x")
    raw.choices[0].logprobs = SimpleNamespace(
        content=[SimpleNamespace(token=f"token_id:{i}", logprob=-1.0) for i in (11, 12)]
    )
    c, _ = client_with([raw], capture_logprobs=True)
    out = await c.complete([])
    assert out.completion_token_ids == [11, 12]


async def test_replay_client_returns_recorded_turns() -> None:
    msgs = [
        Message(role="system", content="s"),
        Message(role="user", content="u"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="c1", function=FunctionCall(name="list_files", arguments="{}"))
            ],
        ),
        Message(role="tool", content="x", tool_call_id="c1"),
        Message(role="assistant", content="done"),
    ]
    replay = ReplayModelClient(msgs)
    first = await replay.complete([{"role": "user"}])
    assert first.tool_calls[0].id == "c1"
    second = await replay.complete([{"role": "assistant"}])
    assert second.content == "done"
    third = await replay.complete([{"role": "assistant"}, {"role": "assistant"}])
    assert "exhausted" in (third.content or "")
