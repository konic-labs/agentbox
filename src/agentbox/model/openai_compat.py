"""OpenAI-compatible AsyncOpenAI client wrapper (vLLM, SGLang, OpenRouter, ...)."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import uuid
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from agentbox.config import ModelConfig
from agentbox.errors import ContextLengthError, ModelError
from agentbox.model.base import ModelResponse
from agentbox.trajectory.schema import FunctionCall, ToolCall

logger = logging.getLogger("agentbox.model")

_CONTEXT_PATTERNS = (
    "maximum context length",
    "context length",
    "context_length_exceeded",
    "context window",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "longer than the model",
    "exceeds the model",
    "max_model_len",
)
_ROOM_PATTERNS = (
    # OpenAI: "... is 8192 tokens. However, you requested 9000 tokens (5000 in the messages, ...)"
    re.compile(r"maximum context length is (\d+) tokens.*?\((\d+) in the messages", re.I | re.S),
    # vLLM: "... is 8192 tokens and your request has 5000 input tokens (4096 > 8192 - 5000)"
    re.compile(
        r"maximum context length is (\d+) tokens.*?(?:has|contains)\s+(\d+)\s+(?:input|prompt)",
        re.I | re.S,
    ),
)
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.S)
_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def is_context_length_error(message: str) -> bool:
    lower = message.lower()
    return any(p in lower for p in _CONTEXT_PATTERNS)


def _completion_room(message: str) -> int | None:
    """Tokens left for the completion, parsed from a vLLM/OpenAI 400 message."""
    for pattern in _ROOM_PATTERNS:
        m = pattern.search(message)
        if m:
            return int(m.group(1)) - int(m.group(2))
    return None


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _extra(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None) or {}
        value = extra.get(name)
    return value


class OpenAICompatClient:
    """Thin, retrying wrapper over ``openai.AsyncOpenAI`` for any compatible endpoint."""

    def __init__(self, config: ModelConfig, *, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        self.model = config.model
        self._owns_client = client is None
        if client is None:
            api_key = (
                config.api_key
                or (os.getenv(config.api_key_env) if config.api_key_env else None)
                or os.getenv("OPENAI_API_KEY")
                or "EMPTY"
            )
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": config.timeout_s,
                "max_retries": 0,  # retries are handled here (typed, with jitter)
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            if config.extra_headers:
                kwargs["default_headers"] = config.extra_headers
            client = AsyncOpenAI(**kwargs)
        self._client = client

    @classmethod
    def from_async_openai(
        cls,
        client: AsyncOpenAI,
        *,
        model: str,
        **defaults: Any,
    ) -> OpenAICompatClient:
        inst = cls(ModelConfig(model=model, **defaults), client=client)
        inst._owns_client = False
        return inst

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- request
    def build_params(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.config
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if cfg.temperature is not None:
            params["temperature"] = cfg.temperature
        if cfg.max_tokens is not None:
            params[cfg.max_tokens_param] = cfg.max_tokens
        for name in ("top_p", "seed", "stop", "presence_penalty", "frequency_penalty"):
            value = getattr(cfg, name)
            if value is not None:
                params[name] = value
        if tools:
            params["tools"] = tools
            choice = tool_choice if tool_choice is not None else cfg.tool_choice
            if choice is not None:
                params["tool_choice"] = choice
            if cfg.parallel_tool_calls is not None:
                params["parallel_tool_calls"] = cfg.parallel_tool_calls
        if cfg.capture_logprobs:
            params["logprobs"] = True
            if cfg.top_logprobs is not None:
                params["top_logprobs"] = cfg.top_logprobs
        extra_body: dict[str, Any] = dict(cfg.extra_body)
        for name in ("top_k", "min_p", "repetition_penalty"):
            value = getattr(cfg, name)
            if value is not None:
                extra_body.setdefault(name, value)
        if cfg.capture_tokens:
            extra_body.setdefault("return_token_ids", True)
        if extra_body:
            params["extra_body"] = extra_body
        params.update(overrides)
        return params

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        cfg = self.config
        params = self.build_params(messages, tools=tools, tool_choice=tool_choice, overrides=kwargs)
        attempt = 0
        shrunk = False
        while True:
            try:
                response = await self._client.chat.completions.create(**params)
                break
            except BadRequestError as exc:
                msg = str(exc)
                if is_context_length_error(msg):
                    room = _completion_room(msg)
                    key = cfg.max_tokens_param
                    if (
                        cfg.adaptive_max_tokens
                        and not shrunk
                        and room is not None
                        and room >= cfg.min_completion_tokens
                        and params.get(key, 0) > room
                    ):
                        params[key] = room
                        shrunk = True
                        continue
                    raise ContextLengthError(msg) from exc
                if tools and ("tool" in msg.lower() or "function" in msg.lower()):
                    raise ModelError(
                        "Model/server does not appear to support tool calling. "
                        f"Details: {msg}"
                    ) from exc
                raise ModelError(f"Chat completion failed: {msg}") from exc
            except (AuthenticationError, PermissionDeniedError, NotFoundError) as exc:
                raise ModelError(f"Chat completion rejected: {exc}") from exc
            except _RETRYABLE as exc:
                if attempt >= cfg.max_retries:
                    raise ModelError(
                        f"Chat completion failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                delay = _retry_after(exc)
                if delay is None:
                    cap = min(cfg.retry_max_backoff_s, cfg.retry_backoff_s * (2**attempt))
                    delay = random.uniform(0, cap)  # full jitter
                attempt += 1
                logger.debug("retrying chat completion in %.2fs (%s)", delay, type(exc).__name__)
                await asyncio.sleep(delay)
            except APIStatusError as exc:
                raise ModelError(f"Chat completion API error: {exc}") from exc
        return self.parse_response(response)

    # --------------------------------------------------------------- response
    def parse_response(self, response: Any) -> ModelResponse:
        if not getattr(response, "choices", None):
            raise ModelError("Chat completion returned no choices")
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=tc.id or f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=FunctionCall(name=fn.name or "", arguments=fn.arguments or "{}"),
                )
            )

        content = message.content
        reasoning = _extra(message, "reasoning_content") or _extra(message, "reasoning")
        if not isinstance(reasoning, str):
            reasoning = None
        if self.config.strip_think and content and "<think>" in content:
            thoughts = _THINK_RE.findall(content)
            content = _THINK_RE.sub("", content).strip() or None
            reasoning = "\n".join(t.strip() for t in [reasoning, *thoughts] if t) or None

        usage = None
        if response.usage:
            u = response.usage
            usage = {
                "prompt_tokens": u.prompt_tokens or 0,
                "completion_tokens": u.completion_tokens or 0,
                "total_tokens": u.total_tokens or 0,
            }
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details else None
            if cached:
                usage["cached_tokens"] = int(cached)
            cdetails = getattr(u, "completion_tokens_details", None)
            rtoks = getattr(cdetails, "reasoning_tokens", None) if cdetails else None
            if rtoks:
                usage["reasoning_tokens"] = int(rtoks)

        completion_ids: list[int] | None = None
        logprobs: list[float] | None = None
        prompt_ids: list[int] | None = None
        if self.config.capture_tokens or self.config.capture_logprobs:
            raw_ids = _extra(choice, "token_ids")
            if isinstance(raw_ids, list):
                completion_ids = [int(t) for t in raw_ids]
            raw_prompt = _extra(response, "prompt_token_ids")
            if isinstance(raw_prompt, list):
                prompt_ids = [int(t) for t in raw_prompt]
            lp = getattr(choice, "logprobs", None)
            entries = getattr(lp, "content", None) if lp is not None else None
            if entries:
                logprobs = [float(e.logprob) for e in entries]
                if completion_ids is None:
                    # vLLM return_tokens_as_token_ids=True encodes ids in the token text
                    parsed = [_TOKEN_ID_RE.match(e.token or "") for e in entries]
                    if all(parsed):
                        completion_ids = [int(m.group(1)) for m in parsed if m]

        raw_choice = None
        if self.config.record_raw:
            try:
                raw_choice = choice.model_dump(mode="json")
            except Exception:
                raw_choice = None

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            reasoning_content=reasoning,
            usage=usage,
            raw_choice=raw_choice,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            completion_logprobs=logprobs,
        )
