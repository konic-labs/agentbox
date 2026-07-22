"""OpenAI-compatible AsyncOpenAI client wrapper."""

from __future__ import annotations

import os
from typing import Any

from openai import APIError, AsyncOpenAI, BadRequestError

from agentbox.config import ModelConfig
from agentbox.errors import ModelError
from agentbox.model.base import ModelResponse
from agentbox.trajectory.schema import FunctionCall, ToolCall


class OpenAICompatClient:
    """Thin wrapper over openai.AsyncOpenAI for any compatible endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = config.model
        api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout_s,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.extra_headers:
            kwargs["default_headers"] = config.extra_headers
        self._client = AsyncOpenAI(**kwargs)

    @classmethod
    def from_async_openai(
        cls,
        client: AsyncOpenAI,
        *,
        model: str,
        **defaults: Any,
    ) -> OpenAICompatClient:
        config = ModelConfig(model=model, **defaults)
        inst = object.__new__(cls)
        inst.config = config
        inst.model = model
        inst._client = client
        return inst

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        **kwargs: Any,
    ) -> ModelResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        if tools:
            params["tools"] = tools
            params["tool_choice"] = (
                tool_choice
                if tool_choice is not None
                else self.config.tool_choice
            )
        if self.config.extra_body:
            params["extra_body"] = self.config.extra_body
        params.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**params)
        except BadRequestError as exc:
            msg = str(exc)
            if tools and ("tool" in msg.lower() or "function" in msg.lower()):
                raise ModelError(
                    "Model/server does not appear to support tool calling. "
                    f"Details: {msg}"
                ) from exc
            raise ModelError(f"Chat completion failed: {msg}") from exc
        except APIError as exc:
            raise ModelError(f"Chat completion API error: {exc}") from exc
        except Exception as exc:
            raise ModelError(f"Chat completion failed: {exc}") from exc

        if not response.choices:
            raise ModelError("Chat completion returned no choices")

        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id or "",
                        type="function",
                        function=FunctionCall(
                            name=tc.function.name or "",
                            arguments=tc.function.arguments or "{}",
                        ),
                    )
                )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        reasoning = getattr(message, "reasoning_content", None)
        raw_choice = None
        try:
            raw_choice = choice.model_dump(mode="json")
        except Exception:
            raw_choice = None

        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            reasoning_content=reasoning,
            usage=usage,
            raw_choice=raw_choice,
        )
