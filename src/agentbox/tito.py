"""Token-in / token-out (TITO) generation for exact single-sequence RL data (GH #95).

Chat-completions servers re-render the whole history every turn, so the tokens
a model generated are generally *not* a prefix of the next prompt (tool-call
JSON is re-serialised, ``<think>`` blocks are dropped). TITO keeps the token
sequence itself:

* the initial prompt (system + user + tools) is rendered once with the HF chat
  template;
* each turn is generated from token ids (``/v1/completions`` with
  ``prompt=[ids]``, or a trainer's engine);
* tool results are appended as *delta tokens* rendered from a fixed dummy
  prefix, never re-rendering earlier turns.

:class:`TitoModelClient` exposes this through the normal ``ModelClient``
interface, so ``Rollout``/``ParallelRunner`` produce prefix-consistent
``TokenTrace`` s with one sequence per trajectory.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agentbox.agent.protocols import parse_hermes_tool_calls
from agentbox.model.base import ModelResponse

# Set per rollout so one client can serve many concurrent conversations.
SESSION_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentbox_tito_session", default=None
)

_MARKER = "AGENTBOX_ASSISTANT_TURN_MARKER"
_DUMMY = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "U"},
    {"role": "assistant", "content": _MARKER},
]


@dataclass
class GenOutput:
    token_ids: list[int]
    text: str
    logprobs: list[float] | None = None
    finish_reason: str | None = None


Generate = Callable[[list[int], dict[str, Any]], Awaitable[GenOutput]]


class ChatTokenizer:
    """Render chat turns to token ids without re-rendering history."""

    def __init__(self, tokenizer: Any, *, chat_template_kwargs: dict[str, Any] | None = None) -> None:
        self.tok = tokenizer
        self.kw = dict(chat_template_kwargs or {})

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: Any) -> ChatTokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("TITO needs transformers: pip install agentbox-rl[hf]") from exc
        return cls(AutoTokenizer.from_pretrained(name), **kwargs)

    def _render(
        self,
        messages: list[dict[str, Any]],
        *,
        generation: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        text = self.tok.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=generation,
            tokenize=False,
            **self.kw,
        )
        return str(text)

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def initial(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[int]:
        return self.encode(self._render(messages, generation=True, tools=tools))

    def tool_delta(self, messages: list[dict[str, Any]], completion_ids: list[int]) -> list[int]:
        """Tokens that follow a generated assistant turn: its close + ``messages`` + gen prompt.

        Rendered after a marker assistant turn that is *not* the last message, so
        templates that special-case the final assistant turn (Qwen3 ``<think>``)
        render it exactly as they would mid-conversation.
        """
        full = self._render([*_DUMMY, *messages], generation=True)
        idx = full.rfind(_MARKER)
        if idx < 0:
            raise ValueError("chat template dropped the assistant content")
        text = full[idx + len(_MARKER):]
        if completion_ids:
            last = self.tok.decode([completion_ids[-1]], skip_special_tokens=False)
            if last and text.startswith(last):
                text = text[len(last):]  # the model already emitted the stop token
        return self.encode(text)


class CompletionsGenerator:
    """Generate from token ids via an OpenAI-compatible ``/v1/completions`` (vLLM)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout_s: float = 300.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.extra_body = dict(extra_body or {})

    async def __call__(self, prompt_ids: list[int], sampling: dict[str, Any]) -> GenOutput:
        body = {"return_token_ids": True, "skip_special_tokens": False, **self.extra_body}
        resp = await self._client.completions.create(
            model=self.model,
            prompt=prompt_ids,
            max_tokens=sampling.get("max_tokens", 1024),
            temperature=sampling.get("temperature", 1.0),
            top_p=sampling.get("top_p", 1.0),
            logprobs=1,
            seed=sampling.get("seed"),
            stop=sampling.get("stop"),
            extra_body=body,
        )
        choice = resp.choices[0]
        ids = getattr(choice, "token_ids", None) or (choice.model_extra or {}).get("token_ids")
        lp = getattr(choice, "logprobs", None)
        token_lps = list(lp.token_logprobs) if lp is not None and lp.token_logprobs else None
        if ids is None:
            raise RuntimeError("server did not return token_ids (vLLM >= 0.10 required)")
        return GenOutput(
            token_ids=[int(t) for t in ids],
            text=choice.text or "",
            logprobs=[float(x) if x is not None else 0.0 for x in token_lps] if token_lps else None,
            finish_reason=choice.finish_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


@dataclass
class _Session:
    ids: list[int] = field(default_factory=list)
    n_messages: int = 0
    last_completion: list[int] = field(default_factory=list)


class TitoModelClient:
    """``ModelClient`` that keeps one exact token sequence per rollout."""

    def __init__(
        self,
        chat: ChatTokenizer,
        generate: Generate,
        *,
        model: str = "tito",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_context_tokens: int | None = None,
    ) -> None:
        self.chat = chat
        self.generate = generate
        self.model = model
        self.sampling = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
        self.max_context_tokens = max_context_tokens
        self._sessions: dict[str, _Session] = {}

    def _key(self, messages: list[dict[str, Any]]) -> str:
        explicit = SESSION_KEY.get()
        if explicit:
            return explicit
        head = json.dumps(messages[:2], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(head.encode()).hexdigest()

    def end_session(self, key: str) -> None:
        self._sessions.pop(key, None)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ModelResponse:
        from agentbox.errors import ContextLengthError

        key = self._key(messages)
        session = self._sessions.get(key)
        if session is None or len(messages) < session.n_messages:
            session = self._sessions[key] = _Session(ids=self.chat.initial(messages, tools))
        else:
            new = [m for m in messages[session.n_messages:] if m.get("role") != "assistant"]
            if new:
                session.ids.extend(self.chat.tool_delta(new, session.last_completion))
        prompt = list(session.ids)
        sampling = {**self.sampling, **{k: v for k, v in kwargs.items() if v is not None}}
        if self.max_context_tokens is not None:
            room = self.max_context_tokens - len(prompt)
            if room < 16:
                raise ContextLengthError(f"TITO context full ({len(prompt)} tokens)")
            sampling["max_tokens"] = min(int(sampling["max_tokens"]), room)
        out = await self.generate(prompt, sampling)
        session.ids.extend(out.token_ids)
        session.last_completion = out.token_ids
        session.n_messages = len(messages) + 1  # + the assistant turn appended by the loop
        content, calls = parse_hermes_tool_calls(out.text)
        return ModelResponse(
            content=out.text if calls else (content or out.text),
            tool_calls=calls,
            finish_reason=out.finish_reason,
            prompt_token_ids=prompt,
            completion_token_ids=out.token_ids,
            completion_logprobs=out.logprobs,
            usage={
                "prompt_tokens": len(prompt),
                "completion_tokens": len(out.token_ids),
                "total_tokens": len(prompt) + len(out.token_ids),
            },
        )

    async def aclose(self) -> None:
        close = getattr(self.generate, "aclose", None)
        if close is not None:
            await close()
