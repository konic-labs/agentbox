"""Agent loop tests with MockModelClient (no Docker)."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

from agentbox.agent.loop import AgentLoop
from agentbox.config import AgentConfig
from agentbox.errors import ContextLengthError, ModelError
from agentbox.model.base import ModelResponse
from agentbox.model.mock import FunctionModelClient, MockModelClient
from agentbox.tools.base import BaseTool, ToolContext
from agentbox.tools.decorator import tool
from agentbox.tools.executor import ToolExecutor
from agentbox.tools.registry import build_tool_registry
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.types import FailureKind, FinalStatus, ToolMode


class FakeSandbox:
    workspace_dir = "/workspace"

    def __init__(self, exit_code: int = 0, stdout: str = "ok") -> None:
        self.files: dict[str, str] = {"a.txt": "hello"}
        self.exit_code = exit_code
        self.stdout = stdout
        self.exec_calls: list[Any] = []

    async def exec(self, command: Any, timeout_s: float | None = 60.0) -> Any:
        self.exec_calls.append((command, timeout_s))
        return SimpleNamespace(
            exit_code=self.exit_code, stdout=self.stdout, stderr="", duration_s=0.0, timed_out=False
        )

    async def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_text(self, path: str, content: str) -> None:
        await asyncio.sleep(0.01)  # a write that takes time must still precede later reads
        self.files[path] = content

    async def write_files(self, files: dict[str, str]) -> None:
        self.files.update(files)

    async def list_dir(self, path: str = ".", recursive: bool = False) -> list[str]:
        return sorted(self.files)


def call(cid: str, name: str, **args: Any) -> ToolCall:
    return ToolCall(id=cid, function=FunctionCall(name=name, arguments=json.dumps(args)))


def turn(*calls: ToolCall, content: str | None = None) -> ModelResponse:
    return ModelResponse(content=content, tool_calls=list(calls))


def done(text: str = "done") -> ModelResponse:
    return ModelResponse(content=text, tool_calls=[], finish_reason="stop")


async def run_loop(
    script: list[ModelResponse] | Any,
    *,
    sandbox: FakeSandbox | None = None,
    mode: ToolMode | list[str] = ToolMode.STRUCTURED,
    custom: list[BaseTool] | None = None,
    deadline: float | None = None,
    **agent_kw: Any,
) -> Any:
    sandbox = sandbox or FakeSandbox()
    registry = build_tool_registry(mode, custom_tools=custom)
    cfg = AgentConfig(max_steps=agent_kw.pop("max_steps", 5), **agent_kw)
    executor = ToolExecutor(
        registry,
        max_output_chars=cfg.max_tool_output_chars,
        tool_timeout_s=cfg.tool_timeout_s,
        parallel_read_only=cfg.parallel_tool_calls,
    )
    model = script if not isinstance(script, list) else MockModelClient(script)
    loop = AgentLoop(model, executor, cfg)
    ctx = ToolContext(sandbox=sandbox, workspace_dir="/workspace", run_id="r", task_id="t")
    return await loop.run(system_prompt="sys", user_prompt="do", ctx=ctx, episode_deadline=deadline)


def roles(result: Any) -> list[str]:
    return [m.role if isinstance(m.role, str) else m.role.value for m in result.messages]


async def test_parallel_read_only_calls_keep_order() -> None:
    result = await run_loop(
        [turn(call("c1", "read_file", path="a.txt"), call("c2", "list_files")), done()]
    )
    assert result.stop_reason == "final_answer"
    assert result.steps == 1
    tool_ids = [m.tool_call_id for m in result.messages if m.tool_call_id]
    assert tool_ids == ["c1", "c2"]


async def test_write_then_read_in_one_turn_is_sequential() -> None:
    result = await run_loop(
        [
            turn(
                call("w", "write_file", path="b.txt", content="NEW"),
                call("r", "read_file", path="b.txt"),
            ),
            done(),
        ]
    )
    tool_msgs = [m for m in result.messages if m.tool_call_id == "r"]
    assert tool_msgs[0].content == "NEW"


async def test_max_steps() -> None:
    script = [turn(call(f"c{i}", "list_files")) for i in range(5)]
    result = await run_loop(script, max_steps=2)
    assert result.final_status == FinalStatus.MAX_STEPS
    assert result.steps == 2
    assert result.failure_kind == FailureKind.POLICY


async def test_tool_errors_are_recorded() -> None:
    sandbox = FakeSandbox(exit_code=3, stdout="boom")
    result = await run_loop(
        [
            turn(
                call("a", "run_command", command="false"),
                call("b", "nope"),
                ToolCall(id="c", function=FunctionCall(name="read_file", arguments="{bad json")),
                call("d", "read_file"),
            ),
            done(),
        ],
        sandbox=sandbox,
    )
    recs = {r.tool_call_id: r for r in result.tool_call_records}
    assert recs["a"].is_error and "exit_code=3" in recs["a"].result
    assert recs["b"].is_error and "unknown tool" in recs["b"].result
    assert "invalid arguments" in recs["c"].result
    assert "missing required argument" in recs["d"].result
    assert all(r.duration_s >= 0 for r in recs.values())


async def test_tool_timeout_produces_tool_message_not_dangling_call() -> None:
    @tool(description="sleeps")
    async def slow(sandbox: Any) -> str:
        await asyncio.sleep(30)
        return "never"

    deadline = time.monotonic() + 1.0
    result = await run_loop(
        [turn(call("s", "slow")), done()],
        mode=ToolMode.CUSTOM,
        custom=[slow],
        deadline=deadline,
        step_timeout_s=10,
    )
    answered = {m.tool_call_id for m in result.messages if m.tool_call_id}
    assert "s" in answered
    assert result.final_status == FinalStatus.TIMEOUT


async def test_context_length_is_policy_failure() -> None:
    async def policy(messages: Any, tools: Any) -> ModelResponse:
        raise ContextLengthError("maximum context length exceeded")

    result = await run_loop(FunctionModelClient(policy))
    assert result.stop_reason == "context_length"
    assert result.final_status == FinalStatus.FAILED
    assert result.failure_kind == FailureKind.POLICY


async def test_model_api_error_is_infra() -> None:
    async def policy(messages: Any, tools: Any) -> ModelResponse:
        raise ModelError("502 bad gateway")

    result = await run_loop(FunctionModelClient(policy))
    assert result.final_status == FinalStatus.ERROR
    assert result.failure_kind == FailureKind.MODEL_API


async def test_submit_ends_episode_and_skips_later_calls() -> None:
    result = await run_loop(
        [
            turn(call("s", "submit", answer="42"), call("x", "list_files")),
            turn(call("never", "list_files")),
        ],
        mode=ToolMode.EXTENDED,
    )
    assert result.stop_reason == "submitted"
    assert result.submitted_answer == "42"
    skipped = [m for m in result.messages if m.tool_call_id == "x"]
    assert skipped and "Skipped" in (skipped[0].content or "")
    assert result.model_calls == 1


async def test_hermes_protocol_parses_text_tool_calls() -> None:
    seen: list[Any] = []

    async def policy(messages: list[dict[str, Any]], tools: Any) -> ModelResponse:
        seen.append((messages, tools))
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if n_assistant == 0:
            return ModelResponse(
                content='Let me look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
            )
        return done()

    result = await run_loop(FunctionModelClient(policy), tool_protocol="hermes")
    first_messages, first_tools = seen[0]
    assert first_tools is None  # tools are rendered into the prompt instead
    assert "<tools>" in first_messages[0]["content"]
    tool_msgs = [m for m in result.messages if m.tool_call_id]
    assert tool_msgs and tool_msgs[0].content == "hello"
    second_messages, _ = seen[1]
    assert "tool_calls" not in second_messages[2]  # assistant replayed as raw text


async def test_tool_output_truncated_head_and_tail() -> None:
    sandbox = FakeSandbox(stdout="H" * 50 + "x" * 50_000 + "TAIL")
    result = await run_loop(
        [turn(call("a", "run_command", command="cat big")), done()],
        sandbox=sandbox,
        max_tool_output_chars=1000,
    )
    msg = next(m for m in result.messages if m.tool_call_id == "a")
    assert len(msg.content or "") < 1200
    assert (msg.content or "").rstrip().endswith("TAIL")
    rec = result.tool_call_records[0]
    assert rec.truncated


async def test_reasoning_not_echoed_by_default() -> None:
    captured: list[Any] = []

    async def policy(messages: list[dict[str, Any]], tools: Any) -> ModelResponse:
        captured.append(messages)
        if len(captured) == 1:
            r = turn(call("a", "list_files"))
            r.reasoning_content = "thinking..."
            return r
        return done()

    result = await run_loop(FunctionModelClient(policy))
    assistant_wire = captured[1][2]
    assert "reasoning_content" not in assistant_wire
    assert result.messages[2].reasoning_content == "thinking..."


async def test_run_command_timeout_is_clamped() -> None:
    sandbox = FakeSandbox()
    await run_loop(
        [turn(call("a", "run_command", command="sleep 1", timeout_s=10_000)), done()],
        sandbox=sandbox,
        tool_timeout_s=30,
    )
    assert sandbox.exec_calls[0][1] == 30


async def test_shared_mock_is_deterministic_across_concurrent_rollouts() -> None:
    mock = MockModelClient([turn(call("w", "write_file", path="f", content="x")), done()])
    results = await asyncio.gather(*(run_loop(mock) for _ in range(4)))
    assert all(r.stop_reason == "final_answer" and r.steps == 1 for r in results)


async def test_native_protocol_salvages_unparsed_tool_call_text() -> None:
    seen: list[Any] = []

    async def policy(messages: list[dict[str, Any]], tools: Any) -> ModelResponse:
        seen.append(messages)
        if len(seen) == 1:  # the server's tool parser gave up: the call is plain text
            return ModelResponse(
                content='Reading it.\n<tool_call>\n{"name": "read_file", "arguments": '
                '{"path": "a.txt"}}\n</tool_call>'
            )
        return done()

    result = await run_loop(FunctionModelClient(policy))
    assert result.salvaged_tool_calls == 1 and result.stop_reason == "final_answer"
    assert [m.content for m in result.messages if m.tool_call_id] == ["hello"]
    replayed = seen[1][2]  # the assistant turn as sent back to the server
    assert replayed["tool_calls"][0]["function"]["name"] == "read_file"
    assert "<tool_call>" not in (replayed.get("content") or "")


async def test_salvage_can_be_disabled_and_reports_malformed_calls() -> None:
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
    off = await run_loop([ModelResponse(content=text)], salvage_tool_calls=False)
    assert off.stop_reason == "final_answer" and off.salvaged_tool_calls == 0
    assert not [m for m in off.messages if m.tool_call_id]

    broken = await run_loop([ModelResponse(content="<tool_call>\n{oops\n</tool_call>"), done()])
    assert broken.salvaged_tool_calls == 1
    rec = broken.tool_call_records[0]
    assert rec.is_error and rec.name == "invalid_tool_call"
    assert "malformed tool call" in rec.result
    # the replayed call must stay valid JSON (native servers re-parse arguments)
    call = next(m for m in broken.messages if m.tool_calls).tool_calls[0]
    assert json.loads(call.function.arguments)["raw"].startswith("{oops")


async def test_before_model_hook_injects_messages() -> None:
    from agentbox.agent.loop import AgentHooks
    from agentbox.trajectory.schema import Message
    from agentbox.types import MessageRole

    seen: list[list[dict[str, Any]]] = []

    class Nudge(AgentHooks):
        async def before_model(self, step: int, messages: list[Message]) -> Message | None:
            messages.clear()  # a copy: mutating it must not break the episode
            if step == 1:  # 0-based: the second model call, after the first tool result
                return Message(role=MessageRole.USER, content="Remember to run the tests.")
            return None

    async def policy(messages: list[dict[str, Any]], tools: Any) -> ModelResponse:
        seen.append(list(messages))  # the client gets the live payload: snapshot it
        return turn(call("a", "list_files")) if len(seen) == 1 else done()

    registry = build_tool_registry(ToolMode.STRUCTURED)
    loop = AgentLoop(
        FunctionModelClient(policy), ToolExecutor(registry), AgentConfig(), hooks=Nudge()
    )
    ctx = ToolContext(sandbox=FakeSandbox(), workspace_dir="/workspace", run_id="r", task_id="t")
    result = await loop.run(system_prompt="sys", user_prompt="do", ctx=ctx)
    assert seen[1][-1] == {"role": "user", "content": "Remember to run the tests."}
    assert [m["role"] for m in seen[0]] == ["system", "user"]  # nothing injected at step 0
    assert result.messages[-2].content == "Remember to run the tests."
    assert result.stop_reason == "final_answer"
