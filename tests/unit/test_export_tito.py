"""Exporters, TITO tokenization and the verl adapter (no Docker, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentbox.export import ExportFilter, export
from agentbox.integrations.art import to_art_dict
from agentbox.integrations.verl import build_agent_loop_class, tasks_to_verl_rows
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.tito import ChatTokenizer, GenOutput, TitoModelClient
from agentbox.trajectory.schema import (
    FunctionCall,
    Message,
    ToolCall,
    Trajectory,
    TrajectoryMetrics,
)
from agentbox.trajectory.tokens import TokenTracker
from agentbox.types import FailureKind, FinalStatus, VerifierType


def _traj(
    task: str, sample: int, reward: float, *, kind: FailureKind = FailureKind.NONE
) -> Trajectory:
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content=f"do {task}"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="c1", function=FunctionCall(name="list_files", arguments="{}"))
            ],
        ),
        Message(role="tool", content="a.py", tool_call_id="c1", name="list_files"),
        Message(role="assistant", content=f"done {reward}"),
    ]
    tracker = TokenTracker()
    msgs[2].tokens = tracker.observe([1, 2, 3], [10, 11], [-0.5, -0.5])
    msgs[4].tokens = tracker.observe([1, 2, 3, 10, 11, 20], [12], [-0.1])
    return Trajectory(
        task_id=task,
        run_id="r",
        rollout_id=f"{task}-{sample}",
        group_id=f"{task}@r",
        sample_index=sample,
        messages=msgs,
        reward=reward,
        final_status=FinalStatus.SUCCESS if reward >= 1 else FinalStatus.FAILED,
        failure_kind=kind if reward < 1 else FailureKind.NONE,
        metrics=TrajectoryMetrics(steps=1),
        tools=[{"type": "function", "function": {"name": "list_files"}}],
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_chat_and_filters(tmp_path: Path) -> None:
    trajs = [_traj("a", 0, 1.0), _traj("a", 1, 0.0), _traj("b", 0, 0.0, kind=FailureKind.SANDBOX)]
    stats = export(trajs, "chat", tmp_path / "chat.jsonl")
    assert (stats.read, stats.kept, stats.rows) == (3, 2, 2)  # infra failure dropped by default
    row = _rows(tmp_path / "chat.jsonl")[0]
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "list_files"
    assert row["tools"] and row["metadata"]["reward"] == 1.0
    ok = export(trajs, "chat", tmp_path / "ok.jsonl.gz", filter=ExportFilter(successful_only=True))
    assert ok.rows == 1
    with pytest.raises(KeyError):
        export(trajs, "nope", tmp_path / "x")


def test_dpo_grpo_tokens_sharegpt_art(tmp_path: Path) -> None:
    trajs = [_traj("a", 0, 1.0), _traj("a", 1, 0.0), _traj("c", 0, 0.0), _traj("c", 1, 0.0)]
    dpo = export(trajs, "dpo", tmp_path / "dpo.jsonl")
    assert dpo.rows == 1 and dpo.skipped_groups == 1
    pair = _rows(tmp_path / "dpo.jsonl")[0]
    assert [m["role"] for m in pair["prompt"]] == ["system", "user"]
    assert pair["chosen"][-1]["content"] == "done 1.0"
    grpo = export(trajs, "grpo", tmp_path / "grpo.jsonl", drop_zero_variance=True)
    assert grpo.rows == 1 and grpo.skipped_groups == 1
    g = _rows(tmp_path / "grpo.jsonl")[0]
    assert g["rewards"] == [1.0, 0.0] and g["advantages"][0] > 0 > g["advantages"][1]
    export(trajs[:1], "tokens", tmp_path / "tok.jsonl")
    seq = _rows(tmp_path / "tok.jsonl")[0]
    assert seq["prompt_ids"] == [1, 2, 3]
    assert seq["response_ids"] == [10, 11, 20, 12] and seq["response_mask"] == [1, 1, 0, 1]
    export(trajs[:1], "tokens", tmp_path / "turns.jsonl", per_turn=True)
    turns = _rows(tmp_path / "turns.jsonl")
    assert turns[1]["prompt_ids"] == [1, 2, 3, 10, 11, 20]
    export(trajs[:1], "sharegpt", tmp_path / "sg.jsonl")
    assert _rows(tmp_path / "sg.jsonl")[0]["conversations"][1]["from"] == "human"
    art = to_art_dict(trajs[0])
    assert art["metrics"]["correct"] == 1.0
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in art["metadata"].values())


class QwenLikeTokenizer:
    """Char-level tokenizer with a Qwen-style template (special tokens are single ids)."""

    SPECIAL = ["<|im_start|>", "<|im_end|>"]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **kw: Any,
    ) -> str:
        out = []
        if tools:
            out.append("<|im_start|>system\nTOOLS<|im_end|>\n")
        for m in messages:
            role = "user" if m["role"] == "tool" else m["role"]
            body = m.get("content") or ""
            if m["role"] == "tool":
                body = f"<tool_response>{body}</tool_response>"
            out.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            for j, s in enumerate(self.SPECIAL):
                if text.startswith(s, i):
                    ids.append(1000 + j)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.SPECIAL[t - 1000] if t >= 1000 else chr(t) for t in ids)


async def test_tito_client_keeps_one_prefix_consistent_sequence() -> None:
    tok = QwenLikeTokenizer()
    chat = ChatTokenizer(tok)
    replies = iter(
        [
            '<tool_call>{"name": "list_files", "arguments": {}}</tool_call><|im_end|>',
            "all done<|im_end|>",
        ]
    )

    async def generate(prompt_ids: list[int], sampling: dict[str, Any]) -> GenOutput:
        text = next(replies)
        return GenOutput(
            token_ids=tok.encode(text),
            text=text.replace("<|im_end|>", ""),
            logprobs=[-0.1] * len(tok.encode(text)),
        )

    client = TitoModelClient(chat, generate)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    r1 = await client.complete(msgs, tools=[{"type": "function"}])
    assert r1.tool_calls and r1.tool_calls[0].function.name == "list_files"
    msgs2 = [
        *msgs,
        {"role": "assistant", "content": r1.content},
        {"role": "tool", "content": "a.py", "tool_call_id": r1.tool_calls[0].id},
    ]
    r2 = await client.complete(msgs2, tools=[{"type": "function"}])
    tracker = TokenTracker()
    t1 = tracker.observe(r1.prompt_token_ids, r1.completion_token_ids, r1.completion_logprobs)
    t2 = tracker.observe(r2.prompt_token_ids, r2.completion_token_ids, r2.completion_logprobs)
    assert t2 is not None and not t2.prefix_break  # no re-rendering => one sequence
    delta = tok.decode(t2.prompt_ids or [])
    assert (
        delta
        == "\n<|im_start|>user\n<tool_response>a.py</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    assert t1 is not None


@pytest.mark.parametrize("where", ["evaluate", "reset"])
async def test_verl_infra_failures_are_never_trained(
    monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    import agentbox.integrations.verl as verl_mod
    from agentbox.runner.provision import ProvisionError

    class FakeEnv:
        def __init__(self, **kw: Any) -> None:
            pass

        async def reset(self, task: Task) -> None:
            if where == "reset":
                raise ProvisionError(FailureKind.SANDBOX, "docker went away")

        async def call_tool(self, name: str, arguments: str) -> Any:
            return SimpleNamespace(message=SimpleNamespace(content="ok"))

        async def evaluate(self) -> Any:
            return SimpleNamespace(
                reward=0.0,
                success=False,
                final_status=FinalStatus.ERROR,
                failure_kind=FailureKind.VERIFIER,
                breakdown={},
            )

        async def close(self) -> None:
            pass

    class FakeBase:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            text, _, _ = await self._call_tool(
                SimpleNamespace(name="read_file", arguments="{}"), {}, None
            )
            return SimpleNamespace(
                reward_score=None, response_mask=[1, 1, 0, 1], extra_fields={"tool": text}
            )

    monkeypatch.setattr(verl_mod, "AgentBoxEnv", FakeEnv)
    loop = build_agent_loop_class(FakeBase, tool_response=lambda text: text)()
    task = Task(task_id="t1", description="d", verifier=VerifierSpec(type=VerifierType.PYTEST))
    out = await loop.run({}, extra_info=tasks_to_verl_rows([task])[0]["extra_info"])
    assert out.response_mask == [0, 0, 0, 0]  # no gradient from an unscorable sample
    assert out.reward_score == 0.0 and out.extra_fields["agentbox"]["infra_failure"] is True
    if where == "reset":  # the batch survives; tools report the missing environment
        assert out.extra_fields["agentbox"]["failure_kind"] == "sandbox"
        assert "no AgentBox environment" in out.extra_fields["tool"]


async def test_verl_agent_loop_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentbox.integrations.verl as verl_mod

    calls: list[Any] = []

    class FakeEnv:
        def __init__(self, **kw: Any) -> None:
            self.closed = False

        async def reset(self, task: Task) -> None:
            calls.append(("reset", task.task_id))

        async def call_tool(self, name: str, arguments: str) -> Any:
            calls.append(("tool", name, arguments))
            return SimpleNamespace(message=SimpleNamespace(content=f"ran {name}"))

        async def evaluate(self) -> Any:
            return SimpleNamespace(
                reward=0.75,
                success=False,
                final_status=FinalStatus.FAILED,
                failure_kind=FailureKind.POLICY,
                breakdown={"verifier": 0.75},
            )

        async def close(self) -> None:
            calls.append(("close",))

    class FakeBase:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            resp, _, _ = await self._call_tool(
                SimpleNamespace(name="read_file", arguments='{"path": "x"}'), {}, None
            )
            return SimpleNamespace(reward_score=None, extra_fields={"tool_text": resp})

    monkeypatch.setattr(verl_mod, "AgentBoxEnv", FakeEnv)
    cls = build_agent_loop_class(FakeBase, tool_response=lambda text: text)
    # verl registers f"{__module__}.{__qualname__}"; hydra must be able to import it
    assert f"{cls.__module__}.{cls.__qualname__}" == "agentbox.integrations.verl.AgentBoxAgentLoop"
    loop = cls()
    task = Task(task_id="t1", description="d", verifier=VerifierSpec(type=VerifierType.PYTEST))
    rows = tasks_to_verl_rows([task])
    assert rows[0]["agent_name"] == "agentbox" and rows[0]["prompt"][0]["role"] == "system"
    out = await loop.run({}, extra_info=rows[0]["extra_info"])
    assert out.reward_score == 0.75
    assert out.extra_fields["tool_text"] == "ran read_file"
    assert out.extra_fields["agentbox"]["reward_breakdown"] == {"verifier": 0.75}
    assert calls[0] == ("reset", "t1") and calls[-1] == ("close",)
    assert "read_file" in [s["function"]["name"] for s in loop.tool_schemas]
    # verl's loop reads `tool.tool_schema` from every self.tools value (parsers, selection)
    schemas = [t.tool_schema.model_dump() for t in loop.tools.values()]
    assert [x["function"]["name"] for x in schemas] == [t.name for t in loop.tools.values()]
