"""AgentBoxEnv + trainer adapters against real sandboxes (no GPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agentbox.config import AgentConfig, ResourceLimits, SandboxConfig
from agentbox.env import AgentBoxEnv
from agentbox.integrations.skyrl import AgentBoxTextEnvCore
from agentbox.integrations.trl import make_environment_factory, tasks_to_rows
from agentbox.integrations.verl import Settings
from agentbox.runner.provision import ProvisionError
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import FailureKind, VerifierType

pytestmark = pytest.mark.docker

FIX = "def add(a, b):\n    return a + b\n"


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _need_docker() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")


SANDBOX = SandboxConfig(ensure_pytest=True, limits=ResourceLimits(network_disabled=True))
TASK = Task(
    task_id="add",
    description="Fix add() in calc.py.",
    starter_files={
        "calc.py": "def add(a, b):\n    return a - b\n",
        "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    },
    verifier=VerifierSpec(type=VerifierType.PYTEST, command="python -m pytest -q"),
)


async def test_env_reset_step_evaluate_trajectory() -> None:
    async with AgentBoxEnv(sandbox=SANDBOX, agent=AgentConfig(max_steps=5)) as env:
        obs = await env.reset(TASK)
        assert obs.messages[0]["role"] == "system" and obs.tools
        res = await env.step(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": "calc.py", "content": FIX}),
                        },
                    }
                ],
            }
        )
        assert not res.done and res.messages[0]["role"] == "tool"
        res = await env.step('<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>')
        assert "1 passed" in res.messages[0]["content"]
        res = await env.step("The fix is in.")
        assert res.done and res.stop_reason == "final_answer"
        ev = await env.evaluate()
        assert ev.success and ev.reward == 1.0
        traj = env.trajectory()
        assert traj.success and traj.metrics.tool_calls == 2 and len(traj.messages) == 7


async def test_env_provision_failure_is_attributed() -> None:
    bad = TASK.model_copy(update={"task_id": "bad", "setup_commands": ["exit 9"]})
    env = AgentBoxEnv(sandbox=SANDBOX)
    with pytest.raises(ProvisionError) as info:
        await env.reset(bad)
    assert info.value.kind == FailureKind.SEED


def test_trl_environment_factory_sync() -> None:
    Env = make_environment_factory(
        [TASK], sandbox=SANDBOX, tools=["read_file", "write_file", "run_tests"]
    )
    env = Env()
    assert {"read_file", "write_file", "run_tests"} <= set(dir(env))
    assert not hasattr(env, "run_command")
    prompt = env.reset(task_id="add")  # no prompt column: the environment supplies it
    assert "calc.py" in (prompt or "")
    rows = tasks_to_rows([TASK])
    # TRL appends reset()'s return value to the row's prompt: never duplicate it
    assert env.reset(**rows[0]) is None
    assert "return a - b" in env.read_file("calc.py")
    env.write_file("calc.py", FIX)
    assert "passed" in env.run_tests()
    assert env.get_reward() == 1.0 and env.last_trajectory is not None
    env.reset()  # self-sampling path
    assert env.get_reward() == 0.0
    assert rows[0]["task_id"] == "add" and rows[0]["prompt"][-1]["role"] == "user"
    try:
        from transformers.utils import get_json_schema
    except ImportError:
        return
    schema = get_json_schema(env.read_file)
    assert schema["function"]["name"] == "read_file"
    assert "path" in schema["function"]["parameters"]["properties"]


def test_trl_infra_failures_are_unscorable() -> None:
    broken = TASK.model_copy(update={"task_id": "broken", "setup_commands": ["exit 7"]})
    Env = make_environment_factory([broken], sandbox=SANDBOX, tools=["read_file"])
    env = Env()
    assert env.reset(task_id="broken") is not None  # no exception: TRL's batch survives
    assert "environment is unavailable" in env.read_file("calc.py")
    assert env.get_reward() is None  # TRL drops it from the group baseline


def test_skyrl_core_text_episode() -> None:
    core = AgentBoxTextEnvCore(TASK, settings=Settings(sandbox=SANDBOX))
    messages, meta = core.init(None)
    assert "<tools>" in messages[0]["content"] and meta["task_id"] == "add"
    call = json.dumps({"name": "write_file", "arguments": {"path": "calc.py", "content": FIX}})
    out = core.step(f"<tool_call>{call}</tool_call>")
    assert not out["done"] and out["observations"][0]["role"] == "user"
    out = core.step("done")
    assert out["done"] and out["reward"] == 1.0


async def test_openrlhf_episode_with_real_tokenizer() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from agentbox.integrations.openrlhf import run_episode

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as exc:  # no cached tokenizer / no network
        pytest.skip(f"tokenizer unavailable: {exc}")
    call = json.dumps({"name": "write_file", "arguments": {"path": "calc.py", "content": FIX}})
    script = iter([f"<tool_call>\n{call}\n</tool_call>", "Fixed."])

    class Engine:
        async def generate(self, prompt_ids: list[int], sp: Any) -> Any:
            text = next(script)
            ids = tok.encode(text, add_special_tokens=False) + [
                tok.convert_tokens_to_ids("<|im_end|>")
            ]
            lps = [{t: SimpleNamespace(logprob=-0.2)} for t in ids]
            return SimpleNamespace(
                outputs=[
                    SimpleNamespace(token_ids=ids, text=text, logprobs=lps, finish_reason="stop")
                ]
            )

    sp = SimpleNamespace(max_tokens=256, temperature=1.0, logprobs=1)
    out = await run_episode(
        "p", TASK.model_dump_json(), sp, 8192, tok, Engine(), settings=Settings(sandbox=SANDBOX)
    )
    assert out["reward"] == 1.0
    obs = out["observation_tokens"]
    assert len(out["action_ranges"]) == 2
    s, e = out["action_ranges"][0]
    assert "write_file" in tok.decode(obs[s:e])
    assert out["rollout_log_probs"][s] == pytest.approx(-0.2)
    assert out["extra_logs"]["agentbox_sequences"] == 1.0  # TITO: no prefix breaks
    text = tok.decode(obs)
    assert "<tool_response>" in text and text.count("<|im_start|>assistant") == 2
