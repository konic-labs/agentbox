"""Trainer-driven episodes with the step-wise environment API.

Your code owns generation (here: the OpenAI SDK against any compatible server);
AgentBox owns the sandbox, the tools and the reward. This is the shape every
trainer integration builds on.

    python examples/env_loop.py --base-url http://localhost:8000/v1 --model qwen3-1.7b
    python examples/env_loop.py --scripted        # no model: a fixed tool-calling policy
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agentbox import AgentBoxEnv, AgentConfig, SandboxConfig, Task

FIX = (
    "def fizzbuzz(n):\n"
    "    if n % 15 == 0:\n        return 'FizzBuzz'\n"
    "    if n % 3 == 0:\n        return 'Fizz'\n"
    "    if n % 5 == 0:\n        return 'Buzz'\n"
    "    return str(n)\n"
)


def scripted(turn: int) -> dict[str, Any]:
    """Assistant messages in OpenAI format: write the fix, run the tests, finish."""
    if turn == 0:
        args = json.dumps({"path": "fizzbuzz.py", "content": FIX})
        fn = {"name": "write_file", "arguments": args}
    elif turn == 1:
        fn = {"name": "run_tests", "arguments": "{}"}
    else:
        return {"role": "assistant", "content": "Done: the tests pass."}
    call = {"id": f"c{turn}", "type": "function", "function": fn}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--scripted", action="store_true")
    args = ap.parse_args()

    client = None
    if not args.scripted:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    task = Task.from_json(Path(__file__).parent / "tasks" / "fix_fizzbuzz" / "task.json")
    env = AgentBoxEnv(sandbox=SandboxConfig(), agent=AgentConfig(max_steps=8))
    obs = await env.reset(task)
    messages, turn = list(obs.messages), 0
    while not env.done:
        if client is None:
            reply = scripted(turn)
        else:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=messages,  # type: ignore[arg-type]
                tools=obs.tools,  # type: ignore[arg-type]
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            reply = resp.choices[0].message.model_dump(exclude_none=True)
        messages.append(reply)
        step = await env.step(reply)  # runs the tool calls in the sandbox
        messages.extend(step.messages)
        turn += 1

    result = await env.evaluate()
    traj = env.trajectory(model=args.model if client else "scripted")
    print(f"reward={result.reward} success={result.success} steps={traj.metrics.steps}")
    print("verifier:", traj.verify.exit_code if traj.verify else None, result.breakdown)
    await env.close()
    if client is not None:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
