"""Drive a sandbox hosted by the HTTP environment server.

Starts `agentbox serve` in-process on a free port, then runs one scripted
episode through RemoteEnv, the same surface as AgentBoxEnv. In production the
server runs on the Docker host and the trainer only needs the URL and token.

    pip install "agentbox-rl[server]"
    python examples/remote_env.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import uvicorn

from agentbox.env.remote import RemoteEnv
from agentbox.server.app import ServerSettings, create_app

TASKS = Path(__file__).parent / "tasks"
FIX = "def reverse_string(s):\n    return s[::-1]\n"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def main() -> None:
    port = free_port()
    app = create_app(ServerSettings(tasks=str(TASKS), token="demo-token"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    env = RemoteEnv(f"http://127.0.0.1:{port}", token="demo-token")
    try:
        obs = await env.reset("fix_reverse_001")
        print("tools:", [t["function"]["name"] for t in obs.tools])
        call = {
            "id": "c0",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({"path": "reverse.py", "content": FIX}),
            },
        }
        await env.step({"role": "assistant", "content": None, "tool_calls": [call]})
        await env.step("The function now reverses its input.")
        result = await env.evaluate()
        print(f"reward={result.reward} success={result.success} failure_kind={result.failure_kind}")
        traj = await env.trajectory()
        print("trajectory:", traj.task_id, traj.final_status, len(traj.messages), "messages")
    finally:
        await env.close()
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
