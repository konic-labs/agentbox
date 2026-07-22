"""Run the coding-mini benchmark with a hermetic mock model (Docker required)."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import uuid

from agentbox.benchmark.loader import load_suite
from agentbox.benchmark.report import build_leaderboard, build_model_report
from agentbox.benchmark.schema import BenchmarkReport
from agentbox.config import ModelConfig, RolloutConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.runner.parallel import ParallelRunner
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.version import __version__

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "benchmarks" / "coding-mini"
OUT = ROOT.parent / "bench-results" / "coding-mini-mock"


def _fix_scripts() -> list[ModelResponse]:
    """Scripted tool calls that solve both mini suite tasks when present."""
    fizz = '''def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
'''
    rev = '''def reverse_string(s):
    return s[::-1]
'''
    steps: list[ModelResponse] = []
    for path, content in (("fizzbuzz.py", fizz), ("reverse.py", rev)):
        steps.append(
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps({"path": path, "content": content}),
                        ),
                    )
                ],
            )
        )
        steps.append(ModelResponse(content="done", tool_calls=[]))
    # pad script for extra model calls
    steps.extend([ModelResponse(content="done", tool_calls=[])] * 20)
    return steps


async def main() -> None:
    suite = load_suite(SUITE)
    assert suite.verify_integrity(), "suite hash mismatch — re-freeze"
    agent = suite.manifest.agent.model_copy(update={"custom_tools": []})
    mock = MockModelClient(_fix_scripts())
    runner = ParallelRunner(
        concurrency=2,
        config=RolloutConfig(
            sandbox=suite.manifest.sandbox,
            model=ModelConfig(model="mock"),
            agent=agent,
            seed=suite.manifest.seed,
        ),
        model=mock,
        agent=agent,
        sandbox=suite.manifest.sandbox,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    traj_dir = OUT / "models" / "mock-solver" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[str]] = defaultdict(list)

    def on_t(traj) -> None:
        p = traj_dir / f"{traj.task_id}_{traj.run_id[:8]}.json"
        traj.save(p)
        paths[traj.task_id].append(str(p))

    t0 = time.monotonic()
    trajs = await runner.run_tasks(
        suite.tasks,
        n_per_task=1,
        progress=True,
        on_trajectory=on_t,
        setup_checks=list(suite.manifest.scoring.setup_checks),
    )
    wall = time.monotonic() - t0
    mr = build_model_report(
        "mock-solver",
        {"model": "mock"},
        trajs,
        wall_clock_s=wall,
        trajectory_paths=dict(paths),
    )
    report = BenchmarkReport(
        report_id=str(uuid.uuid4()),
        suite_id=suite.manifest.suite_id,
        suite_version=suite.manifest.version,
        suite_content_hash=suite.manifest.content_hash,
        agentbox_version=__version__,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        models=[mr],
        leaderboard=build_leaderboard([mr]),
    )
    report.save_json(OUT / "report.json")
    report.save_markdown(OUT / "REPORT.md")
    print(report.leaderboard)
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    asyncio.run(main())
