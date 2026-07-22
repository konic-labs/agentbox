"""Generate coding tasks with a frontier model (optional DSPy).

Requires OPENAI_API_KEY / provider credentials and network-enabled Docker for QC.

  export OPENAI_API_KEY=...
  python examples/generate_tasks.py --model glm-5.2 --base-url https://api.featherless.ai/v1
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentbox.tasks.generate import GenerateConfig, TaskGenerator


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--out", default="tasks/generated")
    parser.add_argument("--no-docker-qc", action="store_true")
    parser.add_argument("--no-dspy", action="store_true")
    args = parser.parse_args()

    gen = TaskGenerator(
        GenerateConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            validate_in_docker=not args.no_docker_qc,
            use_dspy=not args.no_dspy,
            max_retries=2,
        )
    )
    tasks = await gen.generate_many(
        args.n,
        difficulty=args.difficulty,
        domain="python",
        constraints="single-file bug fix with pytest; agent must edit code",
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        path = out / f"{task.task_id}.json"
        task.save_json(path)
        print("wrote", path, "files=", list(task.starter_files))
    print(f"generated {len(tasks)}/{args.n} valid tasks")


if __name__ == "__main__":
    asyncio.run(main())
