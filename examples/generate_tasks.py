"""Generate coding tasks with a teacher model (optional DSPy + LLM judge).

Any OpenAI-compatible endpoint works (self-hosted vLLM on EC2, local, etc.).

  export OPENAI_API_KEY=EMPTY   # or real key if required
  python examples/generate_tasks.py \
    --model Qwen/Qwen3.6-27B \
    --base-url http://127.0.0.1:8000/v1 \
    --n 5
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
    parser.add_argument("--no-llm-qc", action="store_true", help="Skip DSPy LLM task judge")
    parser.add_argument("--no-dspy", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.65)
    args = parser.parse_args()

    gen = TaskGenerator(
        GenerateConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            validate_in_docker=not args.no_docker_qc,
            validate_with_llm=not args.no_llm_qc,
            llm_judge_min_score=args.min_score,
            use_dspy=not args.no_dspy,
            max_retries=2,
        )
    )
    tasks = await gen.generate_many(
        args.n,
        difficulty=args.difficulty,
        domain="python",
        constraints=(
            "stub starter only (NotImplementedError); full pytest contract; "
            "agent must implement logic; no solution spoilers"
        ),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        path = out / f"{task.task_id}.json"
        task.save_json(path)
        print(
            "wrote",
            path,
            "files=",
            list(task.starter_files),
            "llm_score=",
            task.metadata.get("llm_judge_score"),
        )
    print(f"generated {len(tasks)}/{args.n} valid tasks")


if __name__ == "__main__":
    asyncio.run(main())
