"""Write an OpenRLHF prompt dataset whose labels select AgentBox tasks.

python examples/openrlhf/prepare_data.py --tasks examples/tasks --out data/openrlhf.jsonl

``--repeat N`` writes every task N times, so a run can take as many full rollout
batches as it needs: OpenRLHF trains on a last, partial batch too, and with
several ranks that batch hung ZeRO-2's gradient all-reduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentbox import iter_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="examples/tasks")
    ap.add_argument("--out", default="data/openrlhf.jsonl")
    ap.add_argument("--repeat", type=int, default=1, help="write every task this many times")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = list(iter_tasks(args.tasks))
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for _ in range(args.repeat):
            for task in tasks:
                fh.write(json.dumps({"prompt": task.prompt_text(), "label": task.task_id}) + "\n")
                n += 1
    print(f"wrote {n} prompts to {out}")


if __name__ == "__main__":
    main()
