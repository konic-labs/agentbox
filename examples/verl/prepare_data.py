"""Write verl RL datasets routed to the AgentBox agent loop.

    python examples/verl/prepare_data.py --train data/tasks/train.jsonl \
        --val data/tasks/eval.jsonl --out data/verl

The system prompt is rendered for the same tool set the workers use
(agentbox-verl.yaml), so the model sees the tools it can call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentbox import AgentConfig, iter_tasks
from agentbox.integrations.verl import write_verl_parquet

TOOLS = ["list_files", "read_file", "write_file", "run_tests"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/tasks/train.jsonl")
    ap.add_argument("--val", default="data/tasks/eval.jsonl")
    ap.add_argument("--out", default="data/verl")
    args = ap.parse_args()
    out, agent = Path(args.out), AgentConfig(tools=TOOLS)
    for name, src in (("train", args.train), ("val", args.val)):
        path = write_verl_parquet(iter_tasks(src), out / f"{name}.parquet", agent=agent)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
