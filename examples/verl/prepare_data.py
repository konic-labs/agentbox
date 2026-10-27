"""Write verl RL datasets routed to the AgentBox agent loop.

python examples/verl/prepare_data.py --tasks examples/tasks --out data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentbox import iter_tasks
from agentbox.integrations.verl import write_verl_parquet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="examples/tasks")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    tasks = list(iter_tasks(args.tasks))
    train = [t for t in tasks if t.split in (None, "train")]
    val = [t for t in tasks if t.split in ("eval", "test")] or train[:1]
    print(write_verl_parquet(train, out / "train.parquet"), len(train), "rows")
    print(write_verl_parquet(val, out / "val.parquet"), len(val), "rows")


if __name__ == "__main__":
    main()
