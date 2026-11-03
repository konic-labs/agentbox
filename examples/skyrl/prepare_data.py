"""Write SkyRL datasets whose rows run in the AgentBox environment.

    python examples/skyrl/prepare_data.py --train data/tasks/train.jsonl \
        --val data/tasks/eval.jsonl --out data/skyrl

SkyRL hands every column other than ``prompt`` and ``env_class`` to the
environment, so the task goes in a top-level ``agentbox_task`` column. The
environment renders the real system prompt (with the tools) at ``init``; the
``prompt`` column only serves SkyRL's length filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentbox import iter_tasks


def rows(source: str) -> list[dict[str, object]]:
    return [
        {
            "data_source": "agentbox",
            "prompt": [{"role": "user", "content": t.prompt_text()}],
            "env_class": "agentbox",
            "agentbox_task": t.model_dump_json(exclude_defaults=True),
        }
        for t in iter_tasks(source)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/tasks/train.jsonl")
    ap.add_argument("--val", default="data/tasks/eval.jsonl")
    ap.add_argument("--out", default="data/skyrl")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, src in (("train", args.train), ("val", args.val)):
        path = out / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows(src)), path)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
