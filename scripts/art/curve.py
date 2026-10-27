"""Build training.svg's data from real runs: verl console logs and an AgentBox comparison.

    python scripts/art/curve.py RUN1.log [RUN2.log ...] --compare compare.json \\
        --reports base/report.json trained/report.json > scripts/art/training.json

Each log gives one run: the training reward per step (``critic/score/mean``) and
the validation reward (``val-core/agentbox/reward/mean@1``, step 0 is the
validation before training). ``compare.json`` is ``agentbox --json bench compare
BASE.json TRAINED.json``; its success rates are the chart's verdict, and the two
reports give compare.svg its per-task success.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

STEP = re.compile(r"\bstep:(\d+) - ")
SCORE = re.compile(r"critic/score/mean:([0-9.]+)")
VAL = re.compile(r"val-core/agentbox/reward/mean@1:(?:np\.float64\()?([0-9.]+)")


def run_curve(path: str) -> dict[str, list[list[float]]]:
    train, val = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = STEP.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if s := SCORE.search(line):
                train.append([step, round(float(s.group(1)), 4)])
            if v := VAL.search(line):
                val.append([step, round(float(v.group(1)), 4)])
    return {"train": train, "val": val}


def per_task(base_report: str, trained_report: str) -> list[dict[str, int | str]]:
    """Successes per task, out of the samples both runs drew."""
    hits = []
    for path in (base_report, trained_report):
        with open(path) as fh:
            tasks = json.load(fh)["models"][0]["by_task"]
        hits.append({t["task_id"]: (t["successes"], t["n"]) for t in tasks})
    rows = []
    for tid in sorted(hits[0]):
        (base, n), (after, n_after) = hits[0][tid], hits[1][tid]
        if n != n_after:
            raise SystemExit(f"{tid}: {n} base samples but {n_after} trained samples")
        rows.append({"id": tid, "n": n, "base": base, "trained": after})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--compare", required=True, help="agentbox --json bench compare output")
    ap.add_argument("--reports", nargs=2, required=True, metavar=("BASE", "TRAINED"))
    ap.add_argument("--replicate", help="the bench compare of a second run, for the verdict's note")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    with open(args.compare) as fh:
        delta = json.load(fh)["deltas"][0]
    paired = delta["paired"]
    note = [f"{paired['tasks']} tasks × 8, {paired['wins']} better, {paired['losses']} worse"]
    if args.replicate:
        with open(args.replicate) as fh:
            other = json.load(fh)["deltas"][0]["success_rate_b"]
        note.append(f"first run: {other * 100:.1f}%")
    data = {
        "title": (
            f"Reward of {len(args.logs)} GRPO runs with AgentBox and verl on one GPU; held-out "
            f"success {delta['success_rate_a'] * 100:.1f}% before and "
            f"{delta['success_rate_b'] * 100:.1f}% after training"
        ),
        "heading": "Reward while training",
        "subheading": "Qwen3-0.6B, GRPO with verl 0.9.1 on one L4: 8 tasks × 8 samples per step",
        "legend_val": "validation, 9 held-out tasks",
        "legend_train": "training reward, 4-step mean",
        "legend_replicate": "first run",
        "footnote": "dots: training reward per step; validation is greedy; "
        "the verdict is agentbox bench run at temperature 0.6",
        "runs": [run_curve(p) for p in args.logs],
        "eval": {
            "label": "HELD-OUT SUCCESS",
            "base_label": "base model",
            "base": round(delta["success_rate_a"], 4),
            "trained_label": f"after {args.steps} steps",
            "trained": round(delta["success_rate_b"], 4),
            "note": note,
        },
        "compare_title": (
            f"Held-out tasks before and after training: {paired['wins']} better, "
            f"{paired['losses']} worse, {paired['ties']} unchanged"
        ),
        "compare_heading": "Held-out tasks, before and after.",
        "compare_subheading": (
            "agentbox bench compare: successes per task, base model and after "
            f"{args.steps} GRPO steps"
        ),
        "tasks": per_task(*args.reports),
    }
    json.dump(data, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
