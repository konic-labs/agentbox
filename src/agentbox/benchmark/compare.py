"""Compare two benchmark reports with paired, per-task statistics (GH #83)."""

from __future__ import annotations

import math
import random
from typing import Any

from agentbox.benchmark.schema import BenchmarkReport, ComparisonResult


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value for k successes out of n (p = 0.5)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def paired_stats(
    a: dict[str, float], b: dict[str, float], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Per-task paired comparison of success rates (b - a)."""
    tasks = sorted(set(a) & set(b))
    diffs = [b[t] - a[t] for t in tasks]
    if not diffs:
        return {"tasks": 0}
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return {
        "tasks": len(tasks),
        "mean_delta": mean,
        "ci95": [lo, hi],
        "wins": wins,
        "losses": losses,
        "ties": len(diffs) - wins - losses,
        "sign_test_p": _binom_two_sided(wins, wins + losses),
        "regressed_tasks": [t for t, d in zip(tasks, diffs, strict=True) if d < 0],
    }


def compare_reports(
    a: BenchmarkReport,
    b: BenchmarkReport,
    *,
    require_hash_match: bool = False,
) -> ComparisonResult:
    warnings: list[str] = []
    if a.suite_id != b.suite_id:
        warnings.append(f"suite_id differs: {a.suite_id!r} vs {b.suite_id!r}")
    hash_match = (a.suite_content_hash or "") == (b.suite_content_hash or "")
    if not hash_match:
        warnings.append(
            f"content_hash differs: {a.suite_content_hash!r} vs {b.suite_content_hash!r}"
        )
        if require_hash_match:
            warnings.append("require_hash_match=True: comparison may be unfair")

    a_models = {m.model_id: m for m in a.models}
    b_models = {m.model_id: m for m in b.models}
    deltas: list[dict[str, Any]] = []
    for mid in sorted(set(a_models) | set(b_models)):
        ma, mb = a_models.get(mid), b_models.get(mid)
        if ma is None or mb is None:
            warnings.append(f"model {mid!r} missing in one report")
            continue
        per_a = {t.task_id: t.pass_at_1 for t in ma.by_task}
        per_b = {t.task_id: t.pass_at_1 for t in mb.by_task}
        deltas.append(
            {
                "model_id": mid,
                "success_rate_a": ma.aggregate.success_rate,
                "success_rate_b": mb.aggregate.success_rate,
                "delta_success_rate": mb.aggregate.success_rate - ma.aggregate.success_rate,
                "mean_steps_a": ma.aggregate.mean_steps,
                "mean_steps_b": mb.aggregate.mean_steps,
                "delta_mean_steps": mb.aggregate.mean_steps - ma.aggregate.mean_steps,
                "paired": paired_stats(per_a, per_b),
            }
        )
    return ComparisonResult(
        suite_id=a.suite_id,
        hash_match=hash_match,
        warnings=warnings,
        deltas=deltas,
    )
