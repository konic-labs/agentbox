"""Trainer-agnostic RL statistics: pass@k, group advantages, group filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

AdvantageMethod = Literal["grpo", "drgrpo", "rloo", "mean"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): ``1 - C(n-c, k) / C(n, k)``.

    ``n`` samples, ``c`` of them correct. Requires ``k <= n``.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples (got {n})")
    if n - c < k:
        return 1.0
    # numerically stable product form
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1.0 - k / i
    return 1.0 - prob_all_fail


def pass_at_k_table(
    counts: Sequence[tuple[int, int]], ks: Sequence[int] = (1, 5, 10)
) -> dict[int, float]:
    """Mean pass@k over tasks for every k that every task supports."""
    if not counts:
        return {}
    min_n = min(n for n, _ in counts)
    out: dict[int, float] = {}
    for k in ks:
        if k <= min_n:
            out[k] = sum(pass_at_k(n, c, k) for n, c in counts) / len(counts)
    return out


def group_advantages(
    rewards: Sequence[float],
    *,
    method: AdvantageMethod = "grpo",
    eps: float = 1e-6,
) -> list[float]:
    """Per-sample advantages within one group.

    * ``grpo``  — (r - mean) / (std + eps)            (DeepSeekMath)
    * ``drgrpo`` — r - mean, no std scaling           (Dr. GRPO, Liu et al. 2025)
    * ``rloo``  — r - mean of the *other* samples     (leave-one-out baseline)
    * ``mean``  — alias of ``drgrpo``
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    if method in ("drgrpo", "mean"):
        return [r - mean for r in rewards]
    if method == "rloo":
        if n == 1:
            return [0.0]
        total = sum(rewards)
        return [r - (total - r) / (n - 1) for r in rewards]
    if method == "grpo":
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
        return [(r - mean) / (std + eps) for r in rewards]
    raise ValueError(f"unknown advantage method {method!r}")


def is_zero_variance(rewards: Sequence[float], *, tol: float = 1e-9) -> bool:
    """True when every sample has the same reward (no learning signal in GRPO)."""
    return len(rewards) > 0 and max(rewards) - min(rewards) <= tol


def group_stats(groups: Sequence[Sequence[float]]) -> dict[str, float]:
    """Fractions of all-correct / all-wrong / mixed groups and mean in-group std."""
    if not groups:
        return {"groups": 0}
    solved = failed = mixed = 0
    stds = []
    for g in groups:
        if not g:
            continue
        if is_zero_variance(g):
            if max(g) > 0:
                solved += 1
            else:
                failed += 1
        else:
            mixed += 1
        m = sum(g) / len(g)
        stds.append(math.sqrt(sum((r - m) ** 2 for r in g) / len(g)))
    total = solved + failed + mixed
    return {
        "groups": total,
        "frac_all_solved": solved / total if total else 0.0,
        "frac_all_failed": failed / total if total else 0.0,
        "frac_mixed": mixed / total if total else 0.0,
        "frac_zero_variance": (solved + failed) / total if total else 0.0,
        "mean_group_reward_std": sum(stds) / len(stds) if stds else 0.0,
    }
