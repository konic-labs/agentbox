"""Trainer-agnostic RL helpers (advantages, pass@k, group statistics)."""

from agentbox.rl.stats import (
    group_advantages,
    group_stats,
    is_zero_variance,
    pass_at_k,
    pass_at_k_table,
)

__all__ = [
    "group_advantages",
    "group_stats",
    "is_zero_variance",
    "pass_at_k",
    "pass_at_k_table",
]
