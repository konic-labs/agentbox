"""Build and render benchmark reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from agentbox.benchmark.schema import (
    BenchmarkReport,
    ModelReport,
    TaskResult,
)
from agentbox.metrics.aggregate import AggregateMetrics, aggregate_trajectories
from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus


def redact_model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    if "api_key" in out:
        out["api_key"] = "***" if out["api_key"] else None
    return out


def build_task_results(
    model_id: str,
    trajs: Sequence[Trajectory],
    *,
    trajectory_paths: dict[str, list[str]] | None = None,
) -> list[TaskResult]:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        by_task[t.task_id].append(t)

    results: list[TaskResult] = []
    for task_id, group in sorted(by_task.items()):
        n = len(group)
        successes = sum(1 for x in group if x.final_status == FinalStatus.SUCCESS)
        statuses: dict[str, int] = defaultdict(int)
        seed_errors = 0
        setup_fails = 0
        for x in group:
            statuses[x.final_status.value] += 1
            err = (x.error or "").lower()
            if x.final_status == FinalStatus.ERROR:
                if "setup check" in err:
                    setup_fails += 1
                elif "seed" in err:
                    seed_errors += 1
        pass_at_k = 1.0 if successes > 0 else 0.0
        results.append(
            TaskResult(
                task_id=task_id,
                model_id=model_id,
                trajectory_paths=(trajectory_paths or {}).get(task_id, []),
                n=n,
                successes=successes,
                pass_at_1=successes / n if n else 0.0,
                pass_at_k=pass_at_k,
                mean_reward=sum(x.reward for x in group) / n if n else 0.0,
                mean_steps=sum(x.metrics.steps for x in group) / n if n else 0.0,
                mean_duration_s=sum(x.metrics.duration_s for x in group) / n if n else 0.0,
                seed_errors=seed_errors,
                setup_check_failures=setup_fails,
                statuses=dict(statuses),
            )
        )
    return results


def build_model_report(
    model_id: str,
    model_config_public: dict[str, Any],
    trajs: Sequence[Trajectory],
    *,
    wall_clock_s: float,
    trajectory_paths: dict[str, list[str]] | None = None,
) -> ModelReport:
    by_task = build_task_results(
        model_id, trajs, trajectory_paths=trajectory_paths
    )
    agg = aggregate_trajectories(trajs)
    pass_at_k_mean = (
        sum(t.pass_at_k or 0.0 for t in by_task) / len(by_task) if by_task else None
    )
    return ModelReport(
        model_id=model_id,
        model_config_public=redact_model_config(model_config_public),
        aggregate=agg,
        by_task=by_task,
        wall_clock_s=wall_clock_s,
        pass_at_k_mean=pass_at_k_mean,
    )


def build_leaderboard(models: Sequence[ModelReport]) -> list[dict[str, Any]]:
    rows = []
    for m in models:
        rows.append(
            {
                "model_id": m.model_id,
                "success_rate": m.aggregate.success_rate,
                "mean_reward": m.aggregate.mean_reward,
                "mean_steps": m.aggregate.mean_steps,
                "mean_duration_s": m.aggregate.mean_duration_s,
                "pass_at_k_mean": m.pass_at_k_mean,
                "n": m.aggregate.n,
                "wall_clock_s": m.wall_clock_s,
            }
        )
    rows.sort(key=lambda r: (-r["success_rate"], -r["mean_reward"], r["mean_steps"]))
    return rows


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = [
        f"# Benchmark Report: {report.suite_id}",
        "",
        f"- **Suite version:** {report.suite_version}",
        f"- **Content hash:** `{report.suite_content_hash or 'none'}`",
        f"- **AgentBox:** {report.agentbox_version}",
        f"- **Report id:** {report.report_id}",
        f"- **Started:** {report.started_at.isoformat()}",
        f"- **Finished:** {report.finished_at.isoformat()}",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Success rate | Mean reward | Mean steps | Mean duration (s) | Pass@k | n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(report.leaderboard, start=1):
        pak = row.get("pass_at_k_mean")
        pak_s = f"{pak:.3f}" if pak is not None else "—"
        lines.append(
            f"| {i} | {row['model_id']} | {row['success_rate']:.3f} | "
            f"{row['mean_reward']:.3f} | {row['mean_steps']:.2f} | "
            f"{row['mean_duration_s']:.2f} | {pak_s} | {row['n']} |"
        )

    for m in report.models:
        lines.extend(
            [
                "",
                f"## Model: {m.model_id}",
                "",
                f"- Config: `{m.model_config_public}`",
                f"- Success rate: **{m.aggregate.success_rate:.3f}**",
                f"- Mean reward: {m.aggregate.mean_reward:.3f}",
                f"- Mean steps: {m.aggregate.mean_steps:.2f}",
                f"- Statuses: {m.aggregate.by_status}",
                f"- Wall clock: {m.wall_clock_s:.1f}s",
                "",
                "| Task | n | Successes | Pass@1 | Pass@k | Mean reward | Mean steps |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for t in m.by_task:
            pak = f"{t.pass_at_k:.3f}" if t.pass_at_k is not None else "—"
            lines.append(
                f"| {t.task_id} | {t.n} | {t.successes} | {t.pass_at_1:.3f} | "
                f"{pak} | {t.mean_reward:.3f} | {t.mean_steps:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Pass@k definition: {report.pass_at_k_definition}",
            "- API keys are redacted from this report.",
            "",
        ]
    )
    return "\n".join(lines)
