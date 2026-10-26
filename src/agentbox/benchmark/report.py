"""Build and render benchmark reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentbox.benchmark.schema import BenchmarkReport, ModelReport, TaskResult
from agentbox.metrics.aggregate import MetricsAccumulator
from agentbox.rl.stats import pass_at_k
from agentbox.trajectory.schema import Trajectory
from agentbox.types import FailureKind, FinalStatus

PASS_KS = (1, 5, 10)


def redact_model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    if "api_key" in out:
        out["api_key"] = "***" if out["api_key"] else None
    hdrs = out.get("extra_headers")
    if isinstance(hdrs, dict):
        out["extra_headers"] = {k: "***" for k in hdrs}
    return out


class TaskTally:
    """Per-task counters accumulated while streaming (no trajectories kept)."""

    __slots__ = (
        "n",
        "successes",
        "reward_sum",
        "steps_sum",
        "duration_sum",
        "statuses",
        "seed_errors",
        "setup_fails",
        "infra",
    )

    def __init__(self) -> None:
        self.n = 0
        self.successes = 0
        self.reward_sum = 0.0
        self.steps_sum = 0
        self.duration_sum = 0.0
        self.statuses: dict[str, int] = {}
        self.seed_errors = 0
        self.setup_fails = 0
        self.infra = 0

    def add(self, t: Trajectory) -> None:
        self.n += 1
        self.successes += t.final_status == FinalStatus.SUCCESS
        self.reward_sum += t.reward
        self.steps_sum += t.metrics.steps
        self.duration_sum += t.metrics.duration_s
        self.statuses[t.final_status.value] = self.statuses.get(t.final_status.value, 0) + 1
        kind = FailureKind(t.failure_kind)
        self.seed_errors += kind == FailureKind.SEED
        self.setup_fails += kind == FailureKind.SETUP_CHECK
        self.infra += kind.is_infra


class ReportBuilder:
    """Streaming accumulator for one model's benchmark results."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.metrics = MetricsAccumulator()
        self.tasks: dict[str, TaskTally] = {}
        self.paths: dict[str, list[str]] = {}

    def add(self, t: Trajectory, path: str | None = None) -> None:
        self.metrics.add(t)
        self.tasks.setdefault(t.task_id, TaskTally()).add(t)
        if path:
            self.paths.setdefault(t.task_id, []).append(path)

    def task_results(self) -> list[TaskResult]:
        results = []
        for task_id in sorted(self.tasks):
            tt = self.tasks[task_id]
            n = tt.n
            results.append(
                TaskResult(
                    task_id=task_id,
                    model_id=self.model_id,
                    trajectory_paths=self.paths.get(task_id, []),
                    n=n,
                    successes=tt.successes,
                    pass_at_1=pass_at_k(n, tt.successes, 1) if n else 0.0,
                    pass_at_k=pass_at_k(n, tt.successes, n) if n else None,
                    pass_at={str(k): pass_at_k(n, tt.successes, k) for k in PASS_KS if k <= n},
                    mean_reward=tt.reward_sum / n if n else 0.0,
                    mean_steps=tt.steps_sum / n if n else 0.0,
                    mean_duration_s=tt.duration_sum / n if n else 0.0,
                    seed_errors=tt.seed_errors,
                    setup_check_failures=tt.setup_fails,
                    infra_failures=tt.infra,
                    statuses=dict(tt.statuses),
                )
            )
        return results

    def build(self, model_config_public: dict[str, Any], *, wall_clock_s: float) -> ModelReport:
        by_task = self.task_results()
        pass_at: dict[str, float] = {}
        for k in PASS_KS:
            vals = [t.pass_at[str(k)] for t in by_task if str(k) in t.pass_at]
            if vals and len(vals) == len(by_task):
                pass_at[str(k)] = sum(vals) / len(vals)
        return ModelReport(
            model_id=self.model_id,
            model_config_public=redact_model_config(model_config_public),
            aggregate=self.metrics.result(),
            by_task=by_task,
            wall_clock_s=wall_clock_s,
            pass_at_k_mean=(
                sum(t.pass_at_k or 0.0 for t in by_task) / len(by_task) if by_task else None
            ),
            pass_at=pass_at,
        )


def build_task_results(
    model_id: str,
    trajs: Sequence[Trajectory],
    *,
    trajectory_paths: dict[str, list[str]] | None = None,
) -> list[TaskResult]:
    builder = ReportBuilder(model_id)
    for t in trajs:
        builder.add(t)
    builder.paths = dict(trajectory_paths or {})
    return builder.task_results()


def build_model_report(
    model_id: str,
    model_config_public: dict[str, Any],
    trajs: Sequence[Trajectory],
    *,
    wall_clock_s: float,
    trajectory_paths: dict[str, list[str]] | None = None,
) -> ModelReport:
    builder = ReportBuilder(model_id)
    for t in trajs:
        builder.add(t)
    builder.paths = dict(trajectory_paths or {})
    return builder.build(model_config_public, wall_clock_s=wall_clock_s)


def build_leaderboard(models: Sequence[ModelReport]) -> list[dict[str, Any]]:
    rows = []
    for m in models:
        row: dict[str, Any] = {
            "model_id": m.model_id,
            "success_rate": m.aggregate.success_rate,
            "mean_reward": m.aggregate.mean_reward,
            "mean_steps": m.aggregate.mean_steps,
            "mean_duration_s": m.aggregate.mean_duration_s,
            "pass_at_k_mean": m.pass_at_k_mean,
            "n": m.aggregate.n,
            "infra_failures": m.aggregate.infra_failures,
            "wall_clock_s": m.wall_clock_s,
        }
        for k, v in m.pass_at.items():
            row[f"pass@{k}"] = v
        rows.append(row)
    rows.sort(key=lambda r: (-r["success_rate"], -r["mean_reward"], r["mean_steps"]))
    return rows


def render_markdown(report: BenchmarkReport) -> str:
    ks = sorted({k for m in report.models for k in m.pass_at}, key=int)
    pass_cols = "".join(f" pass@{k} |" for k in ks)
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
        "| Rank | Model | Success rate |"
        + pass_cols
        + " Mean reward | Mean steps | Mean duration (s) | Infra failures | n |",
        "| --- | --- | --- |" + " --- |" * len(ks) + " --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(report.leaderboard, start=1):
        pass_vals = "".join(
            f" {row[f'pass@{k}']:.3f} |" if f"pass@{k}" in row else " — |" for k in ks
        )
        lines.append(
            f"| {i} | {row['model_id']} | {row['success_rate']:.3f} |{pass_vals} "
            f"{row['mean_reward']:.3f} | {row['mean_steps']:.2f} | "
            f"{row['mean_duration_s']:.2f} | {row.get('infra_failures', 0)} | {row['n']} |"
        )

    for m in report.models:
        lines.extend(
            [
                "",
                f"## Model: {m.model_id}",
                "",
                f"- Config: `{m.model_config_public}`",
                f"- Success rate: **{m.aggregate.success_rate:.3f}**",
                f"- Mean reward: {m.aggregate.mean_reward:.3f} (std {m.aggregate.reward_std:.3f})",
                f"- Mean steps: {m.aggregate.mean_steps:.2f}",
                f"- Statuses: {m.aggregate.by_status}",
                f"- Failure kinds: {m.aggregate.by_failure_kind}",
                f"- Wall clock: {m.wall_clock_s:.1f}s",
                "",
                "| Task | n | Successes | Pass@1 | Pass@n | Mean reward | Mean steps | Infra |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for t in m.by_task:
            pak = f"{t.pass_at_k:.3f}" if t.pass_at_k is not None else "—"
            lines.append(
                f"| {t.task_id} | {t.n} | {t.successes} | {t.pass_at_1:.3f} | "
                f"{pak} | {t.mean_reward:.3f} | {t.mean_steps:.2f} | {t.infra_failures} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Pass@k definition: {report.pass_at_k_definition}",
            "- Infra failures (sandbox/seed/model API/verifier) are retried and excluded "
            "from training exports; they still count as failures here.",
            "- API keys are redacted from this report.",
            "",
        ]
    )
    return "\n".join(lines)
