"""Benchmark suite, run, and report schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentbox.config import AgentConfig, ModelConfig, SandboxConfig
from agentbox.metrics.aggregate import AggregateMetrics
from agentbox.tasks.schema import Task
from agentbox.types import ToolMode


class SetupCheckSpec(BaseModel):
    """Post-seed environment health check (not task success)."""

    name: str
    command: str
    success_exit_code: int = 0
    timeout_s: float = 60.0
    required: bool = True


class ScoringConfig(BaseModel):
    """How a suite scores rollouts (the task verifier is the ground truth)."""

    model_config = ConfigDict(extra="ignore")  # tolerate removed v1 fields

    primary: Literal["verifier"] = "verifier"
    setup_checks: list[SetupCheckSpec] = Field(default_factory=list)


def _default_bench_agent() -> AgentConfig:
    return AgentConfig(
        tools=ToolMode.STRUCTURED,
        max_steps=40,
        drop_tools_prob=0.0,
        custom_tools=[],
    )


class BenchmarkSuiteManifest(BaseModel):
    """suite.json manifest for a frozen evaluation pack."""

    schema_version: Literal["1"] = "1"
    suite_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    tasks_path: str = "tasks"
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    agent: AgentConfig = Field(default_factory=_default_bench_agent)
    n_per_task: int = 1
    concurrency: int = 8
    seed: int | None = 42
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    content_hash: str | None = None
    hash_version: int | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class BenchmarkSuite(BaseModel):
    """In-memory suite: manifest + loaded tasks."""

    root: Path | None = None
    manifest: BenchmarkSuiteManifest
    tasks: list[Task] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks]

    def freeze(self) -> BenchmarkSuite:
        from agentbox.benchmark.hash import HASH_VERSION, compute_suite_content_hash

        m = self.manifest.model_copy(deep=True)
        m.created_at = m.created_at or datetime.now(UTC)
        m.content_hash = compute_suite_content_hash(self.tasks, m)
        m.hash_version = HASH_VERSION
        return BenchmarkSuite(root=self.root, manifest=m, tasks=list(self.tasks))

    def verify_integrity(self, *, strict: bool = False) -> bool:
        from agentbox.benchmark.hash import compute_suite_content_hash
        from agentbox.errors import SuiteIntegrityError

        if not self.manifest.content_hash:
            if strict:
                raise SuiteIntegrityError("Suite has no content_hash; run freeze()")
            return True
        from agentbox.benchmark.hash import HASH_VERSION

        current = compute_suite_content_hash(self.tasks, self.manifest)
        ok = current == self.manifest.content_hash
        if not ok and strict:
            hint = ""
            if (self.manifest.hash_version or 1) < HASH_VERSION:
                hint = " (suite frozen by an older AgentBox; run `agentbox bench freeze`)"
            raise SuiteIntegrityError(
                f"Suite integrity check failed: expected {self.manifest.content_hash}, "
                f"got {current}{hint}"
            )
        return ok


class ModelUnderTest(BaseModel):
    """One model endpoint under evaluation."""

    model_id: str
    model: ModelConfig


class BenchmarkRunConfig(BaseModel):
    """Configuration for executing a suite against one or more models."""

    suite: BenchmarkSuite
    models: list[ModelUnderTest]
    concurrency: int | None = None
    n_per_task: int | None = None
    save_trajectories: bool = True
    output_dir: Path
    progress: bool = True
    fail_fast: bool = Field(
        default=False, description="Stop a model's run at its first infrastructure failure."
    )
    strict_integrity: bool = False
    resume: bool = Field(default=False, description="Skip rollouts already in output_dir.")
    tags: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskResult(BaseModel):
    task_id: str
    model_id: str
    trajectory_paths: list[str] = Field(
        default_factory=list, description="Trajectory files, relative to the report's directory."
    )
    n: int = 0
    successes: int = 0
    pass_at_1: float = 0.0
    pass_at_k: float | None = None  # pass@n (k = samples per task)
    pass_at: dict[str, float] = Field(default_factory=dict)  # unbiased pass@k, k <= n
    mean_reward: float = 0.0
    mean_steps: float = 0.0
    mean_duration_s: float = 0.0
    seed_errors: int = 0
    setup_check_failures: int = 0
    infra_failures: int = 0
    statuses: dict[str, int] = Field(default_factory=dict)


class ModelReport(BaseModel):
    model_id: str
    model_config_public: dict[str, Any]
    aggregate: AggregateMetrics
    by_task: list[TaskResult] = Field(default_factory=list)
    wall_clock_s: float = 0.0
    pass_at_k_mean: float | None = None
    pass_at: dict[str, float] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    schema_version: Literal["1"] = "1"
    report_id: str
    suite_id: str
    suite_version: str
    suite_content_hash: str | None = None
    agentbox_version: str
    started_at: datetime
    finished_at: datetime
    run_tags: dict[str, str] = Field(default_factory=dict)
    models: list[ModelReport] = Field(default_factory=list)
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    pass_at_k_definition: str = (
        "unbiased estimator pass@k = 1 - C(n-c, k) / C(n, k) per task (Chen et al., 2021), "
        "averaged over tasks; pass@n (pass_at_k) = fraction of tasks with >= 1 success"
    )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def save_markdown(self, path: str | Path) -> None:
        from agentbox.benchmark.report import render_markdown

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(self), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


class ComparisonResult(BaseModel):
    suite_id: str
    hash_match: bool
    warnings: list[str] = Field(default_factory=list)
    deltas: list[dict[str, Any]] = Field(default_factory=list)
