"""Lightweight job descriptors for train-cluster orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobBase(BaseModel):
    job_id: str
    kind: str
    tags: dict[str, str] = Field(default_factory=dict)
    artifact_root: Path | None = None


class GenerateJob(JobBase):
    kind: Literal["generate"] = "generate"
    target: int = 20
    out_dir: Path = Path("generated/tasks")
    model: str
    base_url: str | None = None
    api_key: str | None = None
    concurrency: int = 8
    docker_concurrency: int = 8
    two_stage: bool = False
    enable_llm_qc: bool = True
    enable_docker_qc: bool = True
    enable_static_qc: bool = True
    min_score: float = 0.65
    constraints_file: Path | None = None


class RolloutJob(JobBase):
    kind: Literal["rollout"] = "rollout"
    tasks_dir: Path
    model: str
    base_url: str | None = None
    api_key: str | None = None
    concurrency: int = 8
    n_per_task: int = 1
    max_steps: int = 40
    out_dir: Path = Path("trajectories")


class BenchJob(JobBase):
    kind: Literal["bench"] = "bench"
    suite_dir: Path
    out_dir: Path = Path("bench-results/latest")
    students: list[dict[str, Any]] = Field(default_factory=list)
    concurrency: int | None = None
    n_per_task: int | None = None
    limit: int | None = None
    probe: bool = False
    save_trajectories: bool = True


def job_from_dict(data: dict[str, Any]) -> GenerateJob | RolloutJob | BenchJob:
    kind = data.get("kind")
    if kind == "generate":
        return GenerateJob.model_validate(data)
    if kind == "rollout":
        return RolloutJob.model_validate(data)
    if kind == "bench":
        return BenchJob.model_validate(data)
    raise ValueError(f"unknown job kind: {kind!r}")
