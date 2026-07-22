"""Execute frozen benchmark suites against model endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentbox.benchmark.loader import snapshot_suite
from agentbox.benchmark.report import build_leaderboard, build_model_report
from agentbox.benchmark.schema import (
    BenchmarkReport,
    BenchmarkRunConfig,
    ModelUnderTest,
)
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig
from agentbox.runner.parallel import ParallelRunner
from agentbox.trajectory.schema import Trajectory
from agentbox.version import __version__

logger = logging.getLogger("agentbox.benchmark")


class BenchmarkRunner:
    """Run a BenchmarkSuite against one or more OpenAI-compatible models."""

    def __init__(self, config: BenchmarkRunConfig) -> None:
        self.config = config

    async def run(self) -> BenchmarkReport:
        cfg = self.config
        suite = cfg.suite
        suite.verify_integrity(strict=cfg.strict_integrity)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        snapshot_suite(suite, out / "suite_snapshot")

        started = datetime.now(timezone.utc)
        model_reports = []
        for mut in cfg.models:
            logger.info("benchmark model_id=%s start", mut.model_id)
            mr = await self._run_model(mut)
            model_reports.append(mr)

        finished = datetime.now(timezone.utc)
        report = BenchmarkReport(
            report_id=str(uuid.uuid4()),
            suite_id=suite.manifest.suite_id,
            suite_version=suite.manifest.version,
            suite_content_hash=suite.manifest.content_hash,
            agentbox_version=__version__,
            started_at=started,
            finished_at=finished,
            run_tags=dict(cfg.tags),
            models=model_reports,
            leaderboard=build_leaderboard(model_reports),
        )
        report.save_json(out / "report.json")
        report.save_markdown(out / "REPORT.md")
        return report

    async def _run_model(self, mut: ModelUnderTest) -> Any:
        cfg = self.config
        suite = cfg.suite
        mdir = Path(cfg.output_dir) / "models" / mut.model_id
        traj_dir = mdir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Fairness: suite agent tools/max_steps only; model cannot change tools
        agent_cfg = suite.manifest.agent.model_copy(
            deep=True,
            update={"custom_tools": [], "drop_tools_prob": 0.0},
        )
        concurrency = cfg.concurrency or suite.manifest.concurrency
        n_per = cfg.n_per_task or suite.manifest.n_per_task
        seed = suite.manifest.seed

        rollout_config = RolloutConfig(
            sandbox=suite.manifest.sandbox,
            model=mut.model,
            agent=agent_cfg,
            seed=seed,
        )

        path_map: dict[str, list[str]] = defaultdict(list)
        trajs_acc: list[Trajectory] = []

        def on_trajectory(traj: Trajectory) -> None:
            trajs_acc.append(traj)
            if cfg.save_trajectories:
                fname = f"{traj.task_id}_{traj.run_id[:8]}.json"
                path = traj_dir / fname
                traj.save(path)
                path_map[traj.task_id].append(str(path))

        # Thread setup_checks into each Rollout via ParallelRunner extension
        runner = ParallelRunner(
            concurrency=concurrency,
            config=rollout_config,
            model=mut.model,
            agent=agent_cfg,
            sandbox=suite.manifest.sandbox,
        )
        setup_checks = list(suite.manifest.scoring.setup_checks)

        t0 = time.monotonic()
        trajs = await runner.run_tasks(
            suite.tasks,
            n_per_task=n_per,
            progress=cfg.progress,
            on_trajectory=on_trajectory,
            setup_checks=setup_checks,
        )
        wall = time.monotonic() - t0

        public = mut.model.model_dump(mode="json")
        report = build_model_report(
            mut.model_id,
            public,
            trajs,
            wall_clock_s=wall,
            trajectory_paths=dict(path_map),
        )
        report_path = mdir / "model_report.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report
