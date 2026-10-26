"""Execute frozen benchmark suites against model endpoints (streaming, resumable)."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentbox.benchmark.loader import snapshot_suite
from agentbox.benchmark.report import ReportBuilder, build_leaderboard
from agentbox.benchmark.schema import (
    BenchmarkReport,
    BenchmarkRunConfig,
    ModelReport,
    ModelUnderTest,
)
from agentbox.config import RolloutConfig
from agentbox.model.base import ModelClient
from agentbox.runner.parallel import ParallelRunner
from agentbox.trajectory.io import DirectorySink, iter_trajectories
from agentbox.version import __version__

logger = logging.getLogger("agentbox.benchmark")


class BenchmarkRunner:
    """Run a BenchmarkSuite against one or more OpenAI-compatible models."""

    def __init__(
        self,
        config: BenchmarkRunConfig,
        *,
        model_clients: dict[str, ModelClient] | None = None,
    ) -> None:
        self.config = config
        # Optional pre-built clients per model_id (e.g. MockModelClient for --mock)
        self.model_clients = dict(model_clients or {})

    async def run(self) -> BenchmarkReport:
        cfg = self.config
        suite = cfg.suite
        if not suite.verify_integrity(strict=cfg.strict_integrity):
            logger.warning("suite content hash does not match; results may not be comparable")

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not (cfg.resume and (out / "suite_snapshot").exists()):
            snapshot_suite(suite, out / "suite_snapshot")

        started = datetime.now(timezone.utc)
        model_reports = []
        for mut in cfg.models:
            logger.info("benchmark model_id=%s start", mut.model_id)
            model_reports.append(await self._run_model(mut))

        report = BenchmarkReport(
            report_id=str(uuid.uuid4()),
            suite_id=suite.manifest.suite_id,
            suite_version=suite.manifest.version,
            suite_content_hash=suite.manifest.content_hash,
            agentbox_version=__version__,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            run_tags=dict(cfg.tags),
            models=model_reports,
            leaderboard=build_leaderboard(model_reports),
        )
        report.save_json(out / "report.json")
        report.save_markdown(out / "REPORT.md")
        return report

    async def _run_model(self, mut: ModelUnderTest) -> ModelReport:
        cfg = self.config
        suite = cfg.suite
        mdir = Path(cfg.output_dir) / "models" / mut.model_id
        traj_dir = mdir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Fairness: the suite's agent settings apply to every model.
        agent_cfg = suite.manifest.agent.model_copy(
            deep=True, update={"custom_tools": [], "drop_tools_prob": 0.0}
        )
        rollout_config = RolloutConfig(
            sandbox=suite.manifest.sandbox,
            model=mut.model,
            agent=agent_cfg,
            seed=suite.manifest.seed,
        )
        builder = ReportBuilder(mut.model_id)
        skip: set[tuple[str, int]] | None = None
        if cfg.resume:
            skip = set()
            for traj in iter_trajectories(traj_dir):
                builder.add(traj)
                skip.add((traj.task_id, traj.sample_index))
            if skip:
                logger.info("resume: %d rollouts already done for %s", len(skip), mut.model_id)

        runner = ParallelRunner(
            concurrency=cfg.concurrency or suite.manifest.concurrency,
            config=rollout_config,
            model=self.model_clients.get(mut.model_id) or mut.model,
            agent=agent_cfg,
            sandbox=suite.manifest.sandbox,
        )
        sink = DirectorySink(traj_dir) if cfg.save_trajectories else None
        t0 = time.monotonic()
        stream = runner.stream(
            suite.tasks,
            n_per_task=cfg.n_per_task or suite.manifest.n_per_task,
            setup_checks=list(suite.manifest.scoring.setup_checks),
            skip=skip,
            progress=cfg.progress,
        )
        try:
            async for traj in stream:
                path = None
                if sink is not None:
                    await sink.write(traj)
                    path = str(sink.path_for(traj))
                builder.add(traj, path)
                if cfg.fail_fast and not traj.is_trainable:
                    logger.error(
                        "fail_fast: %s infra failure on %s: %s",
                        mut.model_id,
                        traj.task_id,
                        traj.error,
                    )
                    break
        finally:
            await stream.aclose()
        public: dict[str, Any] = mut.model.model_dump(mode="json")
        report = builder.build(public, wall_clock_s=time.monotonic() - t0)
        (mdir / "model_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report
