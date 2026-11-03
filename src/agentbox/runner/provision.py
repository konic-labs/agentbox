"""Sandbox provisioning shared by Rollout, AgentBoxEnv and trainer integrations.

host slot (``host_max_sandboxes``) -> create (or start from the setup snapshot)
-> seed -> setup checks, with every failure attributed to a :class:`FailureKind`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from agentbox.config import SandboxConfig
from agentbox.sandbox.manager import SandboxManager
from agentbox.sandbox.slots import SlotLease, host_slots
from agentbox.sandbox.types import Sandbox
from agentbox.tasks.schema import Task
from agentbox.tasks.seeder import TaskSeeder
from agentbox.types import FailureKind

logger = logging.getLogger("agentbox.provision")


class ProvisionError(Exception):
    def __init__(self, kind: FailureKind, message: str, *, extra: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.extra = extra or {}


@dataclass
class Provisioned:
    sandbox: Sandbox
    manager: SandboxManager
    config: SandboxConfig
    create_s: float = 0.0
    seed_s: float | None = None
    wait_s: float | None = None  # time spent waiting for a host slot
    lease: SlotLease | None = None


def effective_sandbox(base: SandboxConfig, task: Task) -> SandboxConfig:
    """Merge a task's environment overrides over the run's sandbox config."""
    env = task.environment
    if env is None:
        return base
    update: dict[str, Any] = {}
    if env.image:
        update["image"] = env.image
    if env.env:
        update["env"] = {**base.env, **env.env}
    if env.working_dir:
        update["working_dir"] = env.working_dir
        update["workspace_dir"] = env.working_dir
    if env.ensure_pytest is not None:
        update["ensure_pytest"] = env.ensure_pytest
    limits: dict[str, Any] = {}
    if env.network is not None:
        limits["network_disabled"] = not env.network
    if env.cpu_count is not None:
        limits["cpu_count"] = env.cpu_count
    if env.memory_mb is not None:
        limits["memory_mb"] = env.memory_mb
    if limits:
        update["limits"] = base.limits.model_copy(update=limits)
    return base.model_copy(update=update) if update else base


def manager_for(task: Task, base: SandboxConfig, manager: SandboxManager | None) -> SandboxManager:
    cfg = effective_sandbox(base, task)
    if manager is not None and cfg is manager.config:
        return manager
    return SandboxManager(cfg, backend=manager.backend if manager else None)


async def provision(
    task: Task,
    *,
    manager: SandboxManager,
    run_id: str,
    rollout_id: str,
    group_id: str | None = None,
    setup_checks: list[Any] | None = None,
) -> Provisioned:
    """Create a seeded sandbox for ``task`` or raise :class:`ProvisionError`."""
    cfg = manager.config
    lease: SlotLease | None = None
    wait_s: float | None = None
    if cfg.host_max_sandboxes:
        t_wait = time.monotonic()
        try:
            lease = await host_slots(cfg.host_max_sandboxes).acquire(cfg.host_slot_timeout_s)
        except TimeoutError as exc:
            raise ProvisionError(
                FailureKind.SANDBOX,
                f"no host slot within {cfg.host_slot_timeout_s:.0f}s: {exc} "
                f"(host_max_sandboxes={cfg.host_max_sandboxes})",
            ) from exc
        wait_s = time.monotonic() - t_wait
    try:
        sandbox, create_s, seed_s, cached_image = await _create(
            task, manager, run_id=run_id, rollout_id=rollout_id, group_id=group_id
        )
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    prov = Provisioned(
        sandbox=sandbox,
        manager=manager,
        config=cfg,
        create_s=create_s,
        seed_s=seed_s,
        wait_s=wait_s,
        lease=lease,
    )
    try:
        if cached_image is None:
            seed = await TaskSeeder(manager).seed(sandbox, task)
            prov.seed_s = seed.duration_s
            if not seed.ok:
                raise ProvisionError(FailureKind.SEED, seed.error or "task seed failed")
        if setup_checks:
            from agentbox.benchmark.setup_check import SetupChecker

            check = await SetupChecker(manager).run(sandbox, setup_checks)
            if not check.ok:
                raise ProvisionError(
                    FailureKind.SETUP_CHECK,
                    check.error or "setup check failed",
                    extra={"failed_check": check.failed_check},
                )
    except BaseException:
        await release(prov, keep=False)
        raise
    return prov


async def _create(
    task: Task,
    manager: SandboxManager,
    *,
    run_id: str,
    rollout_id: str,
    group_id: str | None,
) -> tuple[Sandbox, float, float | None, str | None]:
    """Build the setup snapshot if needed, then create the sandbox."""
    cfg = manager.config
    t0 = time.monotonic()
    cached_image: str | None = None
    seed_s: float | None = None
    if cfg.cache_setup and task.setup_commands:
        try:
            base_image = await manager.resolve_image()
            cached_image = await manager.backend.derive_image(
                base_image,
                steps=list(task.setup_commands),
                files=task.seed_files(),
                workdir=cfg.workspace_dir,
                env=cfg.env,
                network=cfg.prepare_network or not cfg.limits.network_disabled,
                shell=manager.shell_for(base_image),
                step_timeout_s=300.0,
                key_extra={"purpose": "setup"},
                max_images=cfg.cache_max_images,
            )
        except Exception as exc:
            raise ProvisionError(FailureKind.SEED, f"setup failed: {exc}") from exc
        seed_s = time.monotonic() - t0
    t_create = time.monotonic()
    try:
        sandbox = await manager.create(
            task_id=task.task_id,
            run_id=run_id,
            rollout_id=rollout_id,
            labels={"agentbox.group_id": group_id} if group_id else None,
            image=cached_image,
            resolve=cached_image is None,
        )
    except Exception as exc:
        raise ProvisionError(FailureKind.SANDBOX, str(exc)) from exc
    return sandbox, time.monotonic() - t_create, seed_s, cached_image


async def release(prov: Provisioned, *, keep: bool) -> None:
    """Destroy (or keep) the sandbox and free its host slot."""
    try:
        if keep:
            prov.manager.keep(prov.sandbox)
            logger.warning(
                "keeping sandbox %s; inspect with: docker exec -it %s sh",
                prov.sandbox.name,
                prov.sandbox.name,
            )
            return
        try:
            await prov.manager.destroy(prov.sandbox)
        except Exception as exc:  # pragma: no cover - teardown must not mask results
            logger.warning("sandbox destroy failed for %s: %s", prov.sandbox.name, exc)
    finally:
        if prov.lease is not None:  # kept sandboxes leave the budget too
            prov.lease.release()
