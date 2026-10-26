"""Prune AgentBox containers safely.

By default only *orphans* are removed: containers whose owning process (recorded
in the ``agentbox.owner`` label as ``host:pid``) is gone on this host, or that are
older than ``older_than_s``. Sandboxes of other running jobs are left alone
(GH #22).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Any, Literal

from agentbox.sandbox.backend import DockerBackend, get_backend

logger = logging.getLogger("agentbox.prune")

PruneScope = Literal["orphans", "all"]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_orphan(labels: dict[str, str], *, older_than_s: float | None = None) -> bool:
    """Decide whether a labelled container no longer has a live owner."""
    created = labels.get("agentbox.created")
    if older_than_s is not None and created is not None:
        try:
            if time.time() - float(created) >= older_than_s:
                return True
        except ValueError:
            pass
    owner = labels.get("agentbox.owner")
    if not owner:
        # Pre-0.3 containers carry no owner; only an age threshold can claim them.
        return False
    host, _, pid_s = owner.rpartition(":")
    if host != socket.gethostname():
        return False
    try:
        return not _pid_alive(int(pid_s))
    except ValueError:
        return False


def select_containers(
    containers: list[dict[str, Any]],
    *,
    scope: PruneScope = "orphans",
    run_id: str | None = None,
    older_than_s: float | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for c in containers:
        labels = c.get("Labels") or {}
        if labels.get("agentbox.cache") == "1" and "agentbox.owner" not in labels:
            continue
        if run_id is not None and labels.get("agentbox.run_id") != run_id:
            continue
        if scope == "all" or run_id is not None or is_orphan(labels, older_than_s=older_than_s):
            selected.append(c)
    return selected


def prune_agentbox_containers(
    docker_client: Any | None = None,
    *,
    force: bool = True,  # noqa: ARG001 - kept for API compatibility
    scope: PruneScope = "orphans",
    run_id: str | None = None,
    older_than_s: float | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Remove AgentBox containers; returns the ids removed (or that would be)."""
    backend = DockerBackend(client=docker_client) if docker_client is not None else get_backend()
    containers = backend.list_containers_sync()
    chosen = select_containers(containers, scope=scope, run_id=run_id, older_than_s=older_than_s)
    removed: list[str] = []
    for c in chosen:
        cid = str(c["Id"])
        if not dry_run:
            backend.remove_container_sync(cid)
            logger.info("pruned container %s", cid[:12])
        removed.append(cid)
    return removed


async def aprune_agentbox_containers(
    docker_client: Any | None = None,
    **kwargs: Any,
) -> list[str]:
    return await asyncio.to_thread(prune_agentbox_containers, docker_client, **kwargs)
