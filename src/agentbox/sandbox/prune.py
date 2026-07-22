"""Prune orphaned AgentBox containers."""

from __future__ import annotations

import logging
from typing import Any

from agentbox.sandbox import docker_backend

logger = logging.getLogger("agentbox.prune")


def prune_agentbox_containers(
    docker_client: Any | None = None,
    *,
    force: bool = True,
) -> list[str]:
    """Remove containers labeled agentbox=1. Returns removed container ids."""
    client = docker_backend.get_client(docker_client)
    removed: list[str] = []
    containers = client.containers.list(all=True, filters={"label": "agentbox=1"})
    for container in containers:
        cid = container.id
        try:
            docker_backend.destroy_container(container, force=force)
            removed.append(cid)
            logger.info("pruned container %s", cid[:12])
        except Exception as exc:
            logger.warning("failed to prune %s: %s", cid[:12], exc)
    return removed


async def aprune_agentbox_containers(
    docker_client: Any | None = None,
    *,
    force: bool = True,
) -> list[str]:
    import asyncio

    return await asyncio.to_thread(
        prune_agentbox_containers, docker_client, force=force
    )
