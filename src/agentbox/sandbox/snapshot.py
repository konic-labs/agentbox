"""Container snapshot helpers (commit image for fast restore)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agentbox.errors import SandboxError
from agentbox.sandbox.types import Sandbox

logger = logging.getLogger("agentbox.snapshot")


async def commit_sandbox(
    sandbox: Sandbox,
    *,
    repository: str = "agentbox/snapshot",
    tag: str | None = None,
    docker_client: Any | None = None,
) -> str:
    """Commit the sandbox container filesystem to a new image.

    Returns the image tag ``repository:tag``.
    """
    from agentbox.sandbox import docker_backend

    client = docker_backend.get_client(docker_client)
    tag = tag or f"{sandbox.task_id[:20]}-{int(time.time())}"
    container = await asyncio.to_thread(client.containers.get, sandbox.id)

    def _commit() -> str:
        image = container.commit(repository=repository, tag=tag)
        # image.tags may be empty for intermediate; construct name
        full = f"{repository}:{tag}"
        logger.info("snapshot.committed sandbox=%s image=%s", sandbox.id[:12], full)
        return full

    try:
        return await asyncio.to_thread(_commit)
    except Exception as exc:
        raise SandboxError(f"Failed to commit sandbox: {exc}") from exc
