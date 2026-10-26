"""Container snapshot helpers (commit a sandbox to an image)."""

from __future__ import annotations

import logging
import time
from typing import Any

from agentbox.errors import SandboxError
from agentbox.sandbox.backend import DockerBackend, get_backend
from agentbox.sandbox.types import Sandbox

logger = logging.getLogger("agentbox.snapshot")


async def commit_sandbox(
    sandbox: Sandbox,
    *,
    repository: str = "agentbox/snapshot",
    tag: str | None = None,
    docker_client: Any | None = None,
) -> str:
    """Commit the sandbox container filesystem to a new image; returns ``repo:tag``."""
    backend = DockerBackend(client=docker_client) if docker_client is not None else get_backend()
    safe_task = "".join(c if c.isalnum() or c in "-_." else "-" for c in sandbox.task_id)
    tag = tag or f"{safe_task[:40] or 'task'}-{int(time.time())}"
    try:
        full = await backend.run_ctl(backend.commit_sync, sandbox.id, repository=repository, tag=tag)
    except Exception as exc:
        raise SandboxError(f"Failed to commit sandbox: {exc}") from exc
    logger.info("snapshot.committed sandbox=%s image=%s", sandbox.id[:12], full)
    return full
