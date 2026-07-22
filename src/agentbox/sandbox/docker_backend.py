"""Low-level synchronous Docker helpers (run via asyncio.to_thread)."""

from __future__ import annotations

import io
import tarfile
import time
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container

from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.errors import DockerNotAvailableError, ImageNotFoundError, SandboxError


def get_client(docker_client: Any | None = None) -> docker.DockerClient:
    if docker_client is not None:
        return docker_client
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        raise DockerNotAvailableError(
            "Docker daemon is not available. Is Docker running?"
        ) from exc


def ensure_image(client: docker.DockerClient, image: str, *, auto_pull: bool) -> None:
    try:
        client.images.get(image)
        return
    except ImageNotFound:
        pass
    if not auto_pull:
        raise ImageNotFoundError(f"Image not found: {image}")
    try:
        client.images.pull(image)
    except DockerException as exc:
        raise ImageNotFoundError(f"Failed to pull image {image}: {exc}") from exc


def create_and_start(
    client: docker.DockerClient,
    *,
    name: str,
    config: SandboxConfig,
    labels: dict[str, str],
) -> Container:
    limits = config.limits
    host_config_kwargs: dict[str, Any] = {}
    if limits.memory_mb is not None:
        host_config_kwargs["mem_limit"] = f"{limits.memory_mb}m"
        swap = limits.memswap_mb if limits.memswap_mb is not None else limits.memory_mb
        host_config_kwargs["memswap_limit"] = f"{swap}m"
    if limits.cpu_count is not None:
        host_config_kwargs["nano_cpus"] = int(limits.cpu_count * 1e9)
    if limits.pids_limit is not None:
        host_config_kwargs["pids_limit"] = limits.pids_limit
    if limits.network_disabled:
        host_config_kwargs["network_mode"] = "none"

    try:
        container = client.containers.create(
            image=config.image,
            name=name,
            command=config.command,
            working_dir=config.working_dir,
            environment=config.env or None,
            labels=labels,
            detach=True,
            stdin_open=True,
            tty=False,
            **host_config_kwargs,
        )
        container.start()
        return container
    except APIError as exc:
        raise SandboxError(f"Failed to create/start container: {exc}") from exc


def destroy_container(container: Container, *, force: bool = True) -> None:
    try:
        container.reload()
        if container.status == "running":
            container.stop(timeout=5)
        container.remove(force=force)
    except NotFound:
        return
    except APIError:
        try:
            container.remove(force=True)
        except Exception:
            return


def exec_run(
    container: Container,
    cmd: list[str],
    *,
    workdir: str | None,
    env: dict[str, str] | None,
) -> tuple[int, str, str, float]:
    start = time.monotonic()
    result = container.exec_run(
        cmd,
        workdir=workdir,
        environment=env,
        demux=True,
        stdout=True,
        stderr=True,
    )
    duration = time.monotonic() - start
    exit_code = int(result.exit_code) if result.exit_code is not None else -1
    out_b, err_b = result.output if result.output else (None, None)
    stdout = (out_b or b"").decode("utf-8", errors="replace")
    stderr = (err_b or b"").decode("utf-8", errors="replace")
    return exit_code, stdout, stderr, duration


def put_files(container: Container, files: dict[str, bytes], *, root: str) -> None:
    """Write multiple files under ``root`` using a single put_archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel_path, content in files.items():
            # tar paths must be relative to the archive root
            arcname = rel_path.lstrip("/")
            info = tarfile.TarInfo(name=arcname)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
    buf.seek(0)
    ok = container.put_archive(root, buf.getvalue())
    if not ok:
        raise SandboxError(f"put_archive failed under {root}")


def get_file_bytes(container: Container, path: str) -> bytes:
    try:
        bits, _stat = container.get_archive(path)
    except NotFound as exc:
        raise FileNotFoundError(path) from exc
    except APIError as exc:
        raise SandboxError(f"get_archive failed for {path}: {exc}") from exc

    data = b"".join(bits)
    buf = io.BytesIO(data)
    with tarfile.open(fileobj=buf, mode="r") as tar:
        members = tar.getmembers()
        if not members:
            raise FileNotFoundError(path)
        member = members[0]
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(path)
        return extracted.read()
