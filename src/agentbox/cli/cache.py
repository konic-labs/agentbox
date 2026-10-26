"""`agentbox cache …` (setup snapshots) and `agentbox image …` (baked images)."""

from __future__ import annotations

import asyncio
import tempfile
import time
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional

import typer

from agentbox.cli._common import emit, fail, state, table

cache_app = typer.Typer(help="Setup-snapshot image cache.", no_args_is_help=True)
image_app = typer.Typer(help="Build and list sandbox images.", no_args_is_help=True)


@cache_app.command("ls")
def cache_ls(ctx: typer.Context) -> None:
    """List cached setup/pytest images."""
    from agentbox.sandbox.backend import get_backend

    now = time.time()
    rows = [
        {
            "tag": (img.get("RepoTags") or ["<none>"])[0],
            "size_mb": round((img.get("Size") or 0) / 1e6, 1),
            "age_h": round((now - float(img.get("Created") or now)) / 3600, 1),
        }
        for img in get_backend().list_cache_images_sync()
    ]
    emit(state(ctx), rows, lambda: table(["tag", "size MB", "age h"],
                                         [[r["tag"], r["size_mb"], r["age_h"]] for r in rows]))


@cache_app.command("prune")
def cache_prune(
    ctx: typer.Context,
    older_than_h: Annotated[Optional[float], typer.Option("--older-than-h")] = None,
) -> None:
    """Remove cached images (all, or older than N hours); in-use images are kept."""
    from agentbox.sandbox.backend import get_backend

    backend = get_backend()
    now = time.time()
    removed = []
    for img in backend.list_cache_images_sync():
        if older_than_h is not None and now - float(img.get("Created") or now) < older_than_h * 3600:
            continue
        for tag in img.get("RepoTags") or []:
            try:
                backend.client.api.remove_image(tag)
                removed.append(tag)
            except Exception:
                pass
    emit(state(ctx), {"removed": removed}, lambda: typer.echo(f"removed {len(removed)} image(s)"))


@cache_app.command("warm")
def cache_warm(
    ctx: typer.Context,
    tasks: Annotated[list[Path], typer.Argument(help="Tasks whose setup snapshots to build")],
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 4,
) -> None:
    """Pre-build setup snapshots before a training run (no agent, no model)."""
    from agentbox.config import SandboxConfig
    from agentbox.runner.provision import manager_for
    from agentbox.tasks.loader import iter_tasks

    st = state(ctx)
    base = st.project.sandbox or SandboxConfig()
    todo = [t for t in iter_tasks(tasks) if t.setup_commands]

    async def warm_one(sem: asyncio.Semaphore, t: object) -> tuple[str, str]:
        async with sem:
            mgr = manager_for(t, base, None)  # type: ignore[arg-type]
            cfg = mgr.config
            try:
                image = await mgr.resolve_image()
                tag = await mgr.backend.derive_image(
                    image,
                    steps=list(t.setup_commands),  # type: ignore[attr-defined]
                    files=t.seed_files(),  # type: ignore[attr-defined]
                    workdir=cfg.workspace_dir,
                    env=cfg.env,
                    network=cfg.prepare_network or not cfg.limits.network_disabled,
                    shell=mgr.shell_for(image),
                    key_extra={"purpose": "setup"},
                    max_images=cfg.cache_max_images,
                )
                return t.task_id, tag  # type: ignore[attr-defined]
            except Exception as exc:
                return t.task_id, f"FAILED: {exc}"  # type: ignore[attr-defined]

    async def go() -> list[tuple[str, str]]:
        sem = asyncio.Semaphore(concurrency)
        return await asyncio.gather(*(warm_one(sem, t) for t in todo))

    results = asyncio.run(go())
    emit(st, dict(results), lambda: table(["task", "image"], results))


@image_app.command("build")
def image_build(
    ctx: typer.Context,
    tag: Annotated[str, typer.Option("--tag", "-t")] = "agentbox/sandbox:latest",
    base: Annotated[str, typer.Option("--base", help="Base image")] = "python:3.12-slim-bookworm",
    dockerfile: Annotated[Optional[Path], typer.Option("--dockerfile", help="Custom Dockerfile")] = None,
) -> None:
    """Build the baked sandbox image (Dockerfile ships with the package)."""
    from agentbox.sandbox.backend import get_backend

    st = state(ctx)
    content = (
        dockerfile.read_text(encoding="utf-8")
        if dockerfile
        else resources.files("agentbox.data").joinpath("sandbox.Dockerfile").read_text(encoding="utf-8")
    )
    client = get_backend().client
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "Dockerfile").write_text(content, encoding="utf-8")
        try:
            image, logs = client.images.build(path=tmp, tag=tag, rm=True, buildargs={"BASE": base})
        except Exception as exc:
            fail(f"build failed: {exc}")
        if not st.json:
            for line in logs:
                if isinstance(line, dict) and "stream" in line:
                    typer.echo(line["stream"], nl=False)
    emit(st, {"tag": tag, "id": image.short_id},
         lambda: typer.secho(f"built {tag} ({image.short_id})", fg=typer.colors.GREEN))


@image_app.command("ls")
def image_ls(ctx: typer.Context) -> None:
    """List local images usable as sandboxes (presets + agentbox images)."""
    from agentbox.sandbox.backend import get_backend
    from agentbox.sandbox.images import PRESETS

    backend = get_backend()
    rows = [{"preset": k, "image": v, "present": backend.image_exists_sync(v)} for k, v in PRESETS.items()]
    emit(state(ctx), rows, lambda: table(["preset", "image", "present"],
                                         [[r["preset"], r["image"], r["present"]] for r in rows]))
