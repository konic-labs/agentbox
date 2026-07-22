"""Workspace path jail helpers."""

from __future__ import annotations

from pathlib import PurePosixPath

from agentbox.errors import PathEscapeError


def resolve_workspace_path(workspace_dir: str, user_path: str) -> str:
    """Resolve ``user_path`` under ``workspace_dir`` and reject escapes.

    Returns a normalized absolute posix path still under the workspace.
    """
    workspace = PurePosixPath(workspace_dir)
    if not workspace.is_absolute():
        raise PathEscapeError(f"workspace_dir must be absolute, got {workspace_dir!r}")

    raw = PurePosixPath(user_path)
    if raw.is_absolute():
        candidate = PurePosixPath(str(raw))
    else:
        candidate = workspace / raw

    # Normalize .. and . without resolving against the host filesystem.
    parts: list[str] = []
    for part in candidate.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts or parts == ["/"]:
                raise PathEscapeError(f"path escapes workspace: {user_path!r}")
            if parts[-1] != "/":
                parts.pop()
            continue
        parts.append(part)

    if not parts:
        normalized = PurePosixPath("/")
    elif parts[0] == "/":
        normalized = PurePosixPath(*parts)
    else:
        normalized = PurePosixPath("/", *parts)

    workspace_str = str(workspace)
    normalized_str = str(normalized)
    if normalized_str != workspace_str and not normalized_str.startswith(
        workspace_str.rstrip("/") + "/"
    ):
        raise PathEscapeError(
            f"path {user_path!r} resolves outside workspace {workspace_dir!r}"
        )
    return normalized_str


def relative_to_workspace(workspace_dir: str, absolute_path: str) -> str:
    """Return path relative to workspace for display."""
    ws = PurePosixPath(workspace_dir)
    ab = PurePosixPath(absolute_path)
    try:
        return str(ab.relative_to(ws))
    except ValueError as exc:
        raise PathEscapeError(
            f"path {absolute_path!r} is not under {workspace_dir!r}"
        ) from exc
