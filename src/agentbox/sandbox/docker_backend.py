"""Deprecated module kept for import compatibility; use :mod:`agentbox.sandbox.backend`."""

from __future__ import annotations

from agentbox.sandbox.backend import DockerBackend, build_tar, get_backend, get_client

__all__ = ["DockerBackend", "build_tar", "get_backend", "get_client"]
