"""Sandbox image presets and ensure helpers."""

from __future__ import annotations

from typing import Literal

from agentbox.config import SandboxConfig

ImagePreset = Literal["python", "node", "go", "baked"]

PRESETS: dict[str, str] = {
    "python": "python:3.12-slim-bookworm",
    "node": "node:22-bookworm-slim",
    "go": "golang:1.22-bookworm",
    "baked": "agentbox/sandbox:latest",
}


def sandbox_config_for_preset(
    preset: ImagePreset = "python",
    **overrides: object,
) -> SandboxConfig:
    """Build SandboxConfig for a language/runtime preset."""
    image = PRESETS[preset]
    data: dict = {"image": image, **overrides}
    # Node/Go images won't have pytest; disable ensure_pytest
    if preset != "python" and preset != "baked":
        data.setdefault("ensure_pytest", False)
    return SandboxConfig(**data)  # type: ignore[arg-type]
