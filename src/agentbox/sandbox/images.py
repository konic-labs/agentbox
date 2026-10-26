"""Sandbox image presets."""

from __future__ import annotations

from typing import Any

from agentbox.config import SandboxConfig

PRESETS: dict[str, str] = {
    "python": "python:3.12-slim-bookworm",
    "python3.11": "python:3.11-slim-bookworm",
    "python3.13": "python:3.13-slim-bookworm",
    "node": "node:22-bookworm-slim",
    "go": "golang:1.24-bookworm",
    "rust": "rust:1-slim-bookworm",
    "baked": "agentbox/sandbox:latest",
}

ImagePreset = str


def register_preset(name: str, image: str) -> None:
    """Register (or override) a named image preset."""
    PRESETS[name] = image


def sandbox_config_for_preset(preset: ImagePreset = "python", **overrides: Any) -> SandboxConfig:
    """Build a SandboxConfig for a language/runtime preset."""
    try:
        image = PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}") from exc
    data: dict[str, Any] = {"image": image, **overrides}
    if not preset.startswith("python") and preset != "baked":
        data.setdefault("ensure_pytest", False)  # non-Python images have no pytest
    return SandboxConfig(**data)
