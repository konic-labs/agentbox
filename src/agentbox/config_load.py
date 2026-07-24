"""Load optional agentbox.yaml / agentbox.toml project config."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EndpointConfig(BaseModel):
    id: str | None = None
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = 4096
    timeout_s: float = 180.0
    extra_body: dict[str, Any] = Field(default_factory=dict)


class GenerateSection(BaseModel):
    concurrency: int = 8
    docker_concurrency: int = 8
    llm_judge: bool = True
    min_score: float = 0.65
    use_dspy: bool = False
    target: int = 20
    max_tokens: int = 8192


class ProjectConfig(BaseModel):
    teacher: EndpointConfig | None = None
    students: list[EndpointConfig] = Field(default_factory=list)
    generate: GenerateSection = Field(default_factory=GenerateSection)
    extra: dict[str, Any] = Field(default_factory=dict)


def _interp(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), "")

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, value)


def _walk_interp(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_interp(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_interp(v) for v in obj]
    return _interp(obj)


def find_config_path(start: Path | None = None) -> Path | None:
    env = os.environ.get("AGENTBOX_CONFIG")
    if env:
        p = Path(env)
        return p if p.exists() else None
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        for name in ("agentbox.yaml", "agentbox.yml", "agentbox.toml", "agentbox.json"):
            p = d / name
            if p.exists():
                return p
        if d.parent == d:
            break
    return None


def load_project_config(path: Path | None = None) -> ProjectConfig:
    """Load project config; returns defaults if no file found."""
    p = path or find_config_path()
    if p is None:
        return ProjectConfig()
    text = p.read_text(encoding="utf-8")
    data: dict[str, Any]
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML required for agentbox.yaml (pip install pyyaml)"
            ) from exc
        data = yaml.safe_load(text) or {}
    elif p.suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(text)
    else:
        data = json.loads(text)
    data = _walk_interp(data)
    # students may be under students or models
    if "models" in data and "students" not in data:
        data["students"] = data.pop("models")
    known = {"teacher", "students", "generate"}
    extra = {k: v for k, v in data.items() if k not in known}
    cfg = ProjectConfig.model_validate({k: data[k] for k in known if k in data})
    cfg.extra = extra
    return cfg
