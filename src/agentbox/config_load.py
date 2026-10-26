"""Project configuration (``agentbox.yaml`` / ``.toml`` / ``.json``) — GH #80.

Example::

    default_model: local
    models:
      local:
        model: Qwen/Qwen3-1.7B
        base_url: http://localhost:8000/v1
        api_key: ${VLLM_API_KEY:-EMPTY}
      judge:
        model: gpt-4.1-mini
        api_key: ${OPENAI_API_KEY:?set OPENAI_API_KEY}
    sandbox: {image: python:3.12-slim-bookworm, limits: {memory_mb: 1024}}
    agent: {max_steps: 30, tools: extended}
    rewards: [{name: verifier}, {name: step_penalty, params: {per_step: 0.01}}]

Interpolation: ``${VAR}``, ``${VAR:-default}``, ``${VAR:?error message}``.
Legacy ``teacher`` / ``students`` sections keep working (students become
model profiles).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentbox.config import AgentConfig, ModelConfig, SandboxConfig
from agentbox.errors import AgentBoxError
from agentbox.rewards import RewardSpec

CONFIG_NAMES = ("agentbox.yaml", "agentbox.yml", "agentbox.toml", "agentbox.json")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|:\?)([^}]*))?\}")


class ConfigError(AgentBoxError):
    """Invalid or unreadable project configuration."""


class EndpointConfig(ModelConfig):
    """A named model endpoint (all :class:`ModelConfig` fields + ``id``)."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    timeout_s: float = 180.0

    def to_model_config(self) -> ModelConfig:
        return ModelConfig.model_validate(self.model_dump(exclude={"id"}))


class GenerateSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = 8
    docker_concurrency: int = 8
    llm_judge: bool = True
    min_score: float = 0.65
    use_dspy: bool = False
    target: int = 20
    max_tokens: int = 8192


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: str | None = None
    models: dict[str, EndpointConfig] = Field(default_factory=dict)
    sandbox: SandboxConfig | None = None
    agent: AgentConfig | None = None
    rewards: list[RewardSpec] | None = None
    teacher: EndpointConfig | None = None
    students: list[EndpointConfig] = Field(default_factory=list)
    generate: GenerateSection = Field(default_factory=GenerateSection)
    extra: dict[str, Any] = Field(default_factory=dict)
    path: Path | None = None

    def model(self, name: str | None = None) -> ModelConfig:
        """Resolve a model profile (``None`` -> ``default_model``)."""
        name = name or self.default_model
        if name is None:
            raise ConfigError("no model profile given and no default_model configured")
        profiles = self.all_models()
        if name not in profiles:
            raise ConfigError(f"unknown model profile {name!r}; known: {sorted(profiles)}")
        return profiles[name].to_model_config()

    def all_models(self) -> dict[str, EndpointConfig]:
        out = dict(self.models)
        for s in self.students:
            key = s.id or s.model.replace("/", "-").replace(":", "-")
            out.setdefault(key, s)
        if self.teacher is not None:
            out.setdefault("teacher", self.teacher)
        return out


def interpolate(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name, op, arg = m.group(1), m.group(2), m.group(3)
        val = os.environ.get(name)
        if val:
            return val
        if op == ":-":
            return arg or ""
        if op == ":?":
            raise ConfigError(arg or f"environment variable {name} is required")
        if val is None:
            raise ConfigError(f"environment variable {name} is not set (use ${{{name}:-default}})")
        return val

    return _VAR.sub(repl, value)


def _walk_interp(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_interp(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_interp(v) for v in obj]
    if isinstance(obj, str):
        return interpolate(obj)
    return obj


def find_config_path(start: Path | None = None) -> Path | None:
    env = os.environ.get("AGENTBOX_CONFIG")
    if env:
        p = Path(env)
        if not p.exists():
            raise ConfigError(f"AGENTBOX_CONFIG points to a missing file: {p}")
        return p
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        for name in CONFIG_NAMES:
            p = d / name
            if p.exists():
                return p
    return None


def _read(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text) or {}
        elif path.suffix == ".toml":
            import tomllib

            data = tomllib.loads(text)
        else:
            data = json.loads(text)
    except Exception as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_project_config(path: Path | None = None, *, required: bool = False) -> ProjectConfig:
    """Load the project config; defaults when no file is found (unless ``required``)."""
    p = path or find_config_path()
    if p is None:
        if required:
            raise ConfigError("no agentbox.yaml found (run `agentbox init`)")
        return ProjectConfig()
    data = _walk_interp(_read(p))
    if "models" in data and isinstance(data["models"], list):  # legacy list form
        data["students"] = data.pop("models")
    known = set(ProjectConfig.model_fields) - {"extra", "path"}
    extra = {k: v for k, v in data.items() if k not in known}
    try:
        cfg = ProjectConfig.model_validate({k: v for k, v in data.items() if k in known})
    except ValidationError as exc:
        raise ConfigError(f"invalid {p}:\n{exc}") from exc
    cfg.extra = extra
    cfg.path = p
    return cfg
