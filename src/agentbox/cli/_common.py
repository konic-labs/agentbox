"""Shared CLI plumbing: global state, output, model resolution, exit codes."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import typer

from agentbox.config import ModelConfig
from agentbox.config_load import ConfigError, ProjectConfig, load_project_config

# Exit codes (GH #78)
EXIT_OK = 0
EXIT_FAILURES = 1  # task failures / thresholds not met
EXIT_USAGE = 2  # bad flags, config errors
EXIT_INFRA = 3  # docker / model endpoint unavailable


@dataclass
class State:
    config_path: Path | None = None
    profile: str | None = None
    json: bool = False
    verbose: int = 0
    quiet: bool = False
    _project: ProjectConfig | None = field(default=None, repr=False)

    @property
    def project(self) -> ProjectConfig:
        if self._project is None:
            try:
                self._project = load_project_config(self.config_path)
            except ConfigError as exc:
                fail(str(exc), code=EXIT_USAGE)
        return self._project


def state(ctx: typer.Context) -> State:
    obj = ctx.find_root().obj
    if not isinstance(obj, State):
        obj = State()
        ctx.find_root().obj = obj
    return obj


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(verbose: int, quiet: bool, fmt: str) -> None:
    level = logging.WARNING
    if quiet:
        level = logging.ERROR
    elif verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "docker", "openai"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def fail(message: str, *, code: int = EXIT_USAGE) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def emit(st: State, data: Any, human: Callable[[], object] | None = None) -> None:
    """JSON with ``--json``, otherwise the human renderer (or pretty JSON)."""
    if st.json or human is None:
        typer.echo(json.dumps(data, indent=None if st.json else 2, default=str))
        return
    human()


def table(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], *, title: str | None = None
) -> None:
    from rich.console import Console
    from rich.table import Table

    t = Table(title=title, show_lines=False, header_style="bold")
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*("" if v is None else str(v) for v in r))
    Console().print(t)


def resolve_model(
    st: State,
    *,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ModelConfig:
    """Flags win over profiles; ``--profile`` / global ``--profile`` / default_model."""
    profile = profile or st.profile
    if model is None:
        try:
            base = st.project.model(profile)
        except ConfigError as exc:
            fail(f"{exc} (pass --model or --profile)")
    else:
        base = ModelConfig(model=model)
        if profile:
            try:
                base = st.project.model(profile).model_copy(update={"model": model})
            except ConfigError as exc:
                fail(str(exc))
    update: dict[str, Any] = {k: v for k, v in (overrides or {}).items() if v is not None}
    if base_url is not None:
        update["base_url"] = base_url
    if api_key_env is not None:
        update["api_key_env"] = api_key_env
        update["api_key"] = os.environ.get(api_key_env)
    return base.model_copy(update=update) if update else base


def parse_kv(values: Sequence[str] | None, *, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            fail(f"invalid {what} {item!r}; expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"
