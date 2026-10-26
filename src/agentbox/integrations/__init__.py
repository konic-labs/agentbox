"""Optional trainer integrations (GH #7).

The core is trainer-agnostic; each integration is a thin adapter that only
imports its framework when used:

=========  =================================================  ==================
name       entry point                                        install
=========  =================================================  ==================
trl        ``integrations.trl.make_environment_factory``      ``agentbox-rl[trl]``
verl       ``integrations.verl.AgentBoxAgentLoop``            ``verl``
openrlhf   ``integrations.openrlhf`` (``--agent_func_path``)  ``openrlhf``
skyrl      ``integrations.skyrl.AgentBoxTextEnv``             ``skyrl-gym``
art        ``integrations.art.to_art_trajectory``             ``agentbox-rl[art]``
=========  =================================================  ==================
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Integration:
    name: str
    package: str
    entry_point: str
    install: str
    description: str


INTEGRATIONS: tuple[Integration, ...] = (
    Integration(
        "trl",
        "trl",
        "agentbox.integrations.trl:make_environment_factory",
        "pip install 'agentbox-rl[trl]'",
        "GRPOTrainer(environment_factory=...) with real sandboxes as tools",
    ),
    Integration(
        "verl",
        "verl",
        "agentbox.integrations.verl:AgentBoxAgentLoop",
        "pip install verl",
        "verl agent loop (token-in/out) with AgentBox tools and rewards",
    ),
    Integration(
        "openrlhf",
        "openrlhf",
        "agentbox.integrations.openrlhf:AgentExecutor",
        "pip install openrlhf",
        "--train.agent_func_path executor running TITO episodes on the vLLM engine",
    ),
    Integration(
        "skyrl",
        "skyrl-gym",
        "agentbox.integrations.skyrl:AgentBoxTextEnv",
        "pip install skyrl-gym",
        "skyrl_gym BaseTextEnv backed by AgentBox sandboxes",
    ),
    Integration(
        "art",
        "openpipe-art",
        "agentbox.integrations.art:to_art_trajectory",
        "pip install 'agentbox-rl[art]'",
        "ART Trajectory / TrajectoryGroup export",
    ),
)


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def available_integrations() -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    for it in INTEGRATIONS:
        row: dict[str, str | None] = dict(asdict(it))
        row["version"] = installed_version(it.package)
        rows.append(row)
    return rows


__all__ = ["INTEGRATIONS", "Integration", "available_integrations", "installed_version"]
