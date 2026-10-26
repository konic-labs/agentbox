"""Deprecated location; use :mod:`agentbox.integrations.art`."""

from __future__ import annotations

from typing import Any

from agentbox.integrations.art import to_art_dict, to_art_trajectory
from agentbox.trajectory.schema import Trajectory


def to_art(traj: Trajectory) -> Any:
    return to_art_trajectory(traj)


__all__ = ["to_art", "to_art_dict"]
