"""SkyRL's training entrypoint with the AgentBox environment registered.

SkyRL builds its environments inside the training process, a Ray task, so the
``agentbox`` environment has to be registered there and not only in the
launcher::

    python examples/skyrl/main_agentbox.py environment.env_class=agentbox <SkyRL overrides>

Tested with skyrl 0.4.0 (``skyrl.train``).
"""

from __future__ import annotations

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from agentbox.integrations.skyrl import register


@ray.remote(num_cpus=1)
def agentbox_entrypoint(cfg: SkyRLTrainConfig) -> None:
    register()  # the registry is per process: this is where environments are made
    BasePPOExp(cfg).run()


def main() -> None:
    register()
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(agentbox_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
