# Baked AgentBox sandbox: Python + pytest + common Unix tools (no per-rollout installs).
ARG BASE=python:3.12-slim-bookworm
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl coreutils findutils grep sed gawk git patch procps \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pytest

WORKDIR /workspace
CMD ["sleep", "infinity"]
