FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl coreutils findutils grep sed gawk git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pytest

WORKDIR /workspace
CMD ["sleep", "infinity"]
