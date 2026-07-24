#!/usr/bin/env bash
# Hermetic CI smoke: unit tests + doctor (no external LLM required).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/unit -q --tb=line
agentbox version
agentbox doctor || true
# Mock bench pipeline (no LLM)
if [[ -d examples/benchmarks/coding-mini ]]; then
  TMP=$(mktemp -d)
  agentbox bench create "$TMP/suite" \
    --from-tasks examples/benchmarks/coding-mini/tasks \
    --suite-id coding-mini-smoke \
    --name "coding mini smoke" \
    --version 0.0.1 || true
  if [[ -d "$TMP/suite" ]]; then
    agentbox bench run "$TMP/suite" --mock --out "$TMP/out" || true
  fi
  rm -rf "$TMP"
fi
echo "ci_smoke ok"
