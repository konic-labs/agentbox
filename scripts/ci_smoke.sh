#!/usr/bin/env bash
# Hermetic CLI smoke test: no model endpoint, every command must succeed.
#   scripts/ci_smoke.sh            CLI, task and suite checks (no Docker)
#   scripts/ci_smoke.sh --docker   + rollouts, trajectory tooling and a benchmark run
#                                  in real sandboxes with the mock model
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER=0
[[ "${1:-}" == "--docker" ]] && DOCKER=1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

SUITE=examples/benchmarks/coding-mini
TASKS=$SUITE/tasks
step() { printf '\n== %s\n' "$*"; }

step "cli"
agentbox version
agentbox --json integrations >"$TMP/integrations.json"
agentbox config show >/dev/null

step "tasks"
agentbox task validate "$TASKS"
agentbox task ls "$TASKS"
agentbox task schema >"$TMP/task.schema.json"
agentbox task new "$TMP/new-task" --id smoke_new_001
agentbox task validate "$TMP/new-task"

step "suite"
agentbox bench validate "$SUITE"

if ((DOCKER)); then
  step "doctor"
  agentbox doctor --no-probe

  step "rollouts (mock model)"
  agentbox run "$TASKS" --mock -n 2 --out "$TMP/run" --no-progress
  agentbox traj validate "$TMP/run"
  agentbox traj stats "$TMP/run"
  agentbox traj export "$TMP/run" --format chat --out "$TMP/chat.jsonl"
  test -s "$TMP/chat.jsonl"

  step "resume skips completed rollouts"
  agentbox --json run "$TASKS" --mock -n 2 --out "$TMP/run" --resume --no-progress >"$TMP/resume.json"
  python - "$TMP/resume.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary["skipped"] == 4 and summary["n"] == 0, summary
PY

  step "benchmark (mock model)"
  agentbox bench run "$SUITE" --mock --strict --out "$TMP/bench" --no-progress
  agentbox bench show "$TMP/bench/report.json"

  step "no leaked sandboxes"
  test -z "$(docker ps -aq --filter label=agentbox.role=sandbox)"
fi

printf '\nci_smoke ok\n'
