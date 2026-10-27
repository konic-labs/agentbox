#!/usr/bin/env bash
# OpenRLHF agent RL with AgentBox episodes (token-in/token-out on OpenRLHF's vLLM engine).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
python "$HERE/prepare_data.py" --tasks "${TASKS:-examples/tasks}" --out data/openrlhf.jsonl
export AGENTBOX_TASKS="${TASKS:-examples/tasks}"
AGENT_FUNC=$(python -c "import agentbox.integrations.openrlhf as m; print(m.__file__)")

python -m openrlhf.cli.train_ppo_ray \
  --train.agent_func_path "$AGENT_FUNC" \
  --data.prompt_dataset data/openrlhf.jsonl \
  --data.input_key prompt --data.label_key label \
  "$@"   # model, vLLM engines, GRPO/advantage flags for your OpenRLHF version
