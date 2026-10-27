#!/usr/bin/env bash
# GRPO with verl + AgentBox. Adjust the verl flags to your version and hardware.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
python "$HERE/prepare_data.py" --tasks "${TASKS:-examples/tasks}" --out data/agentbox
export AGENTBOX_VERL_CONFIG="$HERE/agentbox-verl.yaml"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/agentbox/train.parquet \
  data.val_files=data/agentbox/val.parquet \
  data.return_raw_chat=True \
  data.train_batch_size=32 \
  actor_rollout_ref.model.path="${MODEL:-Qwen/Qwen3-1.7B}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$HERE/agent_loops.yaml" \
  trainer.total_epochs=1 \
  "$@"
