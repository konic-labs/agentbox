#!/usr/bin/env bash
# GRPO on several GPUs with verl + AgentBox sandboxes: Qwen3-0.6B (thinking), full fine-tune.
#
# The algorithm settings match run_grpo_single_gpu.sh (8 tasks x 8 samples per step, lr 2e-6,
# 6 assistant turns), so the reward curves compare. Tested with verl 0.9.1 (vLLM 0.24,
# torch 2.11 / CUDA 13, flash-attn 2.8.3 built for sm_120) on 2x RTX PRO 6000 Blackwell
# (96 GB), 48 vCPU, 500 GB RAM.
#
#   python examples/verl/make_coding_tasks.py data/tasks
#   python examples/verl/prepare_data.py --out data/verl
#   STEPS=30 examples/verl/run_grpo_multi_gpu.sh                 # every GPU, one vLLM replica each
#   TP=2 examples/verl/run_grpo_multi_gpu.sh                     # replicas span 2 GPUs
#   AGENTBOX_VERL_CONFIG=examples/verl/agentbox-verl-remote.yaml \
#     examples/verl/run_grpo_multi_gpu.sh                        # sandboxes on an `agentbox serve` host
#
# Several nodes: start Ray first (`ray start --head` here, `ray start --address=<head>:6379`
# on the others), then run this on the head with NNODES and GPUS (per node). Ray workers do not
# inherit this shell's environment there, so AGENTBOX_VERL_CONFIG and AGENTBOX_TASKS travel in
# runtime_env (below); the config variable may hold one-line JSON when the nodes share no
# filesystem. Export AGENTBOX_SERVER_TOKEN before `ray start` on every node instead: a secret
# does not belong on a command line.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$PWD/data/verl}
NNODES=${NNODES:-1}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}           # per node
TP=${TP:-1}                                     # tensor parallel size of each vLLM replica
WORKERS=${WORKERS:-$((2 * GPUS * NNODES))}      # agent-loop workers (AgentBox processes)
STEPS=${STEPS:-30}
export AGENTBOX_VERL_CONFIG=${AGENTBOX_VERL_CONFIG:-$HERE/agentbox-verl.yaml}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FORWARD=()  # reach Ray workers on every node, not only the ones this shell starts
for var in AGENTBOX_VERL_CONFIG AGENTBOX_TASKS; do
  if [ -n "${!var:-}" ]; then
    FORWARD+=("++ray_kwargs.ray_init.runtime_env.env_vars.$var='${!var}'")
  fi
done
if [ "$TP" -gt 1 ]; then
  # vLLM's custom all-reduce failed with "invalid argument" (CUDA IPC) on our PCIe-only GPU
  # pair under expandable segments; NCCL does the tensor-parallel reductions instead
  FORWARD+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True")
fi

status=0
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/val.parquet" \
  data.train_batch_size=8 data.max_prompt_length=1024 data.max_response_length=3072 \
  data.filter_overlong_prompts=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${MODEL:-Qwen/Qwen3-0.6B}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.agent.num_workers="$WORKERS" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$HERE/agent_loops.yaml" \
  trainer.use_v1=False \
  "${FORWARD[@]}" \
  trainer.n_gpus_per_node="$GPUS" trainer.nnodes="$NNODES" \
  trainer.logger="[console]" trainer.project_name=agentbox \
  trainer.experiment_name="${EXPERIMENT:-qwen3-0.6b-grpo-${NNODES}x${GPUS}gpu-tp${TP}}" \
  trainer.total_training_steps="$STEPS" trainer.test_freq=5 trainer.val_before_train=True \
  trainer.save_freq="${SAVE_FREQ:--1}" trainer.default_local_dir="${CKPT:-$PWD/checkpoints/agentbox}" \
  trainer.rollout_data_dir="${ROLLOUTS:-$PWD/rollouts}" \
  "$@" || status=$?

# Ray SIGKILLs the agent-loop workers at the end. Their sandboxes die with them
# (die_with_owner); this removes older or unbound ones, on every node.
if [ "$NNODES" -gt 1 ]; then
  python -m agentbox sandbox prune --ray-address auto || true
else
  python -m agentbox sandbox prune || true
fi
exit "$status"
