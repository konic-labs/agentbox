#!/usr/bin/env bash
# GRPO on ONE GPU with verl + AgentBox sandboxes: Qwen3-0.6B (thinking), full fine-tune.
#
# Tested with verl 0.9.1 (pip install "verl[vllm]==0.9.1"), torch 2.11 / CUDA 13, on
# 1x L4 24 GB, 4 vCPU, 15 GB RAM. About 2.5 minutes per step (8 tasks x 8 samples).
#
#   python examples/verl/make_coding_tasks.py data/tasks
#   python examples/verl/prepare_data.py --out data/verl
#   STEPS=30 examples/verl/run_grpo_single_gpu.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$PWD/data/verl}
export AGENTBOX_VERL_CONFIG=${AGENTBOX_VERL_CONFIG:-$HERE/agentbox-verl.yaml}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
# small host: keep Ray lean, give its OOM killer headroom, avoid fragmentation
export RAY_memory_usage_threshold=0.97 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# a checkpoint save builds full fp32 state dicts on the CPU; a fixed mmap threshold makes
# glibc hand that memory back afterwards instead of keeping ~2.5 GB in the worker's heap
export MALLOC_MMAP_THRESHOLD_=1048576
STEPS=${STEPS:-30}

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/val.parquet" \
  data.train_batch_size=8 data.max_prompt_length=1024 data.max_response_length=3072 \
  data.filter_overlong_prompts=True data.dataloader_num_workers=0 \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${MODEL:-Qwen/Qwen3-0.6B}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents="[model,hf_model]" \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$HERE/agent_loops.yaml" \
  reward.num_workers=1 \
  trainer.use_v1=False \
  ray_kwargs.ray_init.num_cpus=16 \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory=1500000000 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.logger="[console]" trainer.project_name=agentbox \
  trainer.experiment_name="${EXPERIMENT:-qwen3-0.6b-grpo}" \
  trainer.total_training_steps="$STEPS" trainer.test_freq=5 trainer.val_before_train=True \
  trainer.save_freq="$STEPS" trainer.default_local_dir="${CKPT:-$PWD/checkpoints/agentbox}" \
  trainer.rollout_data_dir="$PWD/rollouts" \
  "$@"
