#!/usr/bin/env bash
# GRPO with SkyRL + AgentBox: a skyrl_gym text environment (Hermes tool calls) per episode.
#
# Tested with skyrl 0.4.0 (vLLM 0.30, torch 2.13 / CUDA 13, flash-attn 2.8.3 built for sm_120)
# on 2x RTX PRO 6000 Blackwell (96 GB): one vLLM engine per GPU, FSDP policy colocated,
# Qwen3-0.6B, 8 tasks x 8 samples per step. Environments run on SkyRL's env threads, one per
# trajectory. skyrl 0.4.0 needs vllm-router 0.1.14: later releases dropped a field it reads.
#
#   python examples/verl/make_coding_tasks.py data/tasks
#   python examples/skyrl/prepare_data.py --out data/skyrl
#   EPOCHS=3 examples/skyrl/run.sh                  # 27 tasks x 3 epochs = 9 steps
#
# Any positive trainer.ckpt_interval also saves a checkpoint at every epoch's end (9 GB each
# for Qwen3-0.6B with its optimizer state); -1 saves none.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$PWD/data/skyrl}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
export AGENTBOX_SKYRL_CONFIG=${AGENTBOX_SKYRL_CONFIG:-$HERE/agentbox-skyrl.yaml}

status=0
python "$HERE/main_agentbox.py" \
  data.train_data="['$DATA/train.parquet']" data.val_data="['$DATA/val.parquet']" \
  environment.env_class=agentbox \
  environment.skyrl_gym.max_env_workers=64 \
  trainer.policy.model.path="${MODEL:-Qwen/Qwen3-0.6B}" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$GPUS" trainer.placement.ref_num_gpus_per_node="$GPUS" \
  generator.inference_engine.num_engines="$GPUS" generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false \
  trainer.train_batch_size=8 trainer.policy_mini_batch_size=8 \
  trainer.micro_train_batch_size_per_gpu=8 trainer.micro_forward_batch_size_per_gpu=8 \
  generator.n_samples_per_prompt=8 generator.max_turns=6 generator.batched=false \
  generator.sampling_params.max_generate_length=2048 generator.max_input_length=4096 \
  trainer.max_prompt_length=1024 trainer.policy.optimizer_config.lr=2e-6 \
  trainer.epochs="${EPOCHS:-3}" trainer.eval_before_train=false trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 trainer.ckpt_path="${CKPT:-$PWD/checkpoints/skyrl}" \
  trainer.logger=console trainer.project_name=agentbox trainer.run_name=skyrl \
  "$@" || status=$?

# Ray SIGKILLs the workers at the end: remove the sandboxes they still hold
python -m agentbox sandbox prune || true
exit "$status"
