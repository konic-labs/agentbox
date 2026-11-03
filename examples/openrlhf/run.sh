#!/usr/bin/env bash
# GRPO with OpenRLHF + AgentBox: every episode runs token-in/token-out on OpenRLHF's own vLLM
# engines, so the trained tokens and logprobs are exactly the sampled ones.
#
# Tested with OpenRLHF 0.11.2 (vLLM 0.29, DeepSpeed 0.19.6, torch 2.13 / CUDA 13, flash-attn
# 2.8.3 built for sm_120) on 2x RTX PRO 6000 Blackwell (96 GB): one vLLM engine per GPU with the
# actor colocated (ZeRO-2), Qwen3-0.6B, 8 tasks x 8 samples per step.
#
#   python examples/verl/make_coding_tasks.py data/tasks
#   STEPS=10 examples/openrlhf/run.sh              # 10 steps of 8 tasks x 8 samples
#
# The run takes exactly STEPS full rollout batches (--data.max_samples): OpenRLHF trains on a
# last, partial batch too, and on two ranks that batch hung ZeRO-2's gradient all-reduce.
#
# Remote sandboxes: AGENTBOX_OPENRLHF_CONFIG with a `remote` section (see the verl example).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TASKS=${TASKS:-data/tasks/train.jsonl}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
STEPS=${STEPS:-10}
BATCH=8                                         # tasks per step (x 8 samples)
python "$HERE/prepare_data.py" --tasks "$TASKS" --out data/openrlhf.jsonl --repeat "$STEPS"
AGENTBOX_TASKS=$(realpath "$TASKS")
export AGENTBOX_TASKS   # labels are task ids, resolved in the vLLM engine actors
export AGENTBOX_OPENRLHF_CONFIG=${AGENTBOX_OPENRLHF_CONFIG:-$HERE/agentbox-openrlhf.yaml}
AGENT_FUNC=$(python -c "import agentbox.integrations.openrlhf as m; print(m.__file__)")

status=0
python -m openrlhf.cli.train_ppo_ray \
  --actor.model_name_or_path "${MODEL:-Qwen/Qwen3-0.6B}" \
  --actor.num_nodes 1 --actor.num_gpus_per_node "$GPUS" \
  --vllm.num_engines "$GPUS" --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.5 --vllm.enable_sleep --ds.enable_sleep --train.colocate_all \
  --ds.zero_stage 2 --ds.param_dtype bf16 --actor.gradient_checkpointing_enable \
  --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
  --rollout.batch_size "$BATCH" --rollout.n_samples_per_prompt 8 --train.batch_size $((BATCH * 8)) \
  --rollout.max_new_tokens 2048 --data.max_len 4096 --rollout.temperature 1.0 \
  --train.micro_batch_size 8 --rollout.micro_batch_size 16 \
  --actor.adam.lr 2e-6 --train.num_episodes 1 --data.max_samples $((BATCH * STEPS)) \
  --train.agent_func_path "$AGENT_FUNC" \
  --data.prompt_dataset data/openrlhf.jsonl --data.input_key prompt --data.label_key label \
  --logger.logging_steps 1 --ckpt.save_steps -1 --ckpt.output_dir "${CKPT:-checkpoints/openrlhf}" \
  "$@" || status=$?

# Sandboxes die with the workers Ray SIGKILLs (die_with_owner); this catches older or unbound ones
python -m agentbox sandbox prune || true
exit "$status"
