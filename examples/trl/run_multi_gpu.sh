#!/usr/bin/env bash
# GRPO with TRL on the GPUs of one node. Every rank builds its own AgentBox environment pool,
# and host_max_sandboxes bounds the sandboxes of all ranks together.
#
# Tested with TRL 1.17.1 (vLLM 0.29, DeepSpeed 0.19.6, torch 2.13 / CUDA 13) on 2x RTX PRO 6000
# Blackwell (96 GB), Qwen3-0.6B, 8 samples per task.
#
#   python examples/verl/make_coding_tasks.py data/tasks
#   MODE=ddp examples/trl/run_multi_gpu.sh                     # DDP, transformers generation
#   MODE=zero2 VLLM=colocate examples/trl/run_multi_gpu.sh     # ZeRO-2, vLLM inside every rank
#   MODE=server examples/trl/run_multi_gpu.sh                  # `trl vllm-serve` on the last GPU
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
MODE=${MODE:-ddp}
VLLM=${VLLM:-off}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
ARGS=(--tasks "${TASKS:-data/tasks/train.jsonl}" --model "$MODEL" --steps "${STEPS:-8}"
      --host-max-sandboxes "${HOST_MAX_SANDBOXES:-64}" --out "${OUT:-grpo-agentbox-$MODE}")

status=0
case "$MODE" in
  ddp)
    accelerate launch --multi_gpu --num_processes "$GPUS" --mixed_precision bf16 \
      "$HERE/../train_trl_grpo.py" "${ARGS[@]}" --vllm "$VLLM" "$@" || status=$? ;;
  zero2)
    accelerate launch --config_file "$HERE/zero2.yaml" --num_processes "$GPUS" \
      "$HERE/../train_trl_grpo.py" "${ARGS[@]}" --vllm "$VLLM" "$@" || status=$? ;;
  server)  # generation on the last GPU, training ranks on the others
    SERVER_GPU=$((GPUS - 1))
    CUDA_VISIBLE_DEVICES=$SERVER_GPU trl vllm-serve --model "$MODEL" --port 8000 \
      --gpu-memory-utilization 0.5 > "${SERVER_LOG:-vllm-serve.log}" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null || true' EXIT
    TRAIN_GPUS=$(seq -s, 0 $((SERVER_GPU - 1)))
    if [ "$SERVER_GPU" -gt 1 ]; then LAUNCH=(--multi_gpu); else LAUNCH=(); fi
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch "${LAUNCH[@]}" --num_processes "$SERVER_GPU" \
      --mixed_precision bf16 "$HERE/../train_trl_grpo.py" "${ARGS[@]}" --vllm server "$@" \
      || status=$? ;;
  *) echo "MODE must be ddp, zero2 or server" >&2; exit 2 ;;
esac
exit "$status"
