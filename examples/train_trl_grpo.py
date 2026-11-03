"""GRPO with TRL, one AgentBox sandbox per rollout.

    pip install "agentbox-rl[trl]"
    python examples/train_trl_grpo.py --tasks examples/tasks --model Qwen/Qwen3-0.6B --steps 20

Several GPUs (one environment pool per rank; examples/trl/run_multi_gpu.sh):

    accelerate launch --multi_gpu --num_processes 2 examples/train_trl_grpo.py --vllm colocate

Infrastructure failures come back as reward None, which TRL treats as
unscorable (no fake negatives in the group baseline).
"""

from __future__ import annotations

import argparse

from trl import GRPOConfig, GRPOTrainer

from agentbox import AgentConfig, ResourceLimits, RewardSpec, SandboxConfig, iter_tasks
from agentbox.integrations.trl import make_environment_factory, tasks_to_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="examples/tasks")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=None, help="default: --group-size")
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--vllm", choices=["off", "colocate", "server"], default="off")
    ap.add_argument("--vllm-server-host", default="127.0.0.1")
    ap.add_argument("--host-max-sandboxes", type=int, default=None, help="cap across ranks")
    ap.add_argument("--host-slot-timeout", type=float, default=1800.0, help="then infra failure")
    ap.add_argument("--out", default="grpo-agentbox")
    args = ap.parse_args()

    tasks = list(iter_tasks(args.tasks))
    env_cls = make_environment_factory(
        tasks,
        sandbox=SandboxConfig(
            limits=ResourceLimits(memory_mb=1024),
            warm_pool=4,
            host_max_sandboxes=args.host_max_sandboxes,
            host_slot_timeout_s=args.host_slot_timeout,
        ),
        agent=AgentConfig(max_steps=8),
        tools=["list_files", "read_file", "write_file", "run_tests"],
        rewards=[RewardSpec(name="verifier"), RewardSpec(name="valid_tool_call_ratio", weight=0.1)],
    )
    vllm = (
        {}
        if args.vllm == "off"
        else {
            "use_vllm": True,
            "vllm_mode": args.vllm,
            "vllm_gpu_memory_utilization": 0.3,
            "vllm_server_host": args.vllm_server_host,
        }
    )
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=tasks_to_dataset(tasks * 4),
        environment_factory=env_cls,
        args=GRPOConfig(
            output_dir=args.out,
            max_steps=args.steps,
            num_generations=args.group_size,
            per_device_train_batch_size=args.per_device_batch or args.group_size,
            max_completion_length=args.max_completion_length,
            max_tool_calling_iterations=8,
            learning_rate=1e-6,
            chat_template_kwargs={"enable_thinking": False},
            logging_steps=1,
            bf16=True,
            report_to="none",
            save_strategy="no",
            **vllm,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
