"""GRPO with TRL, one AgentBox sandbox per rollout.

    pip install "agentbox-rl[trl]"
    python examples/train_trl_grpo.py --tasks examples/tasks --model Qwen/Qwen3-0.6B --steps 20

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
    ap.add_argument("--out", default="grpo-agentbox")
    args = ap.parse_args()

    tasks = list(iter_tasks(args.tasks))
    env_cls = make_environment_factory(
        tasks,
        sandbox=SandboxConfig(limits=ResourceLimits(memory_mb=1024), warm_pool=4),
        agent=AgentConfig(max_steps=8),
        tools=["list_files", "read_file", "write_file", "run_tests"],
        rewards=[RewardSpec(name="verifier"), RewardSpec(name="valid_tool_call_ratio", weight=0.1)],
    )
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=tasks_to_dataset(tasks * 4),
        environment_factory=env_cls,
        args=GRPOConfig(
            output_dir=args.out,
            max_steps=args.steps,
            num_generations=args.group_size,
            per_device_train_batch_size=args.group_size,
            max_completion_length=1024,
            max_tool_calling_iterations=8,
            learning_rate=1e-6,
            chat_template_kwargs={"enable_thinking": False},
            logging_steps=1,
            bf16=True,
            report_to="none",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
