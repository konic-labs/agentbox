# Examples

All examples need Docker. Most use a scripted model, so they run without a GPU
or API key.

| Example | Shows |
| --- | --- |
| [`hello_rollout.py`](hello_rollout.py) | One rollout of a task with a scripted policy; the saved trajectory |
| [`parallel_rollouts.py`](parallel_rollouts.py) | `ParallelRunner` over several tasks; aggregate metrics |
| [`env_loop.py`](env_loop.py) | The step-wise environment API with your own generation (`--scripted`, or any OpenAI-compatible server) |
| [`custom_tools_and_rewards.py`](custom_tools_and_rewards.py) | `@tool`, `BaseTool`, `ToolContext` and a registered reward component |
| [`remote_env.py`](remote_env.py) | `agentbox serve` + `RemoteEnv` (needs `agentbox-rl[server]`) |
| [`train_trl_grpo.py`](train_trl_grpo.py) | GRPO with TRL's `environment_factory` (needs `agentbox-rl[trl]` and a GPU) |
| [`trl/`](trl) | TRL on several GPUs: DDP, DeepSpeed ZeRO-2, vLLM colocated or as a server |
| [`verl/`](verl) | Tested GRPO recipes for verl on one GPU and on several GPUs or nodes (local or remote sandboxes): task generator, dataset preparation, agent-loop registration, worker settings and launch scripts |
| [`openrlhf/`](openrlhf) | A tested GRPO launch script for OpenRLHF's agent mode on several GPUs, with dataset preparation and worker settings |
| [`skyrl/`](skyrl) | A tested GRPO launch script for SkyRL: dataset preparation, an entrypoint that registers the environment, worker settings |
| [`art_integration.py`](art_integration.py) | GRPO groups converted to ART's format (live `TrajectoryGroup`s with `agentbox-rl[art]`) |
| [`bench_run.py`](bench_run.py) | The `coding-mini` benchmark suite with a scripted model |
| [`generate_tasks.py`](generate_tasks.py) | Teacher-generated tasks with QC |
| [`agentbox.yaml`](agentbox.yaml) | A project file with model profiles |
| [`tasks/`](tasks), [`benchmarks/`](benchmarks) | Example tasks and a frozen suite |

```bash
python examples/hello_rollout.py
python examples/env_loop.py --scripted
python examples/env_loop.py --base-url http://localhost:8000/v1 --model qwen3-1.7b
```
