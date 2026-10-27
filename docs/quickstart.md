# Quick start

## Install

```bash
pip install agentbox-rl                 # core: sandboxes, tools, runner, CLI
pip install "agentbox-rl[trl]"          # + TRL / transformers / datasets
pip install "agentbox-rl[hf]"           # + transformers, datasets, pyarrow (TITO, HF export)
pip install "agentbox-rl[server]"       # + FastAPI / uvicorn for `agentbox serve`
pip install "agentbox-rl[rich]"         # nicer tables and progress bars
pip install "agentbox-rl[all]"          # rich + hf + server + generate
```

AgentBox needs Python 3.11+ and a Docker daemon the current user can reach.
verl, OpenRLHF and SkyRL are installed with their own instructions; AgentBox
imports them only when you use their adapter.

## Check the environment

```bash
agentbox doctor
```

`doctor` checks Python, the Docker daemon, the sandbox image (its bash, python
and pytest), free disk space, cached images, leftover containers and installed
trainer integrations. Add `--model ID --base-url URL` to also probe an endpoint
for chat and tool calling (`--api-key-env NAME` reads the key from the
environment; keys never go on the command line).

## A project

```bash
agentbox init                 # agentbox.yaml + tasks/example_add/task.json
agentbox init --trainer trl   # ... and train_trl.py
```

`agentbox.yaml` holds named model profiles plus sandbox, agent and reward
defaults. Every command reads it (`--config` picks another file and `--profile`
another profile); see [configuration](configuration.md).

## First rollouts without a model

```bash
agentbox run tasks --mock -n 2 --out runs/smoke
```

`--mock` uses a no-op policy. The pipeline still provisions real sandboxes,
seeds the task, runs the verifier and writes trajectories, so this checks
Docker and the tasks. Expect `success 0%`, because the mock never edits
anything.

## Rollouts against a real model

Serve a model with tool calling. With vLLM, for example:

```bash
vllm serve Qwen/Qwen3-1.7B --enable-auto-tool-choice --tool-call-parser hermes \
  --served-model-name qwen3-1.7b
```

Point a profile at it:

```yaml
# agentbox.yaml
default_model: local
models:
  local:
    model: qwen3-1.7b
    base_url: http://localhost:8000/v1
    api_key: ${VLLM_API_KEY:-EMPTY}
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
```

```bash
agentbox doctor --model qwen3-1.7b --base-url http://localhost:8000/v1   # tool calling works?
agentbox run tasks -n 8 -c 8 --out runs/first
```

`-n 8` runs a group of 8 samples per task (what GRPO needs), and `-c 8` runs 8
sandboxes at a time. Trajectories stream to `runs/first/trajectories.jsonl` as
rollouts finish, and `runs/first/run.json` records the configuration.
Interrupted runs continue with `--resume`, which skips finished samples.

## Look at the results

```bash
agentbox traj stats runs/first        # success, reward mean/std, groups, tools, failure kinds
agentbox traj show runs/first         # status, verifier output, last tool calls
agentbox traj validate runs/first     # schema + conversation validity
agentbox traj render runs/first -o report.html
```

## Export training data

```bash
agentbox traj export runs/first -f grpo -o grpo.jsonl --drop-zero-variance   # groups with reward spread
agentbox traj export runs/first -f chat -o sft.jsonl --successful-only       # SFT conversations
agentbox traj export runs/first -f tokens -o tokens.jsonl                    # token ids + loss masks
```

Infrastructure failures are left out of every export unless you pass
`--include-infra`. [Trajectories](trajectories.md) lists all formats.

## Python

```python
import asyncio

from agentbox import ModelConfig, ParallelRunner, Task, VerifierSpec

task = Task(
    task_id="fix_add",
    description="add(a, b) in calc.py returns the wrong value. Fix it so the tests pass.",
    starter_files={
        "calc.py": "def add(a, b):\n    return a - b\n",
        "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    },
    verifier=VerifierSpec(type="pytest", protected_paths=["test_*.py"]),
)


async def main() -> None:
    runner = ParallelRunner(
        model=ModelConfig(model="qwen3-1.7b", base_url="http://localhost:8000/v1", api_key="EMPTY"),
        concurrency=8,
    )
    async for group in runner.stream_groups([task], group_size=8):
        print([t.reward for t in group])


asyncio.run(main())
```

## Next

- Train: [training](training.md) and [TRL](integrations/trl.md), [verl](integrations/verl.md),
  [OpenRLHF](integrations/openrlhf.md), [SkyRL](integrations/skyrl.md), [ART](integrations/art.md)
- Write good tasks: [tasks](tasks.md), [rewards](rewards.md)
- Capture exact tokens for RL: [models: token capture and TITO](models.md#token-capture)
