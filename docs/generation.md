# Task Generation

AgentBox can generate complete **Task** definitions with a teacher model so you
do not hand-author large datasets. Generation is optional and isolated under
`agentbox.tasks.generate`.

Works with any **OpenAI-compatible** endpoint (self-hosted vLLM / TGI on EC2,
local servers, cloud gateways). No vendor lock-in.

Install:

```bash
uv pip install -e ".[generate]"   # pulls dspy; OpenAI JSON fallback works without it
```

## Goals

- Emit a full Task (files + setup + verifier + metadata), not just a prompt  
- Provider-agnostic (`base_url` / `api_key`)  
- **Docker QC**: seed + verifier fails on starter  
- **LLM QC**: second API call (same teacher by default) via DSPy structured judge  
- Prefer **stubs**, not near-complete solutions with a one-token bug  

## API

```python
from agentbox.tasks.generate import TaskGenerator, GenerateConfig

# Self-hosted Qwen3.6-27B (example)
gen = TaskGenerator(GenerateConfig(
    model="Qwen/Qwen3.6-27B",
    base_url="http://YOUR_EC2:8000/v1",
    api_key="EMPTY",
    temperature=0.7,
    max_tokens=3072,
    validate_in_docker=True,
    validate_with_llm=True,       # second call: DSPy structured task judge
    llm_judge_min_score=0.65,
    use_dspy=True,
    max_retries=3,
))

task = await gen.generate(
    difficulty="medium",
    domain="python",
    constraints="stub starter only; agent implements from pytest contract",
    tags=["algorithms"],
)
task.save_json("tasks/generated/task.json")

# Re-audit an existing task (Docker + LLM, or LLM-only)
report = await gen.validate_task(existing_task)
llm_only = await gen.validate_task_llm_only(existing_task)
print(report.ok, report.llm_score, report.errors)
```

CLI-style script:

```bash
python examples/generate_tasks.py \
  --model Qwen/Qwen3.6-27B \
  --base-url http://127.0.0.1:8000/v1 \
  --n 5 \
  --difficulty easy \
  --out tasks/generated
```

## CLI

```bash
# Single task (teacher from flags or agentbox.yaml)
agentbox generate one -m MODEL --base-url http://localhost:8000/v1 --out generated/tasks

# Batch with resume, static/docker/llm QC, dedup
agentbox generate batch -m MODEL --base-url URL -n 20 -c 8 \
  --static-qc --docker-qc --llm-judge --dedup --out generated/tasks

# Two-stage: full golden solution → AST stubs → starter-fail + solution-pass
agentbox generate batch -m MODEL --base-url URL -n 20 --two-stage

# Re-audit
agentbox generate validate-llm generated/tasks -m MODEL --base-url URL
agentbox generate validate-docker generated/tasks
```

## Pipeline

```txt
Teacher (DSPy Predict or OpenAI JSON)  [API call 1]
  -> parse fields -> Task
  -> [optional two-stage] solution_files → AST strip → starter stubs
  -> Static QC (stubs, asserts, leaks, path consistency)
  -> Docker: seed + verifier fails on starter
  -> [two-stage] inject golden → verifier must pass
  -> LLM judge (DSPy ValidateCodingTask)  [API call 2, same model by default]
  -> Dedup against resume set (batch)
  -> accept or retry
```

Both generation and validation can hit the **same high-throughput self-hosted**
endpoint (e.g. Qwen3.6-27B on EC2). Optionally set `validator_model` /
`validator_base_url` to a different OpenAI-compatible server.

## LLM task judge (DSPy)

Signature: `ValidateCodingTask` in `agentbox.tasks.generate.signatures`.

Structured fields:

| Field | Meaning |
| --- | --- |
| `accept` | Accept for agent training/bench |
| `score` | 0..1 quality |
| `starter_is_near_solution` | Full impl with tiny bug → reject |
| `description_leaks_fix` | Spoilers in description/comments |
| `agent_must_implement` | Must write real logic (not one-token patch) |
| `reasons` / `suggested_fixes` | Rationale |

Accept requires: `accept` and `score >= llm_judge_min_score` and not near-solution
and not leak and `agent_must_implement`.

Standalone:

```python
from agentbox.config import ModelConfig
from agentbox.tasks.generate import validate_task_llm

report = await validate_task_llm(
    task,
    model_config=ModelConfig(model="Qwen/Qwen3.6-27B", base_url="http://…/v1", api_key="EMPTY"),
    min_score=0.65,
    use_dspy=True,
)
```

## DSPy vs JSON fallback

| Path | When |
| --- | --- |
| DSPy generator / judge | `use_dspy=True` and `dspy` importable |
| OpenAI JSON object | DSPy missing or `use_dspy=False` |

## GenerateConfig

See [Configuration](configuration.md#generateconfig).

Key QC flags:

| Field | Default | Meaning |
| --- | --- | --- |
| `validate_in_docker` | `True` | Live seed + fail-on-starter |
| `validate_with_llm` | `True` | DSPy structured quality judge |
| `llm_judge_min_score` | `0.65` | Minimum judge score to accept |
| `validator_model` | same as `model` | Optional judge model override |

## Provenance metadata

Accepted tasks set:

- `metadata["source"] = "dspy"` (including JSON-fallback path today)  
- `metadata["generator_model"] = <teacher model id>`  
- `metadata["llm_judge_score"]` / `llm_judge_accept` when LLM QC ran  

## Separation of models

| Role | Typical model |
| --- | --- |
| Teacher (generate + judge) | Self-hosted Qwen3.6-27B (high throughput) |
| Student (rollouts) | Smaller local model under test |

## Failure modes

| QC failure | Handling |
| --- | --- |
| Invalid JSON / schema | Retry |
| `setup_commands` fail | Retry |
| Verifier already passes on starter | Reject (if `expect_fail_on_starter`) |
| LLM judge near-solution / leak / low score | Reject + retry |
| Exhausted retries | `TaskGenerationError` |

`generate_many` skips individual failures and returns only successful tasks.
