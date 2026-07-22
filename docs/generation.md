# Task Generation

AgentBox can generate complete **Task** definitions with a frontier model so you
do not hand-author large datasets. Generation is optional and isolated under
`agentbox.tasks.generate`.

Install:

```bash
uv pip install -e ".[generate]"   # pulls dspy; OpenAI JSON fallback works without it
```

## Goals

- Emit a full Task (files + setup + verifier + metadata), not just a prompt  
- Provider-agnostic (same `base_url` / `api_key` pattern as agents)  
- QC in a real Docker sandbox before accepting a task  
- Prefer starters that **fail** the verifier so reward is informative  

## API

```python
from agentbox.tasks.generate import TaskGenerator, GenerateConfig

gen = TaskGenerator(GenerateConfig(
    model="glm-5.2",
    base_url="https://api.featherless.ai/v1",
    api_key="...",
    temperature=0.8,
    validate_in_docker=True,
    use_dspy=True,
    max_retries=2,
))

task = await gen.generate(
    difficulty="medium",
    domain="python",
    constraints="single-file bug fix with pytest; agent must edit code",
    tags=["algorithms"],
)
task.save_json("tasks/generated/task.json")

tasks = await gen.generate_many(10, difficulty="easy", domain="python")
report = await gen.validate_task(existing_task)
```

CLI-style script:

```bash
python examples/generate_tasks.py \
  --model glm-5.2 \
  --base-url https://api.featherless.ai/v1 \
  --n 5 \
  --difficulty easy \
  --out tasks/generated
```

## Pipeline

```txt
Teacher model (DSPy Predict or OpenAI JSON)
  -> parse fields -> Task.model_validate
  -> create fresh sandbox
  -> TaskSeeder.seed (files + setup_commands)
  -> Verifier smoke
  -> reject if seed fails OR (expect_fail_on_starter and already passes)
  -> accept or retry
```

## DSPy vs JSON fallback

| Path | When |
| --- | --- |
| DSPy `CodingTaskGenerator` | `use_dspy=True` and `dspy` importable |
| OpenAI JSON object | DSPy missing or `use_dspy=False` |

Both end in `parse_task_from_prediction` for a single validation path.

DSPy signature outputs: `task_id`, `description`, `starter_files_json`,
`setup_commands_json`, `verifier_json`, `metadata_json`.

## GenerateConfig

See [Configuration](configuration.md#generateconfig). QC sandboxes typically
enable network so `pip install` setup commands can succeed.

## Provenance metadata

Accepted tasks set:

- `metadata["source"] = "dspy"` (including JSON-fallback path today)  
- `metadata["generator_model"] = <teacher model id>`  

## Separation of models

The **generator** model (teacher) may differ from the **agent** model (student).
Common pattern: strong API model generates tasks; local/smaller model collects
trajectories.

## Failure modes

| QC failure | Handling |
| --- | --- |
| Invalid JSON / schema | Retry |
| `setup_commands` fail | Retry |
| Verifier already passes on starter | Reject (if `expect_fail_on_starter`) |
| Exhausted retries | `TaskGenerationError` |

`generate_many` skips individual failures and returns only successful tasks.
