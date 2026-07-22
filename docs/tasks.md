# Tasks

A **Task** is the contract between dataset authoring, container seeding, and
rollouts. Everything needed to prepare the environment lives on the Task object.

## Schema

```python
from agentbox import Task, VerifierSpec
from agentbox.types import VerifierType

task = Task(
    task_id="fix_fizzbuzz_001",
    description="Fix fizzbuzz so tests pass.",  # agent user message
    starter_files={"fizzbuzz.py": "...", "test_fizzbuzz.py": "..."},
    setup_commands=["pip install -q pytest"],
    verifier=VerifierSpec(
        type=VerifierType.PYTEST,
        command="python -m pytest -q",
        success_exit_code=0,
        timeout_s=30,
        reward_success=1.0,
        reward_failure=0.0,
    ),
    metadata={
        "difficulty": "easy",
        "tags": ["python", "algorithms"],
        "estimated_steps": 10,
        "language": "python",
    },
)
```

### Fields

| Field | Purpose |
| --- | --- |
| `task_id` | Unique id |
| `description` | Initial user message for the agent |
| `starter_files` | `path → content` written under `/workspace` |
| `setup_commands` | Shell commands after files, before agent |
| `verifier` | Success criterion → reward |
| `metadata` | Filtering, curriculum, provenance |
| `system_prompt_extra` | Appended to system prompt |
| `allowed_tools` | Optional per-task tool allowlist |
| `max_steps` | Optional per-task step override |

### Load / save

```python
task = Task.from_json("task.json")
task = Task.from_dir("tasks/fix_fizzbuzz")  # task.json + optional files/
task.save_json("out/task.json")
```

`from_dir` merges `files/**` into `starter_files` (files/ wins on conflict).

## Seeding (`TaskSeeder`)

Runs **before** the agent loop:

1. `write_files` for all `starter_files` (nested paths preserved)  
2. Each `setup_commands` entry via shell; stop on first non-zero / timeout  
3. Healthcheck (`true`)  

Failure → rollout `final_status=ERROR`, agent never starts.

## Verification (`Verifier`)

| Type | Command construction |
| --- | --- |
| `pytest` | `command` or `python -m pytest -q {path}` |
| `command` | `command` required |
| `custom` | Reserved; provide `command` today |

Success: `exit_code == success_exit_code` and not timed out.  
Reward: `reward_success` or `reward_failure`.

Default policy: agent “final message” alone does **not** grant reward — tests do.

## Reward shaping

```python
from agentbox.tasks.rewards import shaped_reward
# used inside Rollout when AgentConfig.*_penalty fields are set
```

## Filtering & curriculum

```python
from agentbox.tasks import filter_tasks, sample_curriculum, group_by_difficulty

easy_py = filter_tasks(tasks, difficulty="easy", tags=["python"])
batch = sample_curriculum(tasks, n=20)  # easy → medium → hard preference
groups = group_by_difficulty(tasks)
```

Filters read `metadata.difficulty`, `tags`, `domain`, `language`, `estimated_steps`.

## Bulk datasets

```python
from agentbox.tasks.bulk import (
    save_task_dataset, load_task_dataset,
    export_tasks_jsonl, import_tasks_jsonl,
)

save_task_dataset(tasks, "tasks/generated")
tasks = load_task_dataset("tasks/generated")
export_tasks_jsonl(tasks, "tasks.jsonl")
```

## Automated generation

See [Task Generation](generation.md) for `TaskGenerator`, DSPy, and live QC.

## Example task file

See `examples/tasks/fix_fizzbuzz/task.json` in the repository.
