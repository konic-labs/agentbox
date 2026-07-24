# Changelog

## 0.2.0 — production hardening (lab)

### Added
- **LLM task judge** (`validate_task_llm`, DSPy `ValidateCodingTask`) as generation QC
- **Static QC** (`validate_task_static`) for stubs, asserts, leak patterns, path consistency
- **Two-stage generation** (`--two-stage`): golden solution → AST stub strip → starter-fail + solution-pass Docker QC
- **Batch generate API** (`batch_generate`, `BatchGenerateConfig`) with resume, dedup, review queue
- **Dedup** (`task_signature_hash`, Jaccard near-dup) and difficulty heuristics
- **Review queue** export/import (`review_queue.jsonl` / decisions)
- **CLI** `agentbox generate` (`one`, `batch`, `validate-llm`, `validate-docker`)
- **CLI** `agentbox traj` (`show`, `render`) HTML dashboard (official vs agent self-check)
- **Doctor model probe** (`--model` / `--base-url` tool-calling check)
- **Config load** (`agentbox.yaml` / env interpolation; used by generate + bench)
- **Bench multi-student** (`--student id=model@url`, `--limit`, `--probe`, `--disable-thinking`)
- **Richer** `bench show --out` (task×model matrix, failure clusters)
- **Jobs** types + local artifact store + validation cache
- **HTTP retries** on OpenAI-compatible client (429/5xx)
- Official **verify_stdout/stderr** on trajectory metadata
- Docs: ART GRPO recipe, tool-calling matrix; example `examples/agentbox.yaml`
- CI: unit + hermetic smoke; optional provider doctor job

### Changed
- Stub-first generation prompts; concurrent DSPy uses `dspy.context`
- Generation defaults favor structured JSON + optional LLM/static QC
- Package version **0.2.0**

## 0.1.0
- Initial ART-native rollout engine, benches, DSPy generate + Docker QC
