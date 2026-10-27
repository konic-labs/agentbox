# CLI

`agentbox` is a Typer application. Every command is documented in the
generated [CLI reference](cli-reference.md); this page covers the conventions
and common workflows.

## Conventions

| Global option | Effect |
| --- | --- |
| `--config/-C PATH` | Project file (default: `$AGENTBOX_CONFIG` or the nearest `agentbox.yaml`) |
| `--profile/-P NAME` | Model profile (default: `$AGENTBOX_PROFILE` or `default_model`) |
| `--json` | Machine-readable output on stdout, for every command |
| `-v` / `-vv` / `-q` | Info / debug logging / errors only |
| `--log-format json` | Structured logs on stderr (one JSON object per line) |

Global options go **before** the command: `agentbox --json traj stats runs/a`.

| Exit code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Task failures, thresholds not met (`--min-success-rate`, `--fail-on-regression`), invalid tasks or trajectories |
| 2 | Usage or configuration errors |
| 3 | Docker or the model endpoint is unavailable |

Secrets are passed by environment variable name (`--api-key-env NAME`, or
`api_key_env` in profiles), never as values on the command line. Shell
completion: `agentbox --install-completion`.

## Command map

| Group | Commands |
| --- | --- |
| Rollouts | `run`, `serve` |
| Trajectories | `traj show · stats · filter · validate · replay · render · export · diff` |
| Tasks | `task new · validate · lint · check · select · show · ls · schema` |
| Benchmarks | `bench create · freeze · validate · run · show · compare` |
| Sandboxes | `sandbox up · exec · shell · ls · rm · prune` |
| Images and caches | `image build · ls`, `cache ls · prune · warm` |
| Task generation | `generate one · batch · validate-llm · validate-docker` |
| Training | `train trl` |
| Project | `init`, `config show · validate · path`, `doctor`, `integrations`, `version` |

## Workflows

**First run**

```bash
agentbox doctor && agentbox init && agentbox run tasks --mock -n 2
```

**Collect GRPO data**

```bash
agentbox cache warm tasks/                                  # setup snapshots once
agentbox run tasks/ -n 8 -c 32 --warm-pool 4 --out runs/r1 --compress
agentbox traj stats runs/r1
agentbox traj export runs/r1 -f grpo --drop-zero-variance -o r1.grpo.jsonl
```

**Select learnable tasks for the next round**

```bash
agentbox task select tasks/ --runs runs/r1 --min-pass 0.01 --max-pass 0.99 -o round2.jsonl
agentbox run round2.jsonl -n 8 --out runs/r2
```

**Exact token data for a custom trainer**

```bash
agentbox run tasks/ --tito Qwen/Qwen3-1.7B -n 8 --out runs/tito
agentbox traj export runs/tito -f tokens -o tokens.jsonl
```

**Debug a failing task**

```bash
agentbox run tasks/x --keep-failed -n 1 --out runs/dbg
agentbox traj show runs/dbg                 # verifier output + "sandbox kept: NAME"
agentbox sandbox shell NAME
```

**Gate a model in CI**

```bash
agentbox bench run suites/coding-v1 --profile candidate --strict -o bench/candidate --min-success-rate 0.4
agentbox bench compare bench/main/report.json bench/candidate/report.json --fail-on-regression 0.02
```

**Automation**

```bash
agentbox --json run tasks/ -n 4 --out runs/a | jq '.metrics.success_rate'
agentbox --json --log-format json run ... 2> run.log.jsonl
```

## Deprecated aliases

The 0.2 commands still work but are hidden: `run-dir` (use `run DIR`),
`export` (use `traj export`), `prune` (use `sandbox prune`) and
`build-image` (use `image build`). See [migration](migration.md).
