# Performance

RL rollout collection is a throughput problem with a memory constraint:
thousands of short episodes, each holding a container, a model conversation
and command output. AgentBox is designed so that cost scales with
*concurrency*, not with the number of rollouts.

## Measured

On a 4-vCPU / 16 GB host with an L4 GPU (Docker 27, `python:3.12-slim-bookworm`):

| Scenario | 0.2.x | 0.3 |
| --- | --- | --- |
| Remove a sandbox | 5.11 s (stop timeout on PID 1) | 0.08 s (p50) |
| Create + start a sandbox | n/a | 0.17 s (p50); 0.02 s from the warm pool |
| A command printing 50 MB | 22.8 s, +95 MB RSS | 0.17 s, +1 MB RSS (256 KiB head+tail kept) |
| A command printing 200 MB | timeout, 430 MB RSS | killed at 64 MiB after 0.28 s, 68 MB process RSS |
| `echo` after 8 concurrent command timeouts | 19.1 s (executor threads starved) | 0.13 s; no leaked processes |
| 64 rollouts, concurrency 8, 1.5 s model latency | n/a | time to first model call p50 0.31 s → 0.04 s with `warm_pool: 4` |
| 2 000 streamed rollouts × 50 KB output | all held in memory | < 25 MB peak Python heap |
| 10-turn Qwen3 episode, token data | one prompt per turn | 3–5x smaller (shared prefixes stored once); 1 sequence with TITO |

## Where the time and memory go

**Docker.** One `DockerBackend` per process holds one client with a
connection pool sized to its thread pools. Containers are created and removed
through the low-level API in a single call each (`remove(force=True)`, no stop
timeout). Images are checked and profiled once per process, and pytest and
setup steps are committed into cached images instead of repeating per
rollout.

**Commands.** Output streams into bounded head+tail buffers
(`max_output_bytes`), and a command that exceeds `kill_output_bytes` is killed.
Timeouts kill the command's process tree, found by a per-exec environment
marker, so a hung command frees its thread immediately. Reads of large files
are capped (`read_file` 1 MiB) and copied as streams.

**Agent loop.** The wire payload is built incrementally: each message is
converted once, not once per turn. Raw provider responses are not stored
unless `record_raw: true`. Tool output shown to the model is truncated
(`max_tool_output_chars`); records keep a preview.

**Runner.** Jobs are generated lazily from any iterable, results flow through
a bounded queue, metrics aggregate online, and sinks write as rollouts finish.
`run_tasks` (which returns a list) is the only API that holds everything, and
it is meant for small runs.

**Model clients.** One client per endpoint is reused across rollouts, so
connections are pooled. The SDK's own retries are off, which avoids double
retries. `model_concurrency` caps in-flight requests independently of sandbox
concurrency.

## Knobs

| Knob | Default | Raise when | Lower when |
| --- | --- | --- | --- |
| `concurrency` (`-c`) | 8 | the Docker host has idle CPU and memory | sandboxes contend for CPU (slow tests) |
| `model_concurrency` | = concurrency | the inference server is underutilised | it queues or 429s |
| `warm_pool` | 0 | container start shows up in time-to-first-call | memory is tight, or tasks all have distinct images |
| `limits.cpu_count` / `memory_mb` | 1.0 / 512 | tests are slow or OOM | you need more concurrent sandboxes |
| `max_output_bytes` | 256 KiB | verifier output is cut too early | memory per exec matters |
| `kill_output_bytes` | 64 MiB | tasks legitimately print a lot | runaway output costs time |
| `AGENTBOX_DOCKER_IO_THREADS` | 128 | more than ~100 commands run at once | n/a |
| `AGENTBOX_DOCKER_CTL_THREADS` | 16 | create/remove dominate under high concurrency | n/a |
| `cache_setup` | on | n/a | setup must run fresh each time (non-deterministic setup) |
| `--compress` | off | trajectories are large (token data) | you grep files directly |

Rules of thumb: sandbox concurrency ≈ (host CPUs ÷ `cpu_count`) × 1.5 for
test-heavy tasks, more for model-bound episodes. Raise the open-file limit of
the training process (`ulimit -n 65536`) above roughly 64 concurrent sandboxes.
Pre-build caches with `agentbox cache warm` before a long run.
