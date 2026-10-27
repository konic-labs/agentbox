# Models

The policy is any **OpenAI Chat Completions** endpoint with tool calling (vLLM,
SGLang, TGI, Ollama, OpenAI, OpenRouter, Together, Featherless, ...), or any
object implementing the small `ModelClient` interface: TITO, mock and replay
clients, or your own.

## Endpoints

```python
from agentbox import ModelConfig

ModelConfig(
    model="qwen3-1.7b",
    base_url="http://localhost:8000/v1",
    api_key_env="VLLM_API_KEY",            # or api_key=...; never recorded
    temperature=1.0, top_p=1.0, max_tokens=2048,
    top_k=20, min_p=0.0,                   # vLLM/SGLang samplers (sent in extra_body)
    seed=None, stop=None,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

All fields are listed in the [configuration reference](config-reference.md#modelconfig).
`max_tokens_param: max_completion_tokens` is needed by OpenAI reasoning models.
`tool_choice` and `parallel_tool_calls` are forwarded as request fields.

Probe an endpoint before a long run:

```bash
agentbox doctor --model qwen3-1.7b --base-url http://localhost:8000/v1   # chat + tool calling
agentbox bench run SUITE --model ... --probe                            # probe each model first
```

### Serving open models with tool calling

```bash
vllm serve Qwen/Qwen3-1.7B --enable-auto-tool-choice --tool-call-parser hermes
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --tool-call-parser qwen25
```

Without a tool-call parser, use `agent.tool_protocol: hermes` (AgentBox renders
and parses tool calls itself) or TITO.

## Retries and errors

- The OpenAI SDK's own retries are disabled. AgentBox retries **429, 5xx,
  timeouts and connection errors** up to `max_retries` times, with exponential
  backoff and full jitter (`retry_backoff_s`, `retry_max_backoff_s`) and
  honouring `Retry-After`. Other 4xx errors (bad requests, auth) fail at once.
- Context overflows raise `ContextLengthError`. With `adaptive_max_tokens`, a
  request whose `max_tokens` does not fit is first retried with the room left.
  The loop then ends the episode as `context_length`, a policy failure.
- Errors that remain after retries end the rollout with
  `failure_kind=model_api` (infrastructure), which the runner retries as a
  whole rollout.
- Clients are reused across rollouts and closed by whoever created them; the
  runner owns the clients it builds from a `ModelConfig`.

## Reasoning

`reasoning_content` / `reasoning` fields and inline `<think>` blocks
(`strip_think: true`) are recorded on assistant messages. Whether reasoning is
sent back on later turns is `AgentConfig.send_reasoning` ([agent](agent.md#reasoning-models)).

## Token capture

RL trainers need the exact tokens that were sampled and their logprobs, not a
re-tokenisation of the text.

```yaml
models:
  local:
    model: qwen3-1.7b
    base_url: http://localhost:8000/v1
    capture_tokens: true        # vLLM: extra_body.return_token_ids -> prompt + completion token ids
    capture_logprobs: true      # logprobs=true (top_logprobs: k for alternatives)
```

Each assistant message then carries `tokens: TurnTokens`:

| Field | Meaning |
| --- | --- |
| `completion_ids` | Sampled token ids |
| `logprobs` | Sampler logprob per completion token |
| `prompt_reuse` | Leading prompt tokens shared with the previous turn's prompt + completion |
| `prompt_ids` | The rest of the prompt (only this suffix is stored) |
| `prefix_break` | The prompt did not start with the previous sequence (see below) |

Storing the shared prefix once keeps multi-turn token data 3–5x smaller than
per-turn prompts. `Trajectory.token_trace()` rebuilds training sequences
(`prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`), where
tool-result tokens are masked out. `agentbox traj export -f tokens` writes them
for trainers.

### Prefix breaks and TITO

Chat-completions servers **re-render the whole conversation** every turn. The
template re-serialises tool-call JSON and drops earlier `<think>` blocks, so the
tokens the model generated are usually *not* a prefix of the next prompt. Each
break starts a new training sequence, and recomputed prompts drift from what
was sampled. With Qwen3 on vLLM, almost every turn breaks.

**TITO (token-in/token-out)** avoids this. AgentBox renders the initial prompt
once with the model's chat template, generates from token ids through
`/v1/completions`, and appends each tool result as delta tokens. History is
never re-rendered, so every trajectory is **one** prefix-consistent sequence.

```bash
# the profile's model/base_url name the vLLM server; --tito names the HF tokenizer
agentbox run tasks --tito Qwen/Qwen3-1.7B -n 8 --out runs/tito
```

```python
from agentbox.tito import ChatTokenizer, CompletionsGenerator, TitoModelClient

tok = ChatTokenizer.from_pretrained("Qwen/Qwen3-1.7B", chat_template_kwargs={"enable_thinking": False})
gen = CompletionsGenerator(model="qwen3-1.7b", base_url="http://localhost:8000/v1")
client = TitoModelClient(tok, gen, model="qwen3-1.7b", max_tokens=1024, temperature=1.0,
                         max_context_tokens=16384)
traj = await Rollout.run(task, model=client)
assert len(traj.token_trace().sequences) == 1
await client.aclose()
```

TITO needs `transformers` (`pip install "agentbox-rl[hf]"`). It parses
`<tool_call>` blocks with AgentBox's lenient parser. The OpenRLHF adapter runs
TITO on OpenRLHF's own vLLM engine. Measured on Qwen3-1.7B: 10-turn episodes
gave 1 sequence and 0 breaks with TITO, against a break on nearly every turn
with chat completions.

## Other clients

| Client | Use |
| --- | --- |
| `MockModelClient([...ModelResponse])` | Scripted turns; a shared instance replays the script per conversation (deterministic under concurrency) |
| `FunctionModelClient(async_fn)` | `async_fn(messages, tools) -> ModelResponse`: write a policy in Python |
| `ReplayModelClient(trajectory)` | Replays the assistant turns of a recorded trajectory (`agentbox traj replay`) |
| `agentbox run --mock` | No-op policy for pipeline checks |

Any object with `async complete(messages, *, tools=None, tool_choice="auto", **kwargs)
-> ModelResponse` and a `model` attribute works as a model client.
