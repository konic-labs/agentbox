# Models

AgentBox speaks **OpenAI Chat Completions + tool calling only**. Any compatible
server works: cloud providers, vLLM, SGLang, Ollama, LocalAI, llama.cpp server.

## Clients

| Class | Use |
| --- | --- |
| `OpenAICompatClient` | Production path via official `openai.AsyncOpenAI` |
| `MockModelClient` | Scripted responses for tests / hermetic demos |

Protocol: `ModelClient.complete(messages, *, tools, tool_choice, **kwargs) -> ModelResponse`.

## Configuration

```python
from agentbox.config import ModelConfig
from agentbox.model import OpenAICompatClient

client = OpenAICompatClient(ModelConfig(
    model="glm-5.2",
    base_url="https://api.featherless.ai/v1",
    api_key="...",
    temperature=0.7,
    max_tokens=4096,
    timeout_s=120,
    extra_body={},  # provider-specific
))
```

Via `Agent`:

```python
from agentbox import Agent

agent = Agent(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # optional for many local servers
)
```

## Provider matrix

| Type | Examples | Config |
| --- | --- | --- |
| External | Featherless, OpenRouter, Together, Fireworks, OpenAI | `base_url` + `api_key` |
| Local | vLLM, SGLang, Ollama (`/v1`), LocalAI | `base_url`; `api_key` often unused |

### Examples

```python
# Ollama
Agent(model="qwen2.5-coder:7b", base_url="http://localhost:11434/v1", api_key="ollama")

# vLLM (must enable tool calling — see matrix below)
Agent(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")

# Featherless
Agent(model="glm-5.2", base_url="https://api.featherless.ai/v1", api_key="...")
```

## Tool calling (required for agent rollouts)

AgentBox agents always send OpenAI **tools** with `tool_choice="auto"`.  
Endpoints that reject that return `final_status=error` with **0 steps**.

| Server | Typical requirement |
| --- | --- |
| **vLLM** | `--enable-auto-tool-choice` and `--tool-call-parser <parser>` (e.g. `hermes`, `qwen3_coder`) |
| **Ollama** | Model with tools capability (e.g. recent Qwen chat models) |
| **Cloud OpenAI-compat** | Provider must support function/tool calling |

Probe before a long bench:

```bash
agentbox doctor \
  --model /data/models/unsloth/Qwen3.6-27B-NVFP4 \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY
```

Python:

```python
from agentbox.config import ModelConfig
from agentbox.model.probe import probe_endpoint

results = await probe_endpoint(ModelConfig(
    model="…", base_url="http://localhost:8000/v1", api_key="EMPTY"
))
assert all(r.ok for r in results)
```

## Request / response shape

Request (conceptual):

```json
{
  "model": "glm-5.2",
  "messages": [ ... ],
  "tools": [ ... ],
  "tool_choice": "auto",
  "temperature": 0.7
}
```

Assistant tool call:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_abc",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\": \"main.py\"}"
      }
    }
  ]
}
```

Tool result:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": "..."
}
```

## ModelResponse fields

- `content`, `tool_calls`, `finish_reason`
- `reasoning_content` (if provider exposes it)
- `usage` token counts
- `raw_choice` serialized provider choice

## Errors

`ModelError` wraps API failures. Tool-calling related 400s attempt clearer
messages when the body mentions tools/functions.

## ART / existing AsyncOpenAI

```python
client = OpenAICompatClient.from_async_openai(
    existing_async_openai_client,
    model=model_name,
)
```

Useful when integrating with OpenPipe ART’s `TrainableModel.openai_client()`.

## Mock client

```python
from agentbox.model import MockModelClient, ModelResponse
from agentbox.trajectory.schema import ToolCall, FunctionCall

mock = MockModelClient([
    ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="c1", function=FunctionCall(
            name="list_files", arguments="{}"
        ))],
    ),
    ModelResponse(content="done", tool_calls=[]),
])
```

Each `complete` call consumes the next scripted response. Use a **fresh** mock
per parallel rollout (stateful index).

## Non-goals

- Claude Messages API adapters  
- Raw XML / ReAct string parsers  
- Built-in training / weight updates  

Those remain outside AgentBox; export trajectories instead.
