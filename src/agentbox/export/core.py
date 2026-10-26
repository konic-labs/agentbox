"""Exporter registry and built-in formats."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel, Field

from agentbox.rl.stats import group_advantages, is_zero_variance
from agentbox.trajectory.schema import Trajectory
from agentbox.trajectory.tokens import iter_turn_samples, trace_from_messages
from agentbox.types import MessageRole


class ExportFilter(BaseModel):
    trainable_only: bool = True  # drop infra failures (GH #40)
    successful_only: bool = False
    min_reward: float | None = None
    max_steps: int | None = None
    task_ids: set[str] | None = None

    def accept(self, t: Trajectory) -> bool:
        if self.trainable_only and not t.is_trainable:
            return False
        if self.successful_only and not t.success:
            return False
        if self.min_reward is not None and t.reward < self.min_reward:
            return False
        if self.max_steps is not None and t.metrics.steps > self.max_steps:
            return False
        return not (self.task_ids is not None and t.task_id not in self.task_ids)


class ExportStats(BaseModel):
    format: str
    path: str
    read: int = 0
    kept: int = 0
    rows: int = 0
    skipped_groups: int = 0
    notes: list[str] = Field(default_factory=list)


Exporter = Callable[[Iterable[Trajectory], IO[str], ExportStats, dict[str, Any]], None]
_REGISTRY: dict[str, Exporter] = {}


def register_exporter(name: str) -> Callable[[Exporter], Exporter]:
    def deco(fn: Exporter) -> Exporter:
        _REGISTRY[name] = fn
        return fn

    return deco


def available_formats() -> list[str]:
    return sorted(_REGISTRY)


def _open_out(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8")
    return path.open("w", encoding="utf-8")


def export(
    trajectories: Iterable[Trajectory],
    fmt: str,
    out: str | Path,
    *,
    filter: ExportFilter | None = None,  # noqa: A002
    **options: Any,
) -> ExportStats:
    """Stream ``trajectories`` through a filter into ``fmt`` rows at ``out``."""
    if fmt not in _REGISTRY:
        raise KeyError(f"unknown export format {fmt!r}; available: {available_formats()}")
    flt = filter or ExportFilter()
    stats = ExportStats(format=fmt, path=str(out))

    def filtered() -> Iterator[Trajectory]:
        for t in trajectories:
            stats.read += 1
            if flt.accept(t):
                stats.kept += 1
                yield t

    if fmt == "hf":  # writes parquet itself
        _REGISTRY[fmt](filtered(), None, stats, options)  # type: ignore[arg-type]
        return stats
    with _open_out(Path(out)) as fh:
        _REGISTRY[fmt](filtered(), fh, stats, options)
    return stats


def _write(fh: IO[str], row: dict[str, Any], stats: ExportStats) -> None:
    fh.write(json.dumps(row, ensure_ascii=False))
    fh.write("\n")
    stats.rows += 1


def chat_messages(t: Trajectory, *, include_reasoning: bool = False) -> list[dict[str, Any]]:
    return [m.to_openai_dict(include_reasoning=include_reasoning) for m in t.messages]


def _meta(t: Trajectory) -> dict[str, Any]:
    return {
        "task_id": t.task_id,
        "rollout_id": t.rollout_id,
        "group_id": t.group_id,
        "reward": t.reward,
        "status": t.final_status.value,
        "steps": t.metrics.steps,
        "model": t.model,
    }


def _groups(trajs: Iterable[Trajectory]) -> Iterator[list[Trajectory]]:
    """Group by group_id (or task_id); materialises one run's groups at a time."""
    groups: dict[str, list[Trajectory]] = {}
    for t in trajs:
        groups.setdefault(t.group_id or t.task_id, []).append(t)
    for g in groups.values():
        yield sorted(g, key=lambda t: t.sample_index)


@register_exporter("chat")
def _chat(
    trajs: Iterable[Trajectory], fh: IO[str], stats: ExportStats, opts: dict[str, Any]
) -> None:
    """``{"messages", "tools", "metadata"}`` rows (TRL SFTTrainer conversational format)."""
    include_reasoning = bool(opts.get("include_reasoning", False))
    for t in trajs:
        row: dict[str, Any] = {"messages": chat_messages(t, include_reasoning=include_reasoning)}
        if t.tools and not opts.get("drop_tools", False):
            row["tools"] = t.tools
        row["metadata"] = _meta(t)
        _write(fh, row, stats)


@register_exporter("sharegpt")
def _sharegpt(
    trajs: Iterable[Trajectory], fh: IO[str], stats: ExportStats, opts: dict[str, Any]
) -> None:
    names = {"system": "system", "user": "human", "assistant": "gpt", "tool": "tool"}
    for t in trajs:
        convo = []
        for m in t.messages:
            role = m.role.value if isinstance(m.role, MessageRole) else str(m.role)
            value = m.content or ""
            if m.tool_calls:
                calls = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in m.tool_calls
                ]
                value = (value + "\n" if value else "") + json.dumps(calls, ensure_ascii=False)
            convo.append({"from": names.get(role, role), "value": value})
        _write(fh, {"conversations": convo, "tools": t.tools or [], "metadata": _meta(t)}, stats)


def _prompt_split(t: Trajectory) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    msgs = chat_messages(t)
    idx = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), len(msgs))
    return msgs[:idx], msgs[idx:]


@register_exporter("dpo")
def _dpo(
    trajs: Iterable[Trajectory], fh: IO[str], stats: ExportStats, opts: dict[str, Any]
) -> None:
    """Best vs worst trajectory per group: ``{"prompt", "chosen", "rejected"}`` (TRL DPO)."""
    min_margin = float(opts.get("min_margin", 0.5))
    for group in _groups(trajs):
        if len(group) < 2:
            stats.skipped_groups += 1
            continue
        best = max(group, key=lambda t: t.reward)
        worst = min(group, key=lambda t: t.reward)
        if best.reward - worst.reward < min_margin:
            stats.skipped_groups += 1
            continue
        prompt, chosen = _prompt_split(best)
        _, rejected = _prompt_split(worst)
        row = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "tools": best.tools or [],
            "metadata": {
                "task_id": best.task_id,
                "chosen_reward": best.reward,
                "rejected_reward": worst.reward,
            },
        }
        _write(fh, row, stats)


@register_exporter("grpo")
def _grpo(
    trajs: Iterable[Trajectory], fh: IO[str], stats: ExportStats, opts: dict[str, Any]
) -> None:
    """One row per group with rewards and group-relative advantages."""
    method = opts.get("advantage", "grpo")
    drop_zero = bool(opts.get("drop_zero_variance", False))
    for group in _groups(trajs):
        rewards = [t.reward for t in group]
        if drop_zero and is_zero_variance(rewards):
            stats.skipped_groups += 1
            continue
        advantages = group_advantages(rewards, method=method)
        samples = [
            {"messages": chat_messages(t), "reward": t.reward, "advantage": a, "meta": _meta(t)}
            for t, a in zip(group, advantages, strict=True)
        ]
        row = {
            "group_id": group[0].group_id or group[0].task_id,
            "task_id": group[0].task_id,
            "tools": group[0].tools or [],
            "rewards": rewards,
            "advantages": advantages,
            "samples": samples,
        }
        _write(fh, row, stats)


@register_exporter("tokens")
def _tokens(
    trajs: Iterable[Trajectory], fh: IO[str], stats: ExportStats, opts: dict[str, Any]
) -> None:
    """Exact token data: one row per sequence (``per_turn=True``: one row per turn)."""
    per_turn = bool(opts.get("per_turn", False))
    missing = 0
    for t in trajs:
        base = {
            "task_id": t.task_id,
            "rollout_id": t.rollout_id,
            "group_id": t.group_id,
            "reward": t.reward,
        }
        if per_turn:
            turns = [
                getattr(m, "tokens", None)
                for m in t.messages
                if (m.role.value if isinstance(m.role, MessageRole) else m.role) == "assistant"
            ]
            samples = list(iter_turn_samples(turns))
            if not samples:
                missing += 1
            for i, (prompt, completion, logprobs) in enumerate(samples):
                _write(
                    fh,
                    {
                        **base,
                        "turn": i,
                        "prompt_ids": prompt,
                        "completion_ids": completion,
                        "logprobs": logprobs,
                    },
                    stats,
                )
            continue
        trace = trace_from_messages(t.messages)
        if trace is None:
            missing += 1
            continue
        for i, seq in enumerate(trace.sequences):
            _write(
                fh,
                {
                    **base,
                    "sequence": i,
                    "prompt_ids": seq.prompt_ids,
                    "response_ids": seq.response_ids,
                    "response_mask": seq.response_mask,
                    "response_logprobs": seq.response_logprobs,
                    "num_turns": seq.num_turns,
                    "exact": trace.exact,
                },
                stats,
            )
    if missing:
        stats.notes.append(
            f"{missing} trajectories had no token data (enable ModelConfig.capture_tokens)"
        )


@register_exporter("art")
def _art(
    trajs: Iterable[Trajectory], fh: IO[str], stats: ExportStats, opts: dict[str, Any]
) -> None:
    from agentbox.integrations.art import to_art_dict

    for t in trajs:
        _write(fh, to_art_dict(t), stats)


@register_exporter("hf")
def _hf(
    trajs: Iterable[Trajectory], fh: IO[str] | None, stats: ExportStats, opts: dict[str, Any]
) -> None:
    """Parquet via HF ``datasets`` (``push_to_hub=<repo>`` uploads it)."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("the 'hf' format needs `pip install agentbox-rl[hf]`") from exc

    def rows() -> Iterator[dict[str, Any]]:
        for t in trajs:
            stats.rows += 1
            yield {
                "messages": json.dumps(chat_messages(t), ensure_ascii=False),
                "tools": json.dumps(t.tools or [], ensure_ascii=False),
                **_meta(t),
            }

    ds = Dataset.from_generator(rows)
    ds.to_parquet(stats.path)
    repo = opts.get("push_to_hub")
    if repo:
        ds.push_to_hub(repo, private=bool(opts.get("private", True)))
        stats.notes.append(f"pushed to https://huggingface.co/datasets/{repo}")
