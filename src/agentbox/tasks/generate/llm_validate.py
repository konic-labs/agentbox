"""LLM-as-judge validation for generated tasks (DSPy structured outputs).

Intended to run against the same OpenAI-compatible teacher endpoint used for
generation (e.g. self-hosted Qwen3.6-27B on EC2 / vLLM).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from agentbox.config import ModelConfig
from agentbox.errors import TaskGenerationError
from agentbox.tasks.generate.lm import build_dspy_lm, build_openai_client
from agentbox.tasks.generate.validate import ValidationReport, task_payload_for_judge
from agentbox.tasks.schema import Task

logger = logging.getLogger("agentbox.generate.llm_validate")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "yes", "1", "accept", "accepted"}:
        return True
    if s in {"false", "no", "0", "reject", "rejected"}:
        return False
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        m = re.search(r"[-+]?\d*\.?\d+", str(value))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
        return default


def _prediction_to_report(
    task: Task,
    pred: Any,
    *,
    min_score: float,
) -> ValidationReport:
    accept = _as_bool(getattr(pred, "accept", False))
    score = _as_float(getattr(pred, "score", 0.0))
    # clamp
    score = max(0.0, min(1.0, score))
    near = _as_bool(getattr(pred, "starter_is_near_solution", False))
    leak = _as_bool(getattr(pred, "description_leaks_fix", False))
    must_impl = _as_bool(getattr(pred, "agent_must_implement", True))
    reasons = str(getattr(pred, "reasons", "") or "")
    fixes = str(getattr(pred, "suggested_fixes", "") or "")

    errors: list[str] = []
    warnings: list[str] = []
    ok = accept and score >= min_score and not near and not leak and must_impl
    if not accept:
        errors.append(f"llm_judge reject: {reasons or 'accept=false'}")
    if score < min_score:
        errors.append(f"llm_judge score {score:.2f} < min_score {min_score:.2f}")
    if near:
        errors.append("llm_judge: starter_is_near_solution=true")
    if leak:
        errors.append("llm_judge: description_leaks_fix=true")
    if not must_impl:
        errors.append("llm_judge: agent_must_implement=false (one-token patch task)")

    if ok and reasons:
        warnings.append(f"llm_judge notes: {reasons}")

    return ValidationReport(
        ok=ok,
        task=task,
        errors=errors,
        warnings=warnings,
        llm_ok=ok,
        llm_score=score,
        llm_accept=accept,
        llm_reasons=reasons,
        details={
            "llm_judge": {
                "accept": accept,
                "score": score,
                "starter_is_near_solution": near,
                "description_leaks_fix": leak,
                "agent_must_implement": must_impl,
                "reasons": reasons,
                "suggested_fixes": fixes,
                "min_score": min_score,
            }
        },
    )


async def validate_task_llm(
    task: Task,
    *,
    model_config: ModelConfig,
    min_score: float = 0.65,
    use_dspy: bool = True,
    domain: str = "python",
    dspy_lm: Any | None = None,
    dspy_module: Any | None = None,
) -> ValidationReport:
    """Run DSPy (or OpenAI JSON fallback) structured judge on a Task."""
    payload = task_payload_for_judge(task)
    difficulty = str(
        (task.metadata or {}).get("difficulty")
        or payload.get("difficulty")
        or "unknown"
    )
    claimed_domain = str((task.metadata or {}).get("language") or domain)

    if use_dspy:
        try:
            return await _validate_dspy(
                task,
                model_config=model_config,
                payload=payload,
                difficulty=difficulty,
                claimed_domain=claimed_domain,
                min_score=min_score,
                dspy_lm=dspy_lm,
                dspy_module=dspy_module,
            )
        except Exception as exc:
            logger.warning("DSPy LLM judge failed (%s); trying OpenAI JSON", exc)

    return await _validate_openai_json(
        task,
        model_config=model_config,
        payload=payload,
        difficulty=difficulty,
        claimed_domain=claimed_domain,
        min_score=min_score,
    )


async def _validate_dspy(
    task: Task,
    *,
    model_config: ModelConfig,
    payload: dict[str, str],
    difficulty: str,
    claimed_domain: str,
    min_score: float,
    dspy_lm: Any | None = None,
    dspy_module: Any | None = None,
) -> ValidationReport:
    import dspy

    from agentbox.tasks.generate.signatures import get_validator_module

    lm = dspy_lm or build_dspy_lm(model_config)
    module = dspy_module or get_validator_module()()

    def _run() -> Any:
        with dspy.context(lm=lm):
            return module(
                task_id=payload["task_id"],
                description=payload["description"],
                starter_files_json=payload["starter_files_json"],
                setup_commands_json=payload["setup_commands_json"],
                verifier_json=payload["verifier_json"],
                difficulty=difficulty,
                claimed_domain=claimed_domain,
            )

    pred = await asyncio.to_thread(_run)
    return _prediction_to_report(task, pred, min_score=min_score)


async def _validate_openai_json(
    task: Task,
    *,
    model_config: ModelConfig,
    payload: dict[str, str],
    difficulty: str,
    claimed_domain: str,
    min_score: float,
) -> ValidationReport:
    client = build_openai_client(model_config)
    system = (
        "You are a strict judge of coding tasks for software agents. "
        "Reply with a single JSON object only (no markdown) with keys: "
        "accept (bool), score (0..1 float), starter_is_near_solution (bool), "
        "description_leaks_fix (bool), agent_must_implement (bool), "
        "reasons (string), suggested_fixes (string). "
        "REJECT near-complete starters with only a tiny intentional bug; "
        "REJECT solution leaks in description/comments; "
        "ACCEPT stubs/incomplete code where the agent must implement logic."
    )
    user = (
        f"task_id={payload['task_id']}\n"
        f"difficulty={difficulty}\n"
        f"domain={claimed_domain}\n"
        f"description:\n{payload['description']}\n\n"
        f"starter_files_json:\n{payload['starter_files_json']}\n\n"
        f"setup_commands_json:\n{payload['setup_commands_json']}\n\n"
        f"verifier_json:\n{payload['verifier_json']}\n"
    )
    kwargs: dict[str, Any] = {
        "model": model_config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": min(model_config.temperature, 0.3),
        "max_tokens": model_config.max_tokens or 1024,
    }
    if model_config.extra_body:
        kwargs["extra_body"] = model_config.extra_body
    resp = await client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if isinstance(reasoning, str):
            content = reasoning.strip()
    data = _parse_json_loose(content)

    class _Pred:
        pass

    pred = _Pred()
    for k in (
        "accept",
        "score",
        "starter_is_near_solution",
        "description_leaks_fix",
        "agent_must_implement",
        "reasons",
        "suggested_fixes",
    ):
        setattr(pred, k, data.get(k))
    return _prediction_to_report(task, pred, min_score=min_score)


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise TaskGenerationError("empty LLM judge response")
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise TaskGenerationError("LLM judge response was not valid JSON")


def merge_validation_reports(
    *reports: ValidationReport,
    task: Task | None = None,
) -> ValidationReport:
    """AND-combine QC stages (docker + llm, etc.)."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    ok = True
    seed_ok = None
    verifier_runs = None
    fails_starter = None
    llm_ok = None
    llm_score = None
    llm_accept = None
    llm_reasons = None
    final_task = task

    for r in reports:
        ok = ok and r.ok
        errors.extend(r.errors)
        warnings.extend(r.warnings)
        details.update(r.details or {})
        if r.task is not None:
            final_task = r.task
        if r.seed_ok is not None:
            seed_ok = r.seed_ok if seed_ok is None else (seed_ok and r.seed_ok)
        if r.verifier_runs is not None:
            verifier_runs = r.verifier_runs
        if r.verifier_fails_on_starter is not None:
            fails_starter = r.verifier_fails_on_starter
        if r.llm_ok is not None:
            llm_ok = r.llm_ok
        if r.llm_score is not None:
            llm_score = r.llm_score
        if r.llm_accept is not None:
            llm_accept = r.llm_accept
        if r.llm_reasons:
            llm_reasons = r.llm_reasons

    return ValidationReport(
        ok=ok,
        task=final_task,
        errors=errors,
        warnings=warnings,
        seed_ok=seed_ok,
        verifier_runs=verifier_runs,
        verifier_fails_on_starter=fails_starter,
        llm_ok=llm_ok,
        llm_score=llm_score,
        llm_accept=llm_accept,
        llm_reasons=llm_reasons,
        details=details,
    )
